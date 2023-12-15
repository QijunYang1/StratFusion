#region imports
from AlgorithmImports import *
#endregion
import torch
import torch.nn as nn

import torch.optim as optim


# A2C trading modle
class A2C_Trading(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, num_layers=2):
        super(A2C_Trading, self).__init__()

        # Actor
        
        self.actor_lstm = nn.LSTM(state_dim, hidden_size,num_layers=num_layers, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, action_dim)

        # Critic
        self.critic_lstm = nn.LSTM(state_dim, hidden_size,num_layers=num_layers, batch_first=True)
        self.critic_linear = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, state, hidden_actor=None, hidden_critic=None):
        # Actor forward pass
        if hidden_actor is None:
            hidden_actor = (torch.zeros(self.num_layers, self.hidden_size).detach(),
                            torch.zeros(self.num_layers, self.hidden_size).detach())
        if hidden_critic is None:
            hidden_critic = (torch.zeros(self.num_layers, self.hidden_size).detach(),
                             torch.zeros(self.num_layers, self.hidden_size).detach())

        lstm_out_actor, hidden_actor = self.actor_lstm(state, hidden_actor)
        
        action_mean = torch.tanh(self.actor_linear(lstm_out_actor[-1])) * 0.3
        if torch.isnan(action_mean).any():
            print(lstm_out_actor[-1])
        lstm_out_critic, hidden_critic = self.critic_lstm(state, hidden_critic)
        state_value = self.critic_linear(lstm_out_critic[-1])

        return action_mean, state_value, hidden_actor, hidden_critic


# Define Trading environment 

class StockTradingEnv:
    def __init__(self, features,price,end_step, initial_balance=100000,margin_req = 0.25, lookback_window_size=30):
        # Load the stock data from CSV
        self.stock_data = features #numstock X features X dates
        self.stock_price = price  #numstock X price X dates
        self.num_stocks = features.shape[0] # assuming the num_stocks is the first dim of features
        self.current_price = price
        self.end_step = end_step
        # Initialize parameters
        self.initial_balance = initial_balance
        self.old_value = self.initial_balance
        self.lookback_window_size = lookback_window_size
        
        self.margin_requirement = margin_req
        
        self.shares_owned = np.zeros(self.num_stocks)
        self.shares_shorted = np.zeros(self.num_stocks)
        self.total_portfolio_value = self.initial_balance
        self.borrowed = 0
        self.reset()

    
    def reset(self):
        # Reset the state to an initial state
        self.balance = self.initial_balance

        self.current_step = self.lookback_window_size

        self.shares_owned = np.zeros(self.num_stocks, dtype=float)
        self.shares_shorted = np.zeros(self.num_stocks, dtype=float)
        self.total_portfolio_value = self.initial_balance
        self.borrowed = 0
        # Get the initial state
        state = self._get_current_feature(self.stock_data, self.stock_price)

        return state

    def step(self, actions, curr_feature, curr_price):
        # Execute one time step within the environment
    
        # Execute trades
        self._execute_trades(actions)

        # Update to the next day
        self.current_step += 10
        done = self.current_step >= self.end_step - 10
        
        
        new_portfolio_value = self._calculate_portfolio_value()

        # Check for a margin call
        margin_call_triggered = self._is_margin_call()

        if margin_call_triggered:
            # Liquidate all positions in case of a margin call
            self.liquidate()
            new_portfolio_value = self._calculate_portfolio_value()
            reward = -1000
        # Calculate the reward as the change in portfolio value
        
        reward = new_portfolio_value - self.total_portfolio_value
        self.total_portfolio_value = new_portfolio_value
         # Update balance to new portfolio value
        
        next_state = self._get_current_feature(curr_feature, curr_price)
        

        return next_state, reward, done
    
    

    def _get_state(self, step):
        # Get the stock data for the past lookback_window_size days
        window_start = step - self.lookback_window_size
        window_end = step
        window_data = self.stock_data[window_start:window_end]


        # Flatten the window data and append the balance and portfolio
        state = window_data.flatten().tolist()
        state.append(self.balance)
        state.extend((self.shares_owned + self.shares_shorted).tolist())
        

        return np.array(state)
    
    def _get_current_feature(self, curr_feature, curr_price):
        # get feature and price
        self.current_price = curr_price
        state = curr_feature.flatten().tolist()
        state.append(self.balance)
        state.extend((self.shares_owned + self.shares_shorted).tolist())
        
        return np.array(state)
        
    
    def update_features(self, new_data, price):
        self.stock_data = new_data
        self.current_price = price
    
    
    def _calculate_portfolio_value(self):
        current_prices = self.current_price
        long_value = np.sum(self.shares_owned * current_prices)
        short_value = np.sum(self.shares_shorted * current_prices)
        return self.balance + long_value - short_value
    
    def _execute_trades(self, actions):
        current_prices = self.current_price
        # Normalize actions if their absolute sum exceeds 1
        total_action_intensity = np.sum(np.abs(actions))
        if total_action_intensity > 1:
            actions /= total_action_intensity
            
        # Calculate total buy cost and total sell proceeds
        total_buy_cost = 0
        total_sell_proceeds = 0
        for i, action in enumerate(actions):
            if action > 0:  # Buying
                shares_to_buy = np.floor(action * self.balance / current_prices[i])
                total_buy_cost += shares_to_buy * current_prices[i]
                self.shares_owned[i] += shares_to_buy
            elif action < 0:  # Selling
                shares_to_sell = np.floor(abs(action) * self.balance / current_prices[i])
                total_sell_proceeds += shares_to_sell * current_prices[i]
                self.shares_owned[i] -= shares_to_sell
        # Update balance
        if total_buy_cost > self.balance:  # Buying on margin
                self.borrowed += total_buy_cost - self.balance
                self.balance -= total_buy_cost
        else:
            self.balance -= total_buy_cost
        self.balance += total_sell_proceeds
        repayment = min(self.borrowed, total_sell_proceeds)
        self.borrowed -= repayment
        self.balance -= repayment
 
                
    def _is_margin_call(self):
        # calculate if margin call occurs
        current_prices = self.current_price

        long_position_value = np.sum(self.shares_owned * current_prices)
        short_position_value = np.sum(self.shares_shorted * current_prices)

        # Total portfolio value: cash balance + long positions - short positions
        total_portfolio_value = self.balance + long_position_value - short_position_value

        equity = total_portfolio_value - self.borrowed

        return equity < self.margin_requirement * total_portfolio_value
    

    def liquidate(self):
        #current_prices = self.stock_price[:, self.current_step]
        current_prices = self.current_price
        # Liquidate long positions
        for i in range(self.num_stocks):
            self.balance += self.shares_owned[i] * current_prices[i]
            self.shares_owned[i] = 0

        # Cover short positions
        for i in range(self.num_stocks):
            self.balance -= self.shares_shorted[i] * current_prices[i]
            self.shares_shorted[i] = 0
            
        repayment = min(self.borrowed, self.balance)
        self.borrowed -= repayment
        self.balance -= repayment
        
        
    def render(self):
        # Render the environment to the screen
        # For simplicity, we'll just print the current balance and portfolio
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Position: {-self.shares_shorted + self.shares_owned}")
        print(f"Portfolio Value: {self.total_portfolio_value}")

