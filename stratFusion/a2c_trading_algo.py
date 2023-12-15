#region imports
from AlgorithmImports import *
#endregion
from A2C import A2C_Trading, StockTradingEnv
import joblib
import torch
class A2CTradingModel(AlphaModel):
    securities = []
    duration = timedelta(days=10)

    def __init__(self,algorithm, lookback=30, weight=1 ):
        
        self.lookback = lookback
       

        self.model_key = "A2Cmodel"
        file_name = algorithm.ObjectStore.GetFilePath(self.model_key)
        self.model = joblib.load(file_name)
        
        self.symbol_list = []
        
        self.weight = weight
   
        self.holdings = None
        self.last_holding = None
        self.stock_env = None
        self.hidden_critic = None
        self.hidden_actor = None
        self.state = None

        self.rsis = {}
        self.arma = {}
        self.rdvs = {}
        self.bbs = {}
        self.moms = {}
        self.current_step = -1
        self.indicator_history ={}
        
        self.update_day = 0
        self.trade_day = 0
        self.mod = 1


    def _begin_warm_up(self, algorithm):
        return algorithm.Time >= algorithm.StartDate - timedelta(40)


    def Update(self, algorithm: QCAlgorithm, slice: Slice) -> List[Insight]:

        if len(self.symbol_list) == 0:
            dic = {}
            symbols = ["JPM","BRK.B", "AAPL","AMZN","MSFT"]
            for kvp in algorithm.UniverseManager:
                universe = kvp.Value
                if 'MANUAL' in kvp.Key.Value:
                    for kvp2 in universe.Members:
                        symbol = kvp2.Key
                        security = kvp2.Value
                       
                        if str(symbol) in symbols:
                            dic[str(symbol)] = symbol
            self.holdings = np.zeros(len(symbols))
            self.last_holding = np.zeros(len(symbols))
            for sym in symbols:
                self.symbol_list.append(dic[sym])


            for symbol in self.symbol_list:
                
                self.rsis[symbol] = RelativeStrengthIndex(20)
                self.arma[symbol] = AutoRegressiveIntegratedMovingAverage(1,2,1,20, True)
                self.rdvs[symbol] = RelativeDailyVolume(10)
                self.bbs[symbol] = BollingerBands(20, 2)
                self.moms[symbol] = Momentum(20)
                self.indicator_history[symbol] = pd.DataFrame(columns = ['rsi','arma','volatility','rdv','bdw','perb','mom'])
        
        insights = []
       
        if not self._begin_warm_up(algorithm):
            
            return insights
        
        #algorithm.Log(str(self.indicator_history["AAPL"]))
        
        self.update_indicators(algorithm)
        #algorithm.Log(f'{self.indicator_history[self.symbol_list[-1]].dropna().head()}')
        #algorithm.Log(f'{len(self.indicator_history[self.symbol_list[-1]].dropna())}')
        
        if algorithm.IsWarmingUp:
            return insights
            

        current_features = []
        price = []
        for symbol in self.symbol_list:
            
            self.indicator_history[symbol] = self.indicator_history[symbol].dropna().iloc[-self.lookback:,:]
            current_features.append(self.indicator_history[symbol].sort_index(ascending=True).values.T)
            price.append(algorithm.Securities[symbol].Price)

        current_features = np.array(current_features)
        price = np.array(price)
        if current_features.shape[-1] < self.lookback:
           
            return insights
        self.vol = np.mean(current_features[:,1,1:] -current_features[:,1,:-1])

        if self.stock_env is None:
            self.stock_env = StockTradingEnv(features=current_features,
                price=price,
        end_step=20000,
        lookback_window_size=self.lookback,
        initial_balance = algorithm.Portfolio.Cash)
        
        
        
        insights = self.TradeByModel(algorithm, current_features, price)
        
        return insights


    def update_indicators(self, algorithm):
        if algorithm.Time.day == self.update_day:
            #algorithm.Log(f'{algorithm.Time}, current step:{self.current_step}')
            return
        if len(self.indicator_history[self.symbol_list[-1]]) ==0:
            inited = False
            history = algorithm.History(self.symbol_list, 45, Resolution.Daily)
            bar_history = algorithm.History[TradeBar](self.symbol_list, 25, Resolution.Daily)
            
        else:
            inited = True
            history = algorithm.History(self.symbol_list, 31, Resolution.Daily)
            bar_history = algorithm.History[TradeBar](self.symbol_list, 1, Resolution.Daily)
            

        for symbol in self.symbol_list:
            
            if not history.empty and 'close' in history.columns:
                for i, (time, row) in enumerate(history.loc[symbol].iterrows()):
                    #if i<20: continue
                    if inited and i <30:
                        continue
                    value = row.close
                    
                    try:
                        self.arma[symbol].Update(time, value)
                    except Exception as e:
                        algorithm.Debug(f'{symbol}, {time}, {value}')
                    self.bbs[symbol].Update(time, value)
                    self.moms[symbol].Update(time, value)
                    self.rsis[symbol].Update(time, value)
                    
                        
                    
        
                    if self.rsis[symbol].IsReady:
                        self.indicator_history[symbol].loc[time, 'rsi'] = self.rsis[symbol].Current.Value
            
                    if self.arma[symbol].IsReady:
                        self.indicator_history[symbol].loc[time, 'arma'] = self.arma[symbol].Current.Value
                    if self.bbs[symbol].IsReady:
                        self.indicator_history[symbol].loc[time, 'bdw'] = self.bbs[symbol].BandWidth.Current.Value
                        self.indicator_history[symbol].loc[time, 'perb'] = self.bbs[symbol].PercentB.Current.Value
        
                    if self.moms[symbol].IsReady:
                        self.indicator_history[symbol].loc[time, 'mom'] = self.moms[symbol].Current.Value
            log_return = history.loc[symbol]['close'].apply(lambda x:np.log(x)).diff().dropna()
            self.indicator_history[symbol].loc[time, 'volatility'] = np.std(log_return.values[-30:])
            log_return = history.loc[symbol]['close'].apply(lambda x:np.log(x)).diff().dropna()
            volatility  = log_return.rolling(30).std().dropna()
            
            for i, (time,row) in enumerate(volatility.iteritems()):
                
                self.indicator_history[symbol].loc[time, 'volatility'] = row
            

            for i, bar in enumerate(bar_history):
                bar = bar.get(symbol)
                if bar:
                    self.rdvs[symbol].Update(bar)
                    if self.rdvs[symbol].IsReady:
                        self.indicator_history[symbol].loc[bar.Time, 'rdv'] = self.rdvs[symbol].Current.Value
                       
        self.update_day = algorithm.Time.day
            #algorithm.Log(f"bar time:{bar.Time} hist time: {time}")


    def TradeByModel(self, algorithm, curr_feature, curr_price):
        if algorithm.Time.day == self.trade_day:
            
            return []
        insights= []
       
        mod = 1
       
        self.current_step += 1
        self.trade_day = algorithm.Time.day
        
        if self.current_step % 10 != 0:

            return insights
        self.mod = 1 if self.vol>0 and self.vol<4  else 10
        torch.manual_seed(217540012958050124) #3632672893473863851
        
        torch.random.manual_seed(11249560557726264621)
        #torch.manual_seed(3632672893473863851)
        done = False
        std_dev =torch.tensor([0.02])
        alpha = 0.95
        gamma = 0.99
        epsilon = 0.01
        for i, symbol in enumerate(self.symbol_list):
            self.holdings[i] = algorithm.Securities[symbol].Holdings.HoldingsValue/algorithm.Portfolio.TotalPortfolioValue
            
        
        
        if self.state is None:
            self.state = self.stock_env.reset()
        

                # Convert state to appropriate tensor format for the model
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        
        # Get action probabilities and state value from the model
        
       
        action_mean, state_value, self.hidden_actor, self.hidden_critic = self.model(state_tensor, self.hidden_actor, self.hidden_critic)
        
        self.hidden_actor = (self.hidden_actor[0].detach(), self.hidden_actor[1].detach())
        self.hidden_critic = (self.hidden_critic[0].detach(), self.hidden_critic[1].detach())
        # Sample action from the probability distribution

        action = torch.normal(action_mean, std_dev)  
        action = action.clamp(-0.25, 0.25)
        self.stock_env.balance = algorithm.Portfolio.MarginRemaining  # Update cash

        # Take action in the environment
        #print(action)
        if torch.isnan(action).any():
            
            action_np =np.zeros(action.detach().numpy().shape)
        else:
            action_np = action.detach().numpy()
        
        next_state, reward, done = self.stock_env.step(action_np,curr_feature, curr_price)
        if done:
            self.Liquidate()
        # Execute trading 
       
        for i, act in enumerate(action_np):
            symbol = self.symbol_list[i]
            if act > 0:  # Buying
                margin = algorithm.Portfolio.MarginRemaining
                shares_to_buy = act * margin / algorithm.Portfolio.TotalPortfolioValue
                self.holdings[i]+=shares_to_buy 
                
                
            elif act < 0:  # Selling
                margin = algorithm.Portfolio.MarginRemaining
                shares_to_sell = abs(act) * margin / algorithm.Portfolio.TotalPortfolioValue
                self.holdings[i]-= shares_to_sell
                        
        gap_thresh = 0.01
        for i, symbol in enumerate(self.symbol_list):
            gap = abs(self.holdings[i] - self.last_holding[i])
            
            if self.holdings[i] > 0 and gap > gap_thresh :
                insights.append(Insight.Price(symbol, self.duration, InsightDirection.Up, weight=self.weight * self.holdings[i],confidence=self.weight))
                self.last_holding[i] = self.holdings[i]
            elif self.holdings[i] < 0  and gap > gap_thresh:
                insights.append(Insight.Price(symbol, self.duration, InsightDirection.Down, weight=self.weight * self.holdings[i],confidence=self.weight))
                self.last_holding[i] = self.holdings[i]
                
                
        if algorithm.Portfolio.TotalUnrealisedProfit < 0:
            self.mod =10
        else:
            self.mod = 2
        
        
        
        # Transition to the next state
        self.state = next_state
        
        return insights
        

    