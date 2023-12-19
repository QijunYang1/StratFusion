# region imports
from AlgorithmImports import *
from FactorVAE.Library import FactorVAE
from FeatureExtraction.Library import calc_features
import joblib
import torch
# endregion

class WellDressedRedDog(QCAlgorithm):

    def Initialize(self):
        # self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 2
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.SetStartDate(2020, 3, 1)
        self.SetEndDate(2020, 4, 1)
        self.SetCash(10000000)
        self.lookback = 20
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        universe = self.Universe.ETF("SPY", Market.USA)
        self.universe = self.AddUniverse(universe)  
        # Set Benchmark
        self.SetBenchmark("SPY")
        # Variable to hold the last calculated benchmark value
        self.lastBenchmarkValue = None
        # Our inital benchmark value scaled to match our portfolio
        self.BenchmarkPerformance = self.Portfolio.TotalPortfolioValue

        # TopkDrop parameters
        self.last_top_k = None
        self.last_bottom_k = None
        self.k = 50
        self.drop = 5
        self.port_propotion = 1

        # Initialize model parameters
        C = 184 # Dimension of characteristics.
        H = 184 # Dimension of hidden states.
        portfolio_num = 60  # Number of portfolios.
        factor_num = 60 # Number of contructed factors.
        time_length = 20

        # Load the model parameters
        self.model=FactorVAE(C, H, portfolio_num, factor_num, time_length)
        # epoch = 32 # FactorVAE1-1 -> 32 FactorVAE1-1-ohlcv -> 32
        # path_checkpoint = self.ObjectStore.GetFilePath("End2End/FactorVAE1-1/checkpoint_{}_epoch".format(epoch))
        epoch = 100
        path_checkpoint = self.ObjectStore.GetFilePath("End2End/FactorVAE_seed30/checkpoint_{}_epoch".format(epoch))
        # checkpoint = torch.load(checkpoint_path)
        checkpoint = joblib.load(path_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.SetWarmup(30,Resolution.Daily)

    def OnData(self, data: Slice):
        if self.IsWarmingUp: return

        # Select ETF constituents
        symbol_list = []
        universe_members = self.UniverseManager[self.universe.Configuration.Symbol].Members
        for kvp in universe_members:
            symbol = kvp.Key
            security = kvp.Value
            # if symbol in data.Bars:
            symbol_list.append(symbol)
        if len(symbol_list) == 0:
            self.Debug(f'{self.Time}: No Valid Symbols.')
            return
        
        # Fetch historical data and calculate features
        history = self.History(symbol_list, self.lookback + 100, Resolution.Daily, dataNormalizationMode = DataNormalizationMode.Adjusted)
        history = calc_features(history)
        history.dropna(inplace=True)
        df = history.groupby('symbol').apply(lambda x: x.iloc[-self.lookback:].reset_index(level=0,drop=True)) # Select sequence for each stock
        seq_len = df.groupby('symbol').apply(lambda x: x.shape[0])
        stock_idx = seq_len[seq_len==self.lookback].index
        df = df.loc[stock_idx]
        dim = len(df.index.get_level_values(0).unique())
        data = df.to_numpy(dtype='float32').reshape((dim,self.lookback,-1))
        x = torch.from_numpy(data)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            factor, risk = self.model.predict(x)
        factor = pd.Series(factor,index=stock_idx)
        risk =  pd.Series(risk,index=stock_idx)

        # # # ----------- TopkDrop Strategy with equal weight -------------

        # factor_sorted = factor.sort_values(ascending=False)
        # stcok_rank_today = factor_sorted.index.values
        # top_k = set(stcok_rank_today[:self.k])
        # # self.Debug(f'Number of top_k: {len(top_k)}')
        # if not self.last_top_k: # if not invested
        #     allocation = 1/self.k
        #     real_topk = top_k
        #     for stock in real_topk:
        #         self.SetHoldings(stock, allocation*self.port_propotion)
        # else: # already invested so we need to adjust weights
        #     sold = self.last_top_k - top_k # stock currently held and with a rank > k so we need to sold `drop` of them
        #     sold_first = sold - set(stcok_rank_today) # stock doesn't exist on rank considered to be sold first
        #     n_sold_first = len(sold_first) # number of stocks sold first
        #     if self.drop > n_sold_first:
        #         remaining_sold = sold - sold_first # the remaining to be sold
        #         remaining_sold_sorted = factor_sorted[list(remaining_sold)].sort_values() # get the rank of remaining to be sold
        #         n_still_needed_sold = self.drop - n_sold_first
        #         sold = set(remaining_sold_sorted.index.values[:n_still_needed_sold]).union(sold_first)
        #     else:
        #         sold = set(list(sold_first)[:self.drop])
        #     for stock in sold:
        #         self.Liquidate(stock)
        #     n = len(sold)
        #     buy = list(top_k - self.last_top_k) # Today's topk but not in holdings
        #     buy_sorted = factor_sorted[buy].sort_values(ascending=False) # sort
        #     buy = set(buy_sorted.index.values[:n]) # we need to buy those not in our portfolio
        #     adjust_top_k = self.last_top_k - sold # we need to adjust weights of those still sitting in our portfolio

        #     allocation = 1/self.k
        #     real_topk = adjust_top_k.union(buy)
        #     orders = []
        #     for stock in real_topk:
        #         orders.append(PortfolioTarget(stock, allocation))
        #     self.SetHoldings(orders)

        # #     self.Debug(f'Number of sold: {len(sold)}')
        # # self.Debug(f'Number of current holdings: {len(real_topk)}')
        # n_holding = 0
        # for kvp in self.Portfolio:
        #     h = kvp.Value
        #     if h.Invested:
        #         n_holding+=1
        # self.Debug(f'Portfolio holding count: {n_holding}')
        
        # # Update Topk
        # self.last_top_k = real_topk



        # ----------- TopkDrop Strategy with risk adjusted weights -------------

        factor_sorted = factor.sort_values(ascending=False)
        stcok_rank_today = factor_sorted.index.values
        top_k = set(stcok_rank_today[:self.k])
        if not self.last_top_k: # if not invested
            real_topk = top_k
            risk_top_k = 1/risk[list(real_topk)]
            allocation = risk_top_k/risk_top_k.sum()
            for stock in real_topk:
                self.SetHoldings(stock, allocation[stock]*self.port_propotion)
        else: # already invested so we need to adjust weights
            sold = self.last_top_k - top_k # stock currently held and with a rank > k so we need to sold `drop` of them
            sold_first = sold - set(stcok_rank_today) # stock doesn't exist on rank considered to be sold first
            n_sold_first = len(sold_first) # number of stocks sold first
            if self.drop > n_sold_first:
                remaining_sold = sold - sold_first # the remaining to be sold
                remaining_sold_sorted = factor_sorted[list(remaining_sold)].sort_values() # get the rank of remaining to be sold
                n_still_needed_sold = self.drop - n_sold_first
                sold = set(remaining_sold_sorted.index.values[:n_still_needed_sold]).union(sold_first)
            else:
                sold = set(list(sold_first)[:self.drop])
            for stock in sold:
                self.Liquidate(stock)
            n = len(sold)
            buy = list(top_k - self.last_top_k) # Today's topk but not in holdings
            buy_sorted = factor_sorted[buy].sort_values(ascending=False) # sort
            buy = set(buy_sorted.index.values[:n]) # we need to buy those not in our portfolio
            adjust_top_k = self.last_top_k - sold # we need to adjust weights of those still sitting in our portfolio

            real_topk = adjust_top_k.union(buy)
            risk_top_k = 1/risk[list(real_topk)]
            allocation = risk_top_k/risk_top_k.sum()
            orders = []
            for stock in real_topk:
                orders.append(PortfolioTarget(stock, allocation[stock]))
            self.SetHoldings(orders)
        # self.Log(f'Number of current holdings: {len(real_topk)}')
        # Update Topk
        self.last_top_k = real_topk


        # # ----------- LongShort Strategy with equal weight but dollar neutral -------------

        # factor_sorted = factor.sort_values(ascending=False)
        # stcok_rank_today = factor_sorted.index.values
        # top_k = set(stcok_rank_today[:self.k])
        # bottom_k = set(stcok_rank_today[-self.k:])
        # if not self.last_top_k: # if not invested
        #     allocation = 1/self.k
        #     # Long
        #     real_topk = top_k
        #     for stock in real_topk:
        #         self.SetHoldings(stock, allocation*0.5)
        #     # Short
        #     real_bottomk = bottom_k
        #     for stock in real_bottomk:
        #         self.SetHoldings(stock, -allocation*0.5)
        # else: # already invested so we need to adjust weights
        #     # Adjust Long Positions
        #     sold = self.last_top_k - top_k # stock currently held and with a rank > k so we need to sold `drop` of them
        #     sold_first = sold - set(stcok_rank_today) # stock doesn't exist on rank considered to be sold first
        #     n_sold_first = len(sold_first) # number of stocks sold first
        #     if self.drop > n_sold_first:
        #         remaining_sold = sold - sold_first # the remaining to be sold
        #         remaining_sold_sorted = factor_sorted[list(remaining_sold)].sort_values() # get the rank of remaining to be sold
        #         n_still_needed_sold = self.drop - n_sold_first
        #         sold = set(remaining_sold_sorted.index.values[:n_still_needed_sold]).union(sold_first)
        #     else:
        #         sold = set(list(sold_first)[:self.drop])
        #     for stock in sold:
        #         self.Liquidate(stock)
        #     n = len(sold)
        #     if n!=self.drop:
        #         self.Log(f'Drop does not excute properly!')
        #     buy = list(top_k - self.last_top_k) # Today's topk but not in holdings
        #     buy_sorted = factor_sorted[buy].sort_values(ascending=False) # sort
        #     buy = set(buy_sorted.index.values[:n]) # we need to buy those not in our portfolio
        #     adjust_top_k = self.last_top_k - sold # we need to adjust weights of those still sitting in our portfolio

        #     allocation = 1/self.k
        #     real_topk = adjust_top_k.union(buy)
        #     orders = []
        #     for stock in real_topk:
        #         orders.append(PortfolioTarget(stock, allocation*0.5))
            
        #     # Adjust Short Positions
        #     sold = self.last_bottom_k - bottom_k # stock currently held and with a rank < -k so we need to sold `drop` of them
        #     sold_first = sold - set(stcok_rank_today) # stock doesn't exist on rank considered to be sold first
        #     n_sold_first = len(sold_first) # number of stocks sold first
        #     if self.drop > n_sold_first:
        #         remaining_sold = sold - sold_first # the remaining to be sold
        #         remaining_sold_sorted = factor_sorted[list(remaining_sold)].sort_values(ascending=False) # get the rank of remaining to be sold
        #         n_still_needed_sold = self.drop - n_sold_first
        #         sold = set(remaining_sold_sorted.index.values[:n_still_needed_sold]).union(sold_first)
        #     else:
        #         sold = set(list(sold_first)[:self.drop])
        #     for stock in sold:
        #         self.Liquidate(stock)
        #     n = len(sold)
        #     if n!=self.drop:
        #         self.Log(f'Drop does not excute properly!')
        #     buy = list(bottom_k - self.last_bottom_k) # Today's bottomk but not in holdings
        #     buy_sorted = factor_sorted[buy].sort_values() # sort
        #     buy = set(buy_sorted.index.values[:n]) # we need to buy those not in our portfolio
        #     adjust_bottom_k = self.last_bottom_k - sold # we need to adjust weights of those still sitting in our portfolio

        #     allocation = 1/self.k
        #     real_bottomk = adjust_bottom_k.union(buy)
        #     for stock in real_bottomk:
        #         orders.append(PortfolioTarget(stock, -allocation*0.5))
            
        #     self.SetHoldings(orders)  
        # # self.Log(f'Number of current holdings: {len(real_topk)}')
        # n_holding = 0
        # for kvp in self.Portfolio:
        #     h = kvp.Value
        #     if h.Invested:
        #         n_holding+=1
        # self.Debug(f'Portfolio holding count: {n_holding}')
        # # Update Topk and Bottomk
        # self.last_top_k = real_topk
        # self.last_bottom_k = real_bottomk



        # --------- Plot the benchmark and our equity together -----------

        # store the current benchmark close price
        benchmark = self.Securities["SPY"].Close
        # Calculate the performance of our benchmark and update our benchmark value for plotting
        if self.lastBenchmarkValue is not  None:
           self.BenchmarkPerformance = self.BenchmarkPerformance * (benchmark/self.lastBenchmarkValue*self.port_propotion+1-self.port_propotion)
        # store today's benchmark close price for use tomorrow
        self.lastBenchmarkValue = benchmark
        # make our plots
        self.Plot("Strategy vs Benchmark", "Portfolio Value", self.Portfolio.TotalPortfolioValue)
        self.Plot("Strategy vs Benchmark", "Benchmark", self.BenchmarkPerformance)
        self.Plot("Excessive Performance", 'Performance', self.Portfolio.TotalPortfolioValue-self.BenchmarkPerformance)

        self.Log(f"{self.Portfolio.TotalPortfolioValue}")
        self.Debug(f"{self.Time}: {self.Portfolio.TotalPortfolioValue}")
        self.has_trade = True