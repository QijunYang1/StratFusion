# region imports
from AlgorithmImports import *
# from RegimeFactorVAE.Library import RegimeFactorVAE, calc_market_feeatures
from FeatureExtraction.Library import calc_features
from RegimeFactorVAE.vae_new import RegimeFactorVAE, calc_market_feeatures
import joblib
import torch
# endregion

class WellDressedRedDog(QCAlgorithm):

    def Initialize(self):
        # self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 2
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.SetStartDate(2023,3,1)
        self.SetEndDate(2023,10,10)
        self.SetCash(10000000)
        self.lookback = 20
        self.extra_lookback = 100
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.vix = self.AddIndex("VIX", Resolution.Daily).Symbol
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

        # Model Parameters
        C_stock = 184
        C_market = 39
        H_stock = 184
        H_market = 39
        num_market_feature = 20
        num_market_factor = 20
        num_stock_factor = 60
        num_market_portfolio = 20
        num_stock_portfolio = 60
        num_market_regime = 10
        beta = 0.7
        gru_num_layers = 3

        mfi_dim = 3
        time_length = 20

        # Load the model parameters
        self.model=RegimeFactorVAE(C_stock, C_market, H_stock, H_market, num_market_feature, num_stock_factor, num_market_factor, num_stock_portfolio, num_market_portfolio, time_length, num_market_regime, mfi_dim, beta, gru_num_layers)
        # epoch = 32 # FactorVAE1-1 -> 32 FactorVAE1-1-ohlcv -> 32
        # path_checkpoint = self.ObjectStore.GetFilePath("End2End/FactorVAE1-1/checkpoint_{}_epoch".format(epoch))
        epoch = 6
        path_checkpoint = self.ObjectStore.GetFilePath("End2End/RegimeFactorVAE-linear-stablization/checkpoint_{}_epoch".format(epoch))
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
        
        # Stock: Fetch historical data and calculate features
        history = self.History(symbol_list, self.lookback + self.extra_lookback, Resolution.Daily, dataNormalizationMode = DataNormalizationMode.Adjusted)
        history = calc_features(history)
        history.dropna(inplace=True)
        df = history.groupby('symbol').apply(lambda x: x.iloc[-self.lookback:].reset_index(level=0,drop=True)) # Select sequence for each stock
        seq_len = df.groupby('symbol').apply(lambda x: x.shape[0])
        stock_idx = seq_len[seq_len==self.lookback].index
        df = df.loc[stock_idx]
        dim = len(df.index.get_level_values(0).unique())
        data = df.to_numpy(dtype='float32').reshape((dim,self.lookback,-1))
        x_stock = torch.from_numpy(data).float()

        # Market: Fetch historical data and calculate features
        history_market = self.History([self.vix,self.spy], self.lookback + self.extra_lookback, Resolution.Daily, dataNormalizationMode = DataNormalizationMode.Adjusted) 
        volume_market = history_market.loc['SPY']['volume']
        close_market = pd.DataFrame(history_market['close'])
        close_market = close_market.unstack(level=0).droplevel(level=0,axis=1)
        close_market.columns = ['SPY','VIX']
        df_market = pd.concat([close_market,volume_market],axis=1)
        df_market = calc_market_feeatures(df_market)
        data_market = df_market.iloc[-self.lookback:].to_numpy()
        x_market = torch.from_numpy(data_market).float()
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            factor = self.model.predict(x_stock,x_market)
        factor = pd.Series(factor,index=stock_idx)

        # ----------- TopkDrop Strategy with equal weight -------------

        factor_sorted = factor.sort_values(ascending=False)
        stcok_rank_today = factor_sorted.index.values
        top_k = set(stcok_rank_today[:self.k])
        # self.Debug(f'Number of top_k: {len(top_k)}')
        if not self.last_top_k: # if not invested
            allocation = 1/self.k
            real_topk = top_k
            for stock in real_topk:
                self.SetHoldings(stock, allocation*self.port_propotion)
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

            allocation = 1/self.k
            real_topk = adjust_top_k.union(buy)
            orders = []
            for stock in real_topk:
                orders.append(PortfolioTarget(stock, allocation))
            self.SetHoldings(orders)

        #     self.Debug(f'Number of sold: {len(sold)}')
        # self.Debug(f'Number of current holdings: {len(real_topk)}')
        n_holding = 0
        for kvp in self.Portfolio:
            h = kvp.Value
            if h.Invested:
                n_holding+=1
        self.Debug(f'Portfolio holding count: {n_holding}')
        
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