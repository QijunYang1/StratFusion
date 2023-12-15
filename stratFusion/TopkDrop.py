#region imports
from AlgorithmImports import *
from RegimeFactorVAE import RegimeFactorVAE, calc_market_feeatures
from stratFusion.VAEStockFeatureExtractor import calc_features
import joblib
import torch

#endregion

class TopkDrop(AlphaModel):

    Name = "TopkDropAlphaModel"

    def __init__(self, weight):
        # TopkDrop parameters
        self.last_top_k = None
        self.last_bottom_k = None
        self.k = 50
        self.drop = 5

        # Load the model parameters
        self.epoch = 19
        self.load_flag = False

        # Stategy Parameters
        self.symbol_list = []
        self.lookback = 20
        self.extra_lookback = 100
        self.index_symbol = ['SPY','VIX']
        self.strat_wight = weight
        self.duration = timedelta(days = 1)

        # Variable to hold the last calculated benchmark value
        self.lastBenchmarkValue = None
        # Our inital benchmark value scaled to match our portfolio
        self.BenchmarkPerformance = None

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []

        if algorithm.IsWarmingUp: 
            algorithm.Debug('Warm')
            return insights
    
        if not (self.load_flag):
            # Initialize model parameters
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
            path_checkpoint = algorithm.ObjectStore.GetFilePath("End2End/RegimeFactorVAE-linear-stablization/checkpoint_{}_epoch".format(self.epoch))
            self.model = RegimeFactorVAE(C_stock, C_market, H_stock, H_market, num_market_feature, num_stock_factor, num_market_factor, num_stock_portfolio, num_market_portfolio, time_length, num_market_regime, mfi_dim, beta, gru_num_layers)
            checkpoint = joblib.load(path_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.load_flag = True

        if (algorithm.Time.hour != 9) or (algorithm.Time.minute != 40):
            return insights

        # Select ETF constituents
        symbol_list = []
        for symbol in self.symbol_list:
            if symbol in data.Bars:
                symbol_list.append(symbol)
        if len(symbol_list) == 0:
            algorithm.Debug(f'{algorithm.Time}: No Valid Symbols.')
            return insights
        
        # Stock: Fetch historical data and calculate features
        history = algorithm.History(symbol_list, self.lookback + self.extra_lookback, Resolution.Daily, dataNormalizationMode = DataNormalizationMode.Adjusted)
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
        history_market = algorithm.History(self.index_symbol, self.lookback + self.extra_lookback, Resolution.Daily, dataNormalizationMode = DataNormalizationMode.Adjusted) 
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
            factor, risk = self.model.predict(x_stock, x_market)
        factor = pd.Series(factor,index=stock_idx)
        risk =  pd.Series(risk,index=stock_idx)

        # ----------- TopkDrop Strategy with equal weight -------------
        factor_sorted = factor.sort_values(ascending=False)
        stcok_rank_today = factor_sorted.index.values
        top_k = set(stcok_rank_today[:self.k])
        if not self.last_top_k: # if not invested
            allocation = 1/self.k
            real_topk = top_k
            for stock in real_topk:
                insights.append(Insight.Price(stock, self.duration, InsightDirection.Up, weight=allocation * self.strat_wight))
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
                insights.append(Insight.Price(stock, self.duration , InsightDirection.Flat, weight=0))
            n = len(sold)
            buy = list(top_k - self.last_top_k) # Today's topk but not in holdings
            buy_sorted = factor_sorted[buy].sort_values(ascending=False) # sort
            buy = set(buy_sorted.index.values[:n]) # we need to buy those not in our portfolio
            adjust_top_k = self.last_top_k - sold # we need to adjust weights of those still sitting in our portfolio

            allocation = 1/self.k
            real_topk = adjust_top_k.union(buy)
            for stock in real_topk:
                insights.append(Insight.Price(stock, self.duration , InsightDirection.Up, weight=allocation * self.strat_wight))
        # Update Topk
        self.last_top_k = real_topk

        # --------- Plot the benchmark and our equity together -----------
        # store the current benchmark close price
        benchmark = algorithm.Securities["SPY"].Close
        # Calculate the performance of our benchmark and update our benchmark value for plotting
        if self.lastBenchmarkValue is not None:
           self.BenchmarkPerformance = self.BenchmarkPerformance * (benchmark/self.lastBenchmarkValue)
        # Initialize BenchmarkPerformance
        if self.BenchmarkPerformance is None:
            self.BenchmarkPerformance = algorithm.Portfolio.TotalPortfolioValue
        # store today's benchmark close price for use tomorrow
        self.lastBenchmarkValue = benchmark
        # make our plots
        algorithm.Plot("Strategy vs Benchmark", "Portfolio Value", algorithm.Portfolio.TotalPortfolioValue)
        algorithm.Plot("Strategy vs Benchmark", "Benchmark", self.BenchmarkPerformance)
        algorithm.Plot("Excessive Performance", 'Performance', algorithm.Portfolio.TotalPortfolioValue-self.BenchmarkPerformance)

        algorithm.Log(f"{algorithm.Portfolio.TotalPortfolioValue}")
        algorithm.Debug(f"{algorithm.Time}: {algorithm.Portfolio.TotalPortfolioValue}")

        return insights

    def OnSecuritiesChanged(self,algorithm: QCAlgorithm,changes: SecurityChanges) -> None:
        self.symbol_list = []
        for kvp in algorithm.UniverseManager:
            universe = kvp.Value
            if 'ETF' in kvp.Key.Value:
                for kvp2 in universe.Members:
                    symbol = kvp2.Key
                    security = kvp2.Value
                    self.symbol_list.append(symbol)