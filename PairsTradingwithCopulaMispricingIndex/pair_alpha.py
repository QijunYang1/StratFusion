#region imports
from AlgorithmImports import *

from collections import deque
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kendalltau
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import stats
import numpy as np
import sys
from copula_lib import *

#endregion

class TechPairsAlphaModel(AlphaModel):

    window = {} # stores historical price used to calculate trading day's stock return
    coef = 0    # to be calculated: requested ratio of quantity_u / quantity_v
    day = 0     # keep track of current day for daily rebalance
    month = 0   # keep track of current month for monthly recalculation of optimal trading pair

    pair = []  # stores the selected trading pair

    duration = timedelta(15)
    count = 0

    def __init__(self, lookback_days, num_days, weight_v):
        self.lookback_days = lookback_days   # length of history data in trading period
        self.num_days = num_days             # length of formation period which determine the copula we use    
        self.enter = 0.5         # floor confidence level
        self.exit = 2
        self.weight_v = weight_v             # desired holding weight of asset v in the portfolio, adjusted to avoid insufficient buying power
        self.flag_u_v = 0
        self.flag_v_u = 0
        
    def _begin_warm_up(self, algorithm):
            return algorithm.Time >= algorithm.StartDate - timedelta(30)
     
    def Update(self, algorithm: QCAlgorithm, slice: Slice) -> List[Insight]:

        insights = []
        self.all_pair = {}

        if not self._begin_warm_up(algorithm):
            return insights

        if self.count == 0:
            for kvp in algorithm.UniverseManager:
                universe = kvp.Value
                if 'MANUAL' in kvp.Key.Value:
                    for kvp2 in universe.Members:
                        symbol = kvp2.Key
                        security = kvp2.Value
                        if str(symbol) == 'AAPL' or str(symbol) == 'META':
                            self.window[security.Symbol] = deque(maxlen = 2)
                            security.consolidator = TradeBarConsolidator(timedelta(1))
                            security.consolidator.DataConsolidated += lambda _, consolidated_bar: self.window[consolidated_bar.Symbol].append(consolidated_bar.Close)
                            algorithm.SubscriptionManager.AddConsolidator(security.Symbol, security.consolidator)
                        if str(symbol) == 'AAPL':
                            self.all_pair[str(symbol)] = symbol
                        elif str(symbol) == 'META':
                            self.all_pair[str(symbol)] = symbol
                        
                        if len(list(self.window.keys())) == 2:
                            self.pair = list(self.window.keys())
                            history = algorithm.History(self.pair, 2, Resolution.Hour)
                            history = history.close.unstack(level=0)
                            for symbol in self.window:
                                for i in range(2):
                                    self.window[symbol].append(history[str(symbol)][i])
                                    
            self.pair = [self.all_pair['AAPL'],self.all_pair['META']]
            algorithm.Debug('alpha1 :'+str(self.pair[0])+str(self.pair[1]))
            self.count = 1
        
        insights = []


        self.SetSignal(algorithm, slice)     # only executed at first day of each month

        # Daily rebalance
        if algorithm.Time.day == self.day or slice.QuoteBars.Count == 0:
            return []

        long, short = self.pair[0], self.pair[1]

        if len(self.window[long]) < 2 or len(self.window[short]) < 2:
            return []

        # Compute the mispricing indices for u and v by using estimated copula
        MI_u_v, MI_v_u = misprice_index(self.window,self.pair,self.theta,self.copula,self.ecdf_x,self.ecdf_y)

        if math.isnan(MI_u_v):
            MI_u_v = 0.5
        if math.isnan(MI_v_u):
            MI_v_u = 0.5

        self.flag_u_v += MI_u_v - 0.5
        self.flag_v_u += MI_v_u - 0.5

        # Placing orders: if long is relatively underpriced, buy the pair
        if self.flag_u_v <= - self.exit or self.flag_v_u >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ]) 

        if self.flag_v_u <= -self.exit or self.flag_u_v >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ])
        self.beta = 1/(1+self.coef)

        if self.flag_u_v <= - self.enter and self.flag_v_u >= self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Up, weight=self.weight_v*self.beta * self.coef),
            ])
    
        # Placing orders: if short is relatively underpriced, sell the pair
        
        elif self.flag_u_v >= self.enter and self.flag_v_u <= -self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta*self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price ),
            ])

        self.day = algorithm.Time.day

        algorithm.Plot('Price',self.pair[0],self.window[self.pair[0]][-1])
        algorithm.Plot('Price',self.pair[1],self.window[self.pair[1]][-1])

        algorithm.Plot('Flag','flag_u_v1',self.flag_u_v)
        algorithm.Plot('Flag','flag_v_u1',self.flag_v_u)

        return insights

    def SetSignal(self, algorithm, slice):
        '''Computes the mispricing indices to generate the trading signals.
        It's called on first day of each month'''

        if algorithm.Time.month == self.month:
            return
        
        ## Compute the best copula

        # Pull historical log returns used to determine copula
        history = algorithm.History(self.pair, self.num_days, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.ScaledRaw)
        if history.empty:
            return
            
        history = history.close.unstack(level=0)
        logreturns = (np.log(history) - np.log(history.shift(1))).dropna()
        x, y = logreturns[str(self.pair[0])], logreturns[str(self.pair[1])]

        # Convert the two returns series to two uniform values u and v using the empirical distribution functions
        # params_x,params_y = stats.t.fit(x),stats.t.fit(y)
        # dist_x,dist_y = stats.t(*params_x),stats.t(*params_y)
        ecdf_x, ecdf_y  = ECDF(x), ECDF(y)

        u, v = [ecdf_x(a) for a in x], [ecdf_x(a) for a in y]

        # Compute the Akaike Information Criterion (AIC) for different copulas and choose copula with minimum AIC
        tau = kendalltau(x, y)[0]  # estimate Kendall'rank correlation

        AIC ={}  # generate a dict with key being the copula family, value = [theta, AIC]
        
        for i in ['clayton', 'gumbel','frank']:
            param = set_parameter(i, tau)
            lpdf = [lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
            # Replace nan with zero and inf with finite numbers in lpdf list
            lpdf = np.nan_to_num(lpdf)
            
            loglikelihood = sum(lpdf)
            AIC[i] = [param, -2 * loglikelihood + 2]
        
        # Choose the copula with the minimum AIC
        self.copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        
        ## Compute the signals
        
        # Generate the log return series of the selected trading pair
        logreturns = logreturns.tail(self.lookback_days)
        x, y = logreturns[str(self.pair[0])], logreturns[str(self.pair[1])]
        
        # Estimate Kendall'rank correlation
        tau = kendalltau(x, y)[0] 
        
        # Estimate the copula parameter: theta
        self.theta = set_parameter(self.copula,tau)
        
        # Simulate the empirical distribution function for returns of selected trading pair

        self.ecdf_x, self.ecdf_y  = ECDF(x), ECDF(y)
        # self.ecdf_x, self.ecdf_y  = dist_x.cdf, dist_y.cdf

        # Run linear regression over the two history return series and return the desired trading size ratio
        x, y = history[str(self.pair[0])].apply(lambda x: np.log(x)), history[str(self.pair[1])].apply(lambda x: np.log(x))
        self.coef = stats.linregress(x,y).slope
        
        self.month = algorithm.Time.month

class FinSerPairsAlphaModel(AlphaModel):

    window = {} # stores historical price used to calculate trading day's stock return
    coef = 0    # to be calculated: requested ratio of quantity_u / quantity_v
    day = 0     # keep track of current day for daily rebalance
    month = 0   # keep track of current month for monthly recalculation of optimal trading pair
    pair = []  # stores the selected trading pair
    count = 0
    duration = timedelta(15)

    def __init__(self, lookback_days, num_days, weight_v):
        self.lookback_days = lookback_days   # length of history data in trading period
        self.num_days = num_days             # length of formation period which determine the copula we use               # cap confidence level
        self.enter = 0.5 #flag_enter                 # floor confidence level
        self.exit =  2 #flag_exit
        self.weight_v = weight_v             # desired holding weight of asset v in the portfolio, adjusted to avoid insufficient buying power
        self.flag_u_v = 0
        self.flag_v_u = 0
        
    def _begin_warm_up(self, algorithm):
            return algorithm.Time >= algorithm.StartDate - timedelta(30)
        
    def Update(self, algorithm: QCAlgorithm, slice: Slice) -> List[Insight]:
        insights = []
        self.all_pair = {}

        if not self._begin_warm_up(algorithm):
            return insights

        if self.count == 0:
            for kvp in algorithm.UniverseManager:
                universe = kvp.Value
                if 'MANUAL' in kvp.Key.Value:
                    for kvp2 in universe.Members:
                        symbol = kvp2.Key
                        security = kvp2.Value

                        if str(symbol) == 'WFC' or str(symbol) == 'USB':
                            self.window[security.Symbol] = deque(maxlen = 2)
                            security.consolidator = TradeBarConsolidator(timedelta(1))
                            security.consolidator.DataConsolidated += lambda _, consolidated_bar: self.window[consolidated_bar.Symbol].append(consolidated_bar.Close)
                            algorithm.SubscriptionManager.AddConsolidator(security.Symbol, security.consolidator)
                        if str(symbol) == 'WFC':
                            self.all_pair[str(symbol)] = symbol
                        elif str(symbol) == 'USB':
                            self.all_pair[str(symbol)] = symbol
                        
                        if len(list(self.window.keys())) == 2:
                            self.select_pair = list(self.window.keys())
                            history = algorithm.History(self.select_pair, 2, Resolution.Hour)
                            history = history.close.unstack(level=0)
                            for symbol in self.window:
                                for i in range(2):
                                    self.window[symbol].append(history[str(symbol)][i])

            self.select_pair = [self.all_pair['WFC'],self.all_pair['USB']]
            algorithm.Debug('alpha2 :'+str(self.select_pair[0])+str(self.select_pair[1]))
            self.count = 1
    
        self.SetSignal(algorithm, slice)     # only executed at first day of each month
        # Daily rebalance
        if algorithm.Time.day == self.day or slice.QuoteBars.Count == 0:
            return []
        
        long, short = self.select_pair[0],self.select_pair[1]
        
        if len(self.window[long]) < 2 or len(self.window[short]) < 2:
            return []
            
        # Compute the mispricing indices for u and v by using estimated copula
        MI_u_v, MI_v_u = misprice_index(self.window,self.select_pair,self.theta,self.copula,self.ecdf_x,self.ecdf_y)

        if math.isnan(MI_u_v):
            MI_u_v = 0.5
        if math.isnan(MI_v_u):
            MI_v_u = 0.5

        self.flag_u_v += MI_u_v - 0.5
        self.flag_v_u += MI_v_u - 0.5

        # Placing orders: if long is relatively underpriced, buy the pair
        if self.flag_u_v <= - self.exit or self.flag_v_u >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ]) 

        if self.flag_v_u <= -self.exit or self.flag_u_v >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ])


        self.beta = 1/(1+self.coef)
        if self.flag_u_v <= - self.enter and self.flag_v_u >= self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Up, weight=self.weight_v * self.beta* self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price),
            ])
            # * algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price

        # Placing orders: if short is relatively underpriced, sell the pair
        # elif MI_u_v >= self.cap_CL and MI_v_u <= self.floor_CL:
        
        elif self.flag_u_v >= self.enter and self.flag_v_u <= -self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta*self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price ),
            ])

            #* self.coef *
        self.day = algorithm.Time.day

        algorithm.Plot('Price',self.select_pair[0],self.window[self.select_pair[0]][-1])
        algorithm.Plot('Price',self.select_pair[1],self.window[self.select_pair[1]][-1])
        algorithm.Plot('Flag','flag_u_v2',self.flag_u_v)
        algorithm.Plot('Flag','flag_v_u2',self.flag_v_u)

        return insights

    def SetSignal(self, algorithm, slice):
        '''Computes the mispricing indices to generate the trading signals.
        It's called on first day of each month'''

        if algorithm.Time.month == self.month:
            return
        
        ## Compute the best copula

        # Pull historical log returns used to determine copula
        history = algorithm.History(self.select_pair, self.num_days, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.ScaledRaw)
        if history.empty:
            return
            
        history = history.close.unstack(level=0)
        logreturns = (np.log(history) - np.log(history.shift(1))).dropna()
        x, y = logreturns[str(self.select_pair[0])], logreturns[str(self.select_pair[1])]

        # Convert the two returns series to two uniform values u and v using the empirical distribution functions
        # params_x,params_y = stats.t.fit(x),stats.t.fit(y)
        # dist_x,dist_y = stats.t(*params_x),stats.t(*params_y)
        ecdf_x, ecdf_y  = ECDF(x), ECDF(y)

        u, v = [ecdf_x(a) for a in x], [ecdf_x(a) for a in y]

        # Compute the Akaike Information Criterion (AIC) for different copulas and choose copula with minimum AIC
        tau = kendalltau(x, y)[0]  # estimate Kendall'rank correlation

        AIC ={}  # generate a dict with key being the copula family, value = [theta, AIC]
        
        for i in ['clayton', 'gumbel','frank']:
            param = set_parameter(i, tau)
            lpdf = [lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
            # Replace nan with zero and inf with finite numbers in lpdf list
            lpdf = np.nan_to_num(lpdf)
            
            loglikelihood = sum(lpdf)
            AIC[i] = [param, -2 * loglikelihood + 2]
        
        # Choose the copula with the minimum AIC
        self.copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        
        ## Compute the signals
        
        # Generate the log return series of the selected trading pair
        logreturns = logreturns.tail(self.lookback_days)
        x, y = logreturns[str(self.select_pair[0])], logreturns[str(self.select_pair[1])]
        
        # Estimate Kendall'rank correlation
        tau = kendalltau(x, y)[0] 
        
        # Estimate the copula parameter: theta
        self.theta = set_parameter(self.copula,tau)
        
        # Simulate the empirical distribution function for returns of selected trading pair

        self.ecdf_x, self.ecdf_y  = ECDF(x), ECDF(y)
        # self.ecdf_x, self.ecdf_y  = dist_x.cdf, dist_y.cdf

        
        # Run linear regression over the two history return series and return the desired trading size ratio
        x, y = history[str(self.select_pair[0])].apply(lambda x: np.log(x)), history[str(self.select_pair[1])].apply(lambda x: np.log(x))
        self.coef = stats.linregress(x,y).slope
        
        self.month = algorithm.Time.month

class HealthCarePairsAlphaModel(AlphaModel):

    window = {} # stores historical price used to calculate trading day's stock return
    coef = 0    # to be calculated: requested ratio of quantity_u / quantity_v
    day = 0     # keep track of current day for daily rebalance
    month = 0   # keep track of current month for monthly recalculation of optimal trading pair
    pair = []  # stores the selected trading pair
    count = 0
    duration = timedelta(15)

    def __init__(self, lookback_days, num_days,weight_v):
        self.lookback_days = lookback_days   # length of history data in trading period
        self.num_days = num_days             # length of formation period which determine the copula we use               # cap confidence level
        self.enter = 0.5                # floor confidence level
        self.exit = 3.5
        self.weight_v = weight_v             # desired holding weight of asset v in the portfolio, adjusted to avoid insufficient buying power
        self.flag_u_v = 0
        self.flag_v_u = 0
        
    def _begin_warm_up(self, algorithm):
            return algorithm.Time >= algorithm.StartDate - timedelta(30)

    def Update(self, algorithm: QCAlgorithm, slice: Slice) -> List[Insight]:
        insights = []
        self.all_pair = {}

        if not self._begin_warm_up(algorithm):
            return insights

        if self.count == 0:
            for kvp in algorithm.UniverseManager:
                universe = kvp.Value
                if 'MANUAL' in kvp.Key.Value:
                    for kvp2 in universe.Members:
                        symbol = kvp2.Key
                        security = kvp2.Value
                        
                        if str(symbol) == 'ABBV' or str(symbol) == 'BMY':
                            self.window[security.Symbol] = deque(maxlen = 2)
                            security.consolidator = TradeBarConsolidator(timedelta(1))
                            security.consolidator.DataConsolidated += lambda _, consolidated_bar: self.window[consolidated_bar.Symbol].append(consolidated_bar.Close)
                            algorithm.SubscriptionManager.AddConsolidator(security.Symbol, security.consolidator)
                        if str(symbol) == 'ABBV':
                            self.all_pair[str(symbol)] = symbol
                        elif str(symbol) == 'BMY':
                            self.all_pair[str(symbol)] = symbol
                        
                        if len(list(self.window.keys())) == 2:
                            self.select_pair = list(self.window.keys())
                            history = algorithm.History(self.select_pair, 2, Resolution.Hour)
                            history = history.close.unstack(level=0)
                            for symbol in self.window:
                                for i in range(2):
                                    self.window[symbol].append(history[str(symbol)][i])

            self.select_pair = [self.all_pair['ABBV'],self.all_pair['BMY']]
            algorithm.Debug('alpha 3:'+str(self.select_pair[0])+str(self.select_pair[1]))
            self.count = 1
        
        self.SetSignal(algorithm, slice)     # only executed at first day of each month
        # Daily rebalance

        if algorithm.Time.day == self.day or slice.QuoteBars.Count == 0:
            return []
        
        long, short = self.select_pair[0],self.select_pair[1]
        
        if len(self.window[long]) < 2 or len(self.window[short]) < 2:
            return []
            
        # Compute the mispricing indices for u and v by using estimated copula
        MI_u_v, MI_v_u = misprice_index(self.window,self.select_pair,self.theta,self.copula,self.ecdf_x,self.ecdf_y)

        if math.isnan(MI_u_v):
            MI_u_v = 0.5
        if math.isnan(MI_v_u):
            MI_v_u = 0.5

        self.flag_u_v += MI_u_v - 0.5
        self.flag_v_u += MI_v_u - 0.5

        # Placing orders: if long is relatively underpriced, buy the pair
        if self.flag_u_v <= - self.exit or self.flag_v_u >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ]) 

        if self.flag_v_u <= -self.exit or self.flag_u_v >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ])
        self.beta = 1/(1+self.coef)

        if self.flag_u_v <= - self.enter and self.flag_v_u >= self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Up, weight=self.weight_v * self.beta* self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price),
            ])
            # * algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price

        # Placing orders: if short is relatively underpriced, sell the pair
        # elif MI_u_v >= self.cap_CL and MI_v_u <= self.floor_CL:
        
        elif self.flag_u_v >= self.enter and self.flag_v_u <= -self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta*self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price ),
            ])

            #* self.coef *
        self.day = algorithm.Time.day

        algorithm.Plot('Price',self.select_pair[0],self.window[self.select_pair[0]][-1])
        algorithm.Plot('Price',self.select_pair[1],self.window[self.select_pair[1]][-1])
        algorithm.Plot('Flag','flag_u_v3',self.flag_u_v)
        algorithm.Plot('Flag','flag_v_u3',self.flag_v_u)

        return insights

    def SetSignal(self, algorithm, slice):
        '''Computes the mispricing indices to generate the trading signals.
        It's called on first day of each month'''

        if algorithm.Time.month == self.month:
            return
        
        ## Compute the best copula

        # Pull historical log returns used to determine copula
        history = algorithm.History(self.select_pair, self.num_days, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.ScaledRaw)
        if history.empty:
            return
            
        history = history.close.unstack(level=0)
        logreturns = (np.log(history) - np.log(history.shift(1))).dropna()
        x, y = logreturns[str(self.select_pair[0])], logreturns[str(self.select_pair[1])]

        # Convert the two returns series to two uniform values u and v using the empirical distribution functions
        # params_x,params_y = stats.t.fit(x),stats.t.fit(y)
        # dist_x,dist_y = stats.t(*params_x),stats.t(*params_y)
        ecdf_x, ecdf_y  = ECDF(x), ECDF(y)

        u, v = [ecdf_x(a) for a in x], [ecdf_x(a) for a in y]

        # Compute the Akaike Information Criterion (AIC) for different copulas and choose copula with minimum AIC
        tau = kendalltau(x, y)[0]  # estimate Kendall'rank correlation

        AIC ={}  # generate a dict with key being the copula family, value = [theta, AIC]
        
        for i in ['clayton', 'gumbel','frank']:
            param = set_parameter(i, tau)
            lpdf = [lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
            # Replace nan with zero and inf with finite numbers in lpdf list
            lpdf = np.nan_to_num(lpdf)
            
            loglikelihood = sum(lpdf)
            AIC[i] = [param, -2 * loglikelihood + 2]
        
        # Choose the copula with the minimum AIC
        self.copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        
        ## Compute the signals
        
        # Generate the log return series of the selected trading pair
        logreturns = logreturns.tail(self.lookback_days)
        x, y = logreturns[str(self.select_pair[0])], logreturns[str(self.select_pair[1])]
        
        # Estimate Kendall'rank correlation
        tau = kendalltau(x, y)[0] 
        
        # Estimate the copula parameter: theta
        self.theta = set_parameter(self.copula,tau)
        
        # Simulate the empirical distribution function for returns of selected trading pair

        self.ecdf_x, self.ecdf_y  = ECDF(x), ECDF(y)
        # self.ecdf_x, self.ecdf_y  = dist_x.cdf, dist_y.cdf

        
        # Run linear regression over the two history return series and return the desired trading size ratio
        x, y = history[str(self.select_pair[0])].apply(lambda x: np.log(x)), history[str(self.select_pair[1])].apply(lambda x: np.log(x))
        self.coef = stats.linregress(x,y).slope
        
        self.month = algorithm.Time.month

class BasicMaterialPairsAlphaModel(AlphaModel):

    window = {} # stores historical price used to calculate trading day's stock return
    coef = 0    # to be calculated: requested ratio of quantity_u / quantity_v
    day = 0     # keep track of current day for daily rebalance
    month = 0   # keep track of current month for monthly recalculation of optimal trading pair
    pair = []  # stores the selected trading pair
    count = 0
    duration = timedelta(15)

    def __init__(self, lookback_days, num_days,weight_v):
        self.lookback_days = lookback_days   # length of history data in trading period
        self.num_days = num_days             # length of formation period which determine the copula we use               # cap confidence level
        self.enter = 0.5               # floor confidence level
        self.exit = 2
        self.weight_v = weight_v             # desired holding weight of asset v in the portfolio, adjusted to avoid insufficient buying power
        self.flag_u_v = 0
        self.flag_v_u = 0

    def _begin_warm_up(self, algorithm):
            return algorithm.Time >= algorithm.StartDate - timedelta(30)

    def Update(self, algorithm: QCAlgorithm, slice: Slice) -> List[Insight]:
        insights = []
        self.all_pair = {}

        if not self._begin_warm_up(algorithm):
            return insights

        if self.count == 0:
            for kvp in algorithm.UniverseManager:
                universe = kvp.Value
                if 'MANUAL' in kvp.Key.Value:
                    for kvp2 in universe.Members:
                        symbol = kvp2.Key
                        security = kvp2.Value
                        
                        if str(symbol) == 'PPG' or str(symbol) == 'CRHCY':
                            self.window[security.Symbol] = deque(maxlen = 2)
                            security.consolidator = TradeBarConsolidator(timedelta(1))
                            security.consolidator.DataConsolidated += lambda _, consolidated_bar: self.window[consolidated_bar.Symbol].append(consolidated_bar.Close)
                            algorithm.SubscriptionManager.AddConsolidator(security.Symbol, security.consolidator)
                        if str(symbol) == 'PPG':
                            self.all_pair[str(symbol)] = symbol
                        elif str(symbol) == 'CRHCY':
                            self.all_pair[str(symbol)] = symbol
                        
                        if len(list(self.window.keys())) == 2:
                            self.select_pair = list(self.window.keys())
                            history = algorithm.History(self.select_pair, 2, Resolution.Hour)
                            history = history.close.unstack(level=0)
                            for symbol in self.window:
                                for i in range(2):
                                    self.window[symbol].append(history[str(symbol)][i])

            self.select_pair = [self.all_pair['PPG'],self.all_pair['CRHCY']]
            algorithm.Debug('alpha 4:'+str(self.select_pair[0])+str(self.select_pair[1]))
            self.count = 1
        
        self.SetSignal(algorithm, slice)     # only executed at first day of each month
        # Daily rebalance
        if algorithm.Time.day == self.day or slice.QuoteBars.Count == 0:
            return []
        
        long, short = self.select_pair[0],self.select_pair[1]
        
        if len(self.window[long]) < 2 or len(self.window[short]) < 2:
            return []
            
        # Compute the mispricing indices for u and v by using estimated copula
        MI_u_v, MI_v_u = misprice_index(self.window,self.select_pair,self.theta,self.copula,self.ecdf_x,self.ecdf_y)

        if math.isnan(MI_u_v):
            MI_u_v = 0.5
        if math.isnan(MI_v_u):
            MI_v_u = 0.5

        self.flag_u_v += MI_u_v - 0.5
        self.flag_v_u += MI_v_u - 0.5

        # Placing orders: if long is relatively underpriced, buy the pair
        if self.flag_u_v <= - self.exit or self.flag_v_u >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ]) 

        if self.flag_v_u <= -self.exit or self.flag_u_v >= self.exit:

            self.flag_v_u = 0
            self.flag_u_v = 0

            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=0),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=0),
            ])
            
        self.beta = 1/(1+self.coef)

        if self.flag_u_v <= - self.enter and self.flag_v_u >= self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Up, weight=self.weight_v * self.beta* self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price),
            ])
            # * algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price

        # Placing orders: if short is relatively underpriced, sell the pair
        # elif MI_u_v >= self.cap_CL and MI_v_u <= self.floor_CL:
        
        elif self.flag_u_v >= self.enter and self.flag_v_u <= -self.enter:
            insights.extend([
                Insight.Price(short, self.duration, InsightDirection.Up, weight=self.weight_v*self.beta),
                Insight.Price(long, self.duration, InsightDirection.Down, weight=self.weight_v*self.beta*self.coef*algorithm.Portfolio[long].Price / algorithm.Portfolio[short].Price ),
            ])

            #* self.coef *
        self.day = algorithm.Time.day

        algorithm.Plot('Price',self.select_pair[0],self.window[self.select_pair[0]][-1])
        algorithm.Plot('Price',self.select_pair[1],self.window[self.select_pair[1]][-1])
        algorithm.Plot('Flag','flag_u_v4',self.flag_u_v)
        algorithm.Plot('Flag','flag_v_u4',self.flag_v_u)

        return insights

    def SetSignal(self, algorithm, slice):
        '''Computes the mispricing indices to generate the trading signals.
        It's called on first day of each month'''

        if algorithm.Time.month == self.month:
            return
        
        ## Compute the best copula

        # Pull historical log returns used to determine copula
        history = algorithm.History(self.select_pair, self.num_days, Resolution.Hour, dataNormalizationMode=DataNormalizationMode.ScaledRaw)
        if history.empty:
            return
            
        history = history.close.unstack(level=0)
        logreturns = (np.log(history) - np.log(history.shift(1))).dropna()
        x, y = logreturns[str(self.select_pair[0])], logreturns[str(self.select_pair[1])]

        # Convert the two returns series to two uniform values u and v using the empirical distribution functions
        # params_x,params_y = stats.t.fit(x),stats.t.fit(y)
        # dist_x,dist_y = stats.t(*params_x),stats.t(*params_y)
        ecdf_x, ecdf_y  = ECDF(x), ECDF(y)

        u, v = [ecdf_x(a) for a in x], [ecdf_x(a) for a in y]

        # Compute the Akaike Information Criterion (AIC) for different copulas and choose copula with minimum AIC
        tau = kendalltau(x, y)[0]  # estimate Kendall'rank correlation

        AIC ={}  # generate a dict with key being the copula family, value = [theta, AIC]
        
        for i in ['clayton', 'gumbel','frank']:
            param = set_parameter(i, tau)
            lpdf = [lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
            # Replace nan with zero and inf with finite numbers in lpdf list
            lpdf = np.nan_to_num(lpdf)
            
            loglikelihood = sum(lpdf)
            AIC[i] = [param, -2 * loglikelihood + 2]
        
        # Choose the copula with the minimum AIC
        self.copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        
        ## Compute the signals
        
        # Generate the log return series of the selected trading pair
        logreturns = logreturns.tail(self.lookback_days)
        x, y = logreturns[str(self.select_pair[0])], logreturns[str(self.select_pair[1])]
        
        # Estimate Kendall'rank correlation
        tau = kendalltau(x, y)[0] 
        
        # Estimate the copula parameter: theta
        self.theta = set_parameter(self.copula,tau)
        
        # Simulate the empirical distribution function for returns of selected trading pair

        self.ecdf_x, self.ecdf_y  = ECDF(x), ECDF(y)
        # self.ecdf_x, self.ecdf_y  = dist_x.cdf, dist_y.cdf

        
        # Run linear regression over the two history return series and return the desired trading size ratio
        x, y = history[str(self.select_pair[0])].apply(lambda x: np.log(x)), history[str(self.select_pair[1])].apply(lambda x: np.log(x))
        self.coef = stats.linregress(x,y).slope
        
        self.month = algorithm.Time.month
