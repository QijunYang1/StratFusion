from AlgorithmImports import *
from itertools import groupby

class InsightWeightingPortfolioConstructionModel1(PortfolioConstructionModel):
    '''Provides an implementation of IPortfolioConstructionModel that generates percent targets based on the
    Insight.Weight. The target percent holdings of each Symbol is given by the Insight.Weight from the last
    active Insight for that symbol.
    For insights of direction InsightDirection.Up, long targets are returned and for insights of direction
    InsightDirection.Down, short targets are returned.
    If the sum of all the last active Insight per symbol is bigger than 1, it will factor down each target
    percent holdings proportionally so the sum is 1.
    It will ignore Insight that have no Insight.Weight value.'''

    def __init__(self,rebalance = Resolution.Daily, portfolioBias = PortfolioBias.LongShort):
        '''Initialize a new instance of InsightWeightingPortfolioConstructionModel
        Args:
            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.
                              If None will be ignored.
                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.
                              The function returns null if unknown, in which case the function will be called again in the
                              next loop. Returning current time will trigger rebalance.
            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)'''
        super().__init__()

        self.portfolioBias = portfolioBias
        # If the argument is an instance of Resolution or Timedelta
        # Redefine rebalancingFunc
        rebalancingFunc = rebalance
        if isinstance(rebalance, int):
            rebalance = Extensions.ToTimeSpan(rebalance)
        if isinstance(rebalance, timedelta):
            rebalancingFunc = lambda dt: dt + rebalance
        if rebalancingFunc:
            self.SetRebalancingFunc(rebalancingFunc)

    def GetTargetInsights(self):
        # Get insight that haven't expired of each symbol that is still in the universe
        activeInsights = filter(self.ShouldCreateTargetForInsight,
            self.Algorithm.Insights.GetActiveInsights(self.Algorithm.UtcTime))
        # Get the last generated active insight for each symbol
        lastActiveInsights = []
        for sourceModel, f in groupby(sorted(activeInsights, key = lambda ff: ff.SourceModel), lambda fff: fff.SourceModel):
            for symbol, g in groupby(sorted(list(f), key = lambda gg: gg.Symbol), lambda ggg: ggg.Symbol):
                lastActiveInsights.append(sorted(g, key = lambda x: x.GeneratedTimeUtc)[-1])
        return lastActiveInsights
    
    def ShouldCreateTargetForInsight(self, insight):
        '''Method that will determine if the portfolio construction model should create a
        target for this insight
        Args:
            insight: The insight to create a target for'''
        # Ignore insights that don't have Weight value
        return insight.Weight is not None

    def DetermineTargetPercent(self, activeInsights):
        '''Will determine the target percent for each insight
        Args:
            activeInsights: The active insights to generate a target for'''
        
        Insights = filter(self.ShouldCreateTargetForInsight,
            self.Algorithm.Insights.GetActiveInsights(self.Algorithm.UtcTime))
        all_model_weight = {}
        pairs_active_insights = []
        a2c_active_insights = []
        pair_alpha_weight = None
        a2c_alpha_weight = None
        for sourceModel, f in groupby(sorted(Insights, key = lambda ff: ff.SourceModel), lambda fff: fff.SourceModel):
            # Treat PairsAlphaModel seperately
            if 'PairsAlphaModel' in sourceModel:
                for symbol, g in groupby(sorted(list(f), key = lambda gg: gg.Symbol), lambda ggg: ggg.Symbol):
                    pairs_active_insights.append(sorted(g, key = lambda x: x.GeneratedTimeUtc)[-1])
                    if pair_alpha_weight is None: pair_alpha_weight = pairs_active_insights[0].Confidence
            elif 'A2C' in sourceModel:
                for symbol, g in groupby(sorted(list(f), key = lambda gg: gg.Symbol), lambda ggg: ggg.Symbol):
                    a2c_active_insights.append(sorted(g, key = lambda x: x.GeneratedTimeUtc)[-1])
                    if a2c_alpha_weight is None: a2c_alpha_weight = a2c_active_insights[0].Confidence
            else:
                for symbol, g in groupby(sorted(list(f), key = lambda gg: gg.Symbol), lambda ggg: ggg.Symbol):
                    if symbol not in all_model_weight: all_model_weight[symbol] = 0 # intialization
                    for insight in g:
                        all_model_weight[symbol] += (insight.Direction if self.RespectPortfolioBias(insight) else InsightDirection.Flat) * self.GetValue(insight)
        
        pair_res = {}
        weightSums = sum(self.GetValue(insight) for insight in pairs_active_insights if self.RespectPortfolioBias(insight))
        weightFactor = 1.0
        if pair_alpha_weight is not None: 
            if weightSums > pair_alpha_weight:
                weightFactor = pair_alpha_weight / weightSums
            for insight in pairs_active_insights:
                if insight.Symbol not in pair_res: pair_res[insight.Symbol]=0
                pair_res[insight.Symbol] += (insight.Direction if self.RespectPortfolioBias(insight) else InsightDirection.Flat) * self.GetValue(insight) * weightFactor

        a2c_res = {}
        weightSums = sum(self.GetValue(insight) for insight in pairs_active_insights if self.RespectPortfolioBias(insight))
        weightFactor = 1.0
        if a2c_alpha_weight is not None: 
            if weightSums > a2c_alpha_weight:
                weightFactor = a2c_alpha_weight / weightSums
            for insight in a2c_active_insights:
                if insight.Symbol not in a2c_res: a2c_res[insight.Symbol]=0
                a2c_res[insight.Symbol] += (insight.Direction if self.RespectPortfolioBias(insight) else InsightDirection.Flat) * self.GetValue(insight) * weightFactor

        # Sum up
        for key in pair_res.keys():
            if key in all_model_weight:
                all_model_weight[key] += pair_res[key]
            else:
                all_model_weight[key] = pair_res[key]
        # Sum up
        for key in a2c_res.keys():
            if key in all_model_weight:
                all_model_weight[key] += a2c_res[key]
            else:
                all_model_weight[key] = a2c_res[key]

        return dict((insight, all_model_weight[insight.Symbol]) for insight in activeInsights)

    def GetValue(self, insight):
        '''Method that will determine which member will be used to compute the weights and gets its value
        Args:
            insight: The insight to create a target for
        Returns:
            The value of the selected insight member'''
        return abs(insight.Weight)

    def RespectPortfolioBias(self, insight):
        '''Method that will determine if a given insight respects the portfolio bias
        Args:
            insight: The insight to create a target for
        '''
        return self.portfolioBias == PortfolioBias.LongShort or insight.Direction == self.portfolioBias