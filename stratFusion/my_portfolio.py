#region imports
from AlgorithmImports import *
#endregion


# Your New Python File

class MyPortfolioConstructionModel(InsightWeightingPortfolioConstructionModel):






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













