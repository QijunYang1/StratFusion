#region imports
from AlgorithmImports import *
from collections import deque
from pair_alpha import TechPairsAlphaModel, FinSerPairsAlphaModel, HealthCarePairsAlphaModel, BasicMaterialPairsAlphaModel
from portfolio_con import InsightWeightingPortfolioConstructionModel1
# from ETFUinverse import ETFConstituentsUniverseSelectionModel
#endregion

class CopulaPairsTradingAlgorithm(QCAlgorithm):

    undesired_symbols_from_previous_deployment = []
    checked_symbols_from_previous_deployment = False
    previous_expiry_time = None

    def Initialize(self):
        self.SetStartDate(2023,12,5) 
        self.SetEndDate(2023,12,14)
        self.SetCash(10000000)

        # Margin Model Setting
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.SetSecurityInitializer(MySecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))

        # self.SetSecurityInitializer(self.CustomSecurityInitializer)
        # Benchmark Model
        # self.bench.SetFeeModel(CustomFeeModel(self))

        # ----------------- Pairs Trading Strategy ----------------------------------
        lookback_days = self.GetParameter("lookback_days", 250)
        pairs_tickers =  ['AAPL','META','WFC','USB','PPG','CRHCY','ABBV','BMY']
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in pairs_tickers]
        self.AddUniverseSelection(ManualUniverseSelectionModel(symbols))

        # Tech Sector: AAPL META; # FinService Sector: WFC USB;         
        # HealthCare Sector: ABBV BMY; # Basic Material Sector: PPG CRHCY
        # weight_pairs represents the portfolio weights on the Pairs Trading Strategy
        
        weight_pairs = 1
        num_days = 1000

        self.AddAlpha(TechPairsAlphaModel(lookback_days,self.GetParameter("num_days",num_days),
            self.GetParameter("weight_v",weight_pairs)
        ))
        self.AddAlpha(FinSerPairsAlphaModel(lookback_days,self.GetParameter("num_days", num_days),
            self.GetParameter("weight_v",weight_pairs)
        ))

        self.AddAlpha(HealthCarePairsAlphaModel(lookback_days,self.GetParameter("num_days", num_days),
            self.GetParameter("weight_v",weight_pairs)
            ))

        self.AddAlpha(BasicMaterialPairsAlphaModel( lookback_days, self.GetParameter("num_days", num_days),
            self.GetParameter("weight_v",weight_pairs)
        ))

        # ---------------- Portfolio Construction ----------------------------
        # self.Settings.RebalancePortfolioOnSecurityChanges = False
        # self.Settings.RebalancePortfolioOnInsightChanges = False
        # spy = Symbol.Create("SPY", SecurityType.Equity, Market.USA)
        # self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel('SPY'))
        # self.spy = self.AddEquity("SPY").Symbol
        # self.vix = self.AddIndex("VIX").Symbol
        
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        # self.rebalance_func
        self.SetExecution(ImmediateExecutionModel())
        self.SetWarmUp(timedelta(40))

    def rebalance_func(self, time):
        # Rebalance when all of the following are true:
        # - There are new insights or old insights have been cancelled
        # - The algorithm isn't warming up
        # - There is QuoteBar data in the current slice
        latest_expiry_time = sorted([insight.CloseTimeUtc for insight in self.Insights], reverse=True)[0] if self.Insights.Count else None
        if self.previous_expiry_time != latest_expiry_time and not self.IsWarmingUp and self.CurrentSlice.QuoteBars.Count > 0:
            self.previous_expiry_time = latest_expiry_time
            return time
        return None


# Outside of the algorithm class
class MySecurityInitializer(BrokerageModelSecurityInitializer):

    def __init__(self, brokerage_model: IBrokerageModel, security_seeder: ISecuritySeeder) -> None:
        super().__init__(brokerage_model, security_seeder)

    def Initialize(self, security: Security) -> None:
        # First, call the superclass definition
        # This method sets the reality models of each security using the default reality models of the brokerage model
        super().Initialize(security)

        # Next, overwrite some of the reality models        
        security.SetSlippageModel(VolumeShareSlippageModel())