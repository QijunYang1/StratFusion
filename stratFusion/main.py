# region imports
from AlgorithmImports import *
from collections import deque
from pair_alpha import TechPairsAlphaModel, FinSerPairsAlphaModel, HealthCarePairsAlphaModel, BasicMaterialPairsAlphaModel
# from ETFUinverse import ETFConstituentsUniverseSelectionModel
#endregion

#region imports
from portfolio_con import InsightWeightingPortfolioConstructionModel1
from a2c_trading_algo import A2CTradingModel
from ETFUinverse import ETFConstituentsUniverseSelectionModel
from TopkDrop import TopkDrop
#endregion

class CopulaPairsTradingAlgorithm(QCAlgorithm):

    undesired_symbols_from_previous_deployment = []
    checked_symbols_from_previous_deployment = False
    previous_expiry_time = None
    
    def Initialize(self):
        
        self.SetStartDate(2016,1,1)
        self.SetEndDate(2017,1,1)
        
        self.SetCash(30000000)
        self.UniverseSettings.Resolution = Resolution.Minute

        # Margin Model Setting
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(MySecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))
        # self.SetSecurityInitializer(self.CustomSecurityInitializer)
        # Benchmark Model
        # self.bench.SetFeeModel(CustomFeeModel(self))

        # ----------------- Market augmentation ----------------------------------
        a2c_symbols = ["JPM","BRK.B", "AAPL","AMZN","MSFT"]
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in a2c_symbols]
        self.AddUniverseSelection(ManualUniverseSelectionModel(symbols))
        a2c_lookback = 30
        self.AddAlpha(A2CTradingModel(self, a2c_lookback, weight=1/3))
        # # ----------------- Pairs Trading Strategy ----------------------------------
        lookback_days = 250
        pairs_tickers =  ['AAPL','META','WFC','USB','ABBV','BMY','PPG','CRHCY']
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in pairs_tickers]
        self.AddUniverseSelection(ManualUniverseSelectionModel(symbols))

        # Tech Sector: AAPL META; 
        # FinService Sector: WFC USB; 
        # HealthCare Sector: ABBV BMY; 
        # Basic Material Sector: PPG CRHCY

        # weight_pairs represents the portfolio weights on the Pairs Trading Strategy
        weight_pairs = 1/3
        num_days = 1000

        self.AddAlpha(TechPairsAlphaModel(lookback_days,num_days,weight_pairs))
        self.AddAlpha(FinSerPairsAlphaModel(lookback_days,num_days,weight_pairs))
        self.AddAlpha(HealthCarePairsAlphaModel(lookback_days,num_days,weight_pairs))
        self.AddAlpha(BasicMaterialPairsAlphaModel(lookback_days,num_days,weight_pairs))

        # # ----------------- FactorVAE ----------------------------------
        
        spy = Symbol.Create("SPY", SecurityType.Equity, Market.USA)

        # weight_pairs represents the portfolio weights on the Pairs Trading Strategy
        weight_pairs = 1/3
        
        
        num_days = 1000
        
        self.AddAlpha(TechPairsAlphaModel(lookback_days,num_days,weight_pairs
        ))
        self.AddAlpha(FinSerPairsAlphaModel(lookback_days,num_days,weight_pairs
        ))
        self.AddAlpha(HealthCarePairsAlphaModel(lookback_days,num_days,weight_pairs
        ))
        self.AddAlpha(BasicMaterialPairsAlphaModel(lookback_days,num_days,weight_pairs
        ))

        # # ----------------- FactorVAE ----------------------------------
        strat_wight = 1
        #spy = Symbol.Create("SPY", SecurityType.Equity, Market.USA))
        # tsUniverseSelectionModel(spy)))tsUniverseSelectionModel(spy)))

        #self.vix = self.AddIndex("VIX").Symboll
        self.AddAlpha( TopkDrop(weight=strat_wight) )
        

        # ---------------- Portfolio Construction ----------------------------
        # self.Settings.RebalancePortfolioOnSecurityChanges = False
        # self.Settings.RebalancePortfolioOnInsightChanges = False
        # spy = Symbol.Create("SPY", SecurityType.Equity, Market.USA)
        # self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel('SPY'))
        # self.spy = self.AddEquity("SPY").Symbol
        # self.vix = self.AddIndex("VIX").Symbol

        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel1())
        # self.rebalance_func
        self.SetExecution(ImmediateExecutionModel())
      
        self.SetWarmUp(timedelta(40))
        

    # def CustomSecurityInitializer(self, security):
    #     '''Initialize the security with raw prices and zero fees 
    #     Args:
    #         security: Security which characteristics we want to change'''
    #     security.SetDataNormalizationMode(DataNormalizationMode.Raw)
    #     security.SetFeeModel(ConstantFeeModel(0))

    
def rebalance_func(self, time):
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









