# region imports
from AlgorithmImports import *
# endregion
import torch
from portfolio_con import InsightWeightingPortfolioConstructionModel1
from a2c_trading_algo import A2CTradingModel
#from ETFUinverse import ETFConstituentsUniverseSelectionModel

class WellDressedTanWolf(QCAlgorithm):
    undesired_symbols_from_previous_deployment = []
    checked_symbols_from_previous_deployment = False
    previous_expiry_time = None
    def Initialize(self):
        self.SetStartDate(2020, 3, 1)  # Set Start Date
        self.SetEndDate(2020, 4, 1)
        self.SetCash(10000000)  # Set Strategy Cash 
        self.ind_period = 20
        self.lookback = 30
        
       
        #spy = Symbol.Create("SPY", SecurityType.Equity, Market.USA)

        #self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel(spy))
        #self.spy = self.AddEquity("SPY").Symbol
        #self.vix = self.AddIndex("VIX").Symbol
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        a2c_symbols = ["JPM","BRK.B", "AAPL","AMZN","MSFT"]
        symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in a2c_symbols]
       
        self.AddUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.UniverseSettings.Resolution = Resolution.Minute

        self.AddAlpha(A2CTradingModel(self, self.lookback, weight=1))
        self.SetWarmup(timedelta(45))
        
        


        self.Settings.RebalancePortfolioOnSecurutyyChanges = False
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        
        self.SetExecution(ImmediateExecutionModel())
        #self.SetRiskManagement(MaximumDrawdownPercentPortfolio())



        # create the 36-minutes data consolidator
        #threeCountConsolidator = TradeBarConsolidator(timedelta(days=1))
        #threeCountConsolidator.DataConsolidated += self.ConsolidationHandler
        #for symbol in self.symbol_list:
        #    self.SubscriptionManager.AddConsolidator(symbol, threeCountConsolidator)
        
        self.indicator_history ={}

        
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
                    
                    
    
        
    def OnMarginCall(self, requests: List[SubmitOrderRequest]) -> List[SubmitOrderRequest]:
        for i, order in enumerate(requests):
        # liquidate an extra 10% each time you get a margin call to give yourself more padding
            new_quantity = int(order.Quantity * 1.1)
            requests[i] = SubmitOrderRequest(order.OrderType, order.SecurityType, 
                                        order.Symbol, new_quantity, order.StopPrice, 
                                        order.LimitPrice, 0, self.Time, "OnMarginCall")
        return requests
    

        
    def OnData(self, data):

        
        pass
        #self.Log(f'{self.Portfolio.TotalPortfolioValue:.6f}')
        
        
            
        
    


    def OnEndOfAlgorithm(self):
        #self.Log(f"Indicator History: {self.indicator_history[self.aapl].head().to_string()}")
        self.Log(str(torch.seed()))
        self.Log(str(torch.random.seed()))




