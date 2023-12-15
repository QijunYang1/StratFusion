#region imports
from AlgorithmImports import *
#endregion

class MarketCapAlgorithm(QCAlgorithm):
    
    filteredByPrice = None
    
    def Initialize(self):
        self.SetStartDate(2017, 1, 1)  
        self.SetEndDate(2018, 1, 1) 
        self.SetCash(100000)

        # Addiing a universe based on two joint selections: self.CoarseSelectionFilter and self.FineSelectionFunction
        # The second argument is option,that is, to have fundamental filter, you still need the first filter
        
        self.AddUniverse(self.CoarseSelectionFilter, self.FineSelectionFunction)
    
        self.UniverseSettings.Resolution = Resolution.Daily
        self.count = 0

    def CoarseSelectionFilter(self, coarse):
        
        sortedByDollarVolume = sorted(coarse, key=lambda c: c.DollarVolume, reverse=True)
        
        self.filter_coarse = [c.Symbol for c in sortedByDollarVolume]
        
        return self.filter_coarse

    def FineSelectionFunction(self, fine):
        
        # fine1 = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.FinancialServices]
        # sortedByMarketCap = sorted(fine1, key=lambda c: c.MarketCap, reverse=True)

        # filteredFine = [i.Symbol for i in sortedByMarketCap][:10]

        # fine2 = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Technology]
        # sortedByMarketCap = sorted(fine2, key=lambda c: c.MarketCap, reverse=True)

        fine2 = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.CommunicationServices]
        sortedByMarketCap = sorted(fine2, key=lambda c: c.MarketCap, reverse=True)

        # filteredFine = [i.Symbol for i in sortedByMarketCap][:10]

        # fine3 = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.BasicMaterials]
        # sortedByMarketCap = sorted(fine3, key=lambda c: c.MarketCap, reverse=True)

        # fine3 = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Healthcare]
        # sortedByMarketCap = sorted(fine3, key=lambda c: c.MarketCap, reverse=True)

        # fine3 = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Utilities]
        # sortedByMarketCap = sorted(fine3, key=lambda c: c.MarketCap, reverse=True)
        # filteredFine = [i.Symbol for i in sortedByMarketCap][:15]

        # fine3 = [x for x in fine if x.AssetClassification.MorningstarIndustryGroupCode == MorningstarIndustryGroupCode.Semiconductors]
        # sortedByMarketCap = sorted(fine3, key=lambda c: c.MarketCap, reverse=True)
        filteredFine = [i.Symbol for i in sortedByMarketCap][:15]
        self.filter_fine = filteredFine
        
        return self.filter_fine
        
    def OnSecuritiesChanged(self, changes):
        self.changes = changes
         
    def OnData(self, data):

        tick_list = []
        if self.count == 0:
            for ticker in self.filter_fine:
                tick_list.append(str(ticker))

            self.Debug(tick_list) # print to see in the Console
            self.count = 1
