#region imports
from AlgorithmImports import *
#endregion

# Your New Python File
class MyRiskManagementModel(RiskManagementModel):
    securities = []

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        super().OnSecuritiesChanged(algorithm, changes)
        for security in changes.AddedSecurities:
            # Store and manage Symbol-specific data
            
            security.indicator = algorithm.SMA(security.Symbol, 20)
            algorithm.WarmUpIndicator(security.Symbol, security.indicator)

            self.securities.append(security)

        for security in changes.RemovedSecurities:
            if security in self.securities:
                algorithm.DeregisterIndicator(security.indicator)
                self.securities.remove(security)