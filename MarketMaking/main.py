#region imports
from AlgorithmImports import *
#endregion
import numpy as np
import decimal
import heapq
from itertools import groupby
from heapq import heappush
import random
import math
from decimal import *
from scipy import stats

class marketMaking(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 10, 1)
        self.SetEndDate(2023, 10, 14)
        self.SetCash(10000)
        self.security = self.AddEquity("SPY", Resolution.Second, Market.USA,leverage=10)
        self.symbol = self.security.Symbol

        self.A = 0
        self.k = 0
        self.vol = 0
        self.gamma = 0.01
        self.ask_ticket_num = 20
        self.bid_ticket_num = 20

        # self.security.SetFeeModel(ConstantFeeModel(0))
        
        self.Schedule.On(self.DateRules.EveryDay(self.symbol),  self.TimeRules.At(9, 0), self.estimate_parameter)
    
    def OnData(self, data):
        if self.IsWarmingUp:
            return
        quote_bar = data.QuoteBars[self.symbol]
        mid = quote_bar.Close
        if self.Portfolio.Invested:
            q = abs(self.Portfolio[self.symbol].Quantity)
            q = q if self.Portfolio[self.symbol].IsLong\
            else -q
        else:
            q = 0
        
        # # AS
        # reservation_p = mid - q * self.gamma * self.vol * (6.5*60*60)
        # spread = self.gamma * self.vol * (6.5*60*60) + 2/self.gamma * np.log(1+self.gamma/self.k)
        # ask_p = reservation_p + spread/2
        # bid_p = reservation_p - spread/2

        # ASQ
        shared_shift = 1/self.gamma*np.log(1+self.gamma/self.k)
        bid_spread = shared_shift + (2*q+1)/2*np.sqrt(self.vol*self.gamma/(2*self.k*self.A)*np.power(1+self.gamma/self.k,1+self.gamma/self.k))
        ask_spread = shared_shift - (2*q-1)/2*np.sqrt(self.vol*self.gamma/(2*self.k*self.A)*np.power(1+self.gamma/self.k,1+self.gamma/self.k))
        bid_p = mid - bid_spread
        ask_p = mid + ask_spread

        open_bid_quantity = abs(self.Transactions.GetOpenOrdersRemainingQuantity(
            lambda order_ticket: order_ticket.Quantity > 0
        ))
        open_ask_quantity = abs(self.Transactions.GetOpenOrdersRemainingQuantity(
            lambda order_ticket: order_ticket.Quantity < 0
        ))

        if open_bid_quantity < self.bid_ticket_num:
            self.LimitOrder(self.symbol, 1, bid_p)
        else:
            filtered_open_orders = self.Transactions.GetOpenOrders(lambda x: x.Direction == OrderDirection.Buy)
            # sorted: reverse. False will sort ascending, True will sort descending. Default is False
            filtered_open_orders = sorted(filtered_open_orders,key=lambda x: x.OrderSubmissionData.BidPrice,reverse=True)
            ticket = self.Transactions.GetOrderTicket(filtered_open_orders[0].Id)
            ticket.UpdateLimitPrice(bid_p)

        if open_ask_quantity < self.ask_ticket_num:
            self.LimitOrder(self.symbol, -1, ask_p)
        else:
            filtered_open_orders = self.Transactions.GetOpenOrders(lambda x: x.Direction == OrderDirection.Sell)
            # sorted: reverse. False will sort ascending, True will sort descending. Default is False
            filtered_open_orders = sorted(filtered_open_orders,key=lambda x: x.OrderSubmissionData.AskPrice)
            ticket = self.Transactions.GetOrderTicket(filtered_open_orders[0].Id)
            ticket.UpdateLimitPrice(ask_p)

        # self.Plot("Price", "MidPrice", mid)
        # self.Plot("Price", "Ask", quote_bar.Ask.Close)
        # self.Plot("Price", "Bid", quote_bar.Bid.Close)
        # self.Plot("Price", "ReservationPrice", reservation_p)
        # self.Plot("Price", "MyAsk", ask_p)
        # self.Plot("Price", "MyBid", bid_p)
        # self.Plot('Repository','q',q)

        # self.Log(f'open_bid_quantity: {open_bid_quantity}')
        # self.Log(f'open_ask_quantity: {open_ask_quantity}')

    def estimate_parameter(self) -> None:
        self.estimate_arriveRate()
        self.estimate_vol()


    def OnOrderEvent(self, orderEvent):
        # order = self.Transactions.GetOrderById(orderEvent.OrderId)
        # if orderEvent.Status == OrderStatus.Filled:
        #     self.Debug(f"{self.Time}: {order.Type}: {orderEvent}")
        #     return
        return

    def estimate_arriveRate(self):
        #  Estimate second Arrive rate
        # y = A*exp(-k*x)
        i = 1
        quote_df = self.History(QuoteBar, self.symbol, timedelta(days=i), Resolution.Second)
        trade_df = self.History(TradeBar, self.symbol, timedelta(days=i), Resolution.Second)
        while quote_df.empty:
            i+=1
            quote_df = self.History(QuoteBar, self.symbol, timedelta(days=i), Resolution.Second)
            trade_df = self.History(TradeBar, self.symbol, timedelta(days=i), Resolution.Second)
        mid_close = quote_df['close'].unstack(level=0)
        trade_close = trade_df['close'].unstack(level=0)
        mid_close.columns = ['SPY']
        trade_close.columns = ['SPY']

        delta = abs(trade_close - mid_close) # Assume ask and bid spread are the same
        x,y = np.unique(np.float32(delta['SPY'].values),return_counts=True)
        y = y[::-1].cumsum()[::-1] # take break though phenonmenon into account
        y = y/(6.5*60*60) # second arrive rate
        y = np.log(y) 
        regressor = stats.linregress(x, y)
        self.A = np.exp(regressor.intercept)
        self.k = -regressor.slope

    def estimate_vol(self):
        '''
        A second approach is to use the estimator proposed by Garman and Klass [60] 
        (orig- inally devised for volatility across different days) adapted to the 
        case of a time window of several minutes. The interest of this estimator 
        is that it is based on the long-term oscillations of the price.
        '''
        #  Estimate second Vol
        i = 1
        df = self.History(QuoteBar,self.symbol,timedelta(days=i),Resolution.Second)
        while df.empty:
            i+=1
            df = self.History(QuoteBar,self.symbol,timedelta(days=i),Resolution.Second)
        # Mid price vol
        diff5M = df['close'].unstack(0).resample('5Min').apply(lambda x: x[-1]-x[0])
        vol = (diff5M**2).sum()/len(diff5M)/(5*60)
        self.vol = vol[0]
    