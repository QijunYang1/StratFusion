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

class MarketMaking(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 10, 11)
        self.SetEndDate(2023, 10, 11)
        self.SetCash(10000)
        self.SetWarmUp(timedelta(days=1))
        self.security = self.AddEquity("SPY", Resolution.Tick, Market.USA)
        self.symbol = self.security.Symbol
        self.tick_sz = self.security.SymbolProperties.MinimumPriceVariation
        
        self.Nlevel = 5
        self.W = [math.exp( -i ) for i in range(self.Nlevel)]
        self.cont = 0

        self.L_bid = []
        self.L_ask = []
        self.arrive_dist = []
        self.A = 0
        self.k = 0
        self.vol = 0
        self.gamma = 0.05
        self.ask_ticket_num = 20
        self.bid_ticket_num = 20

        self.security.SetFeeModel(ConstantFeeModel(0))
        
        # self.Schedule.On(self.DateRules.EveryDay(self.symbol), self.TimeRules.Every(timedelta(seconds=60)), self.submit_order)
    
    def OnData(self, data):
        if self.IsWarmingUp:
            # if not data.ContainsKey(self.symbol) and not data.Ticks.ContainsKey(self.symbol):
            #     return
            self.build_LOB(data)
            return
        self.build_LOB(data)
        self.submit_order()
        
    def submit_order(self):
        # if not data.ContainsKey(self.symbol) and not data.Ticks.ContainsKey(self.symbol):
        #     return
        if self.IsWarmingUp:
            return
        ask_p = 0
        bid_p = 0
        if self.L_bid and self.L_ask:
            mid = (-self.L_bid[0][0]+self.L_ask[0][0])/2
            if self.Portfolio.Invested:
                q = abs(self.Portfolio[self.symbol].Quantity)
                q = q if self.Portfolio[self.symbol].IsLong\
                else -q
            else:
                q = 0
            reservation_p = mid - q * self.gamma * self.vol * (6.5*60*60)
            spread = self.gamma * self.vol * (6.5*60*60) + 2/self.gamma * np.log(1+self.gamma/self.k)
            ask_p = reservation_p + spread/2
            bid_p = reservation_p - spread/2
            
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
            # self.Debug(self.Time)
            # self.Debug('Submit order!')
            self.Plot("Price", "MidPrice", mid)
            self.Plot("Price", "Ask", self.L_ask[0][0])
            self.Plot("Price", "Bid", -self.L_bid[0][0])
            self.Plot("Price", "ReservationPrice", reservation_p)
            self.Plot("Price", "MyAsk", ask_p)
            self.Plot("Price", "MyBid", bid_p)
            self.Plot('Repository','q',q)


            # self.LimitOrder(self.symbol, -1, ask_p)
            # self.LimitOrder(self.symbol, 1, bid_p)

        # if ask_p != 0:
        #     if self.ask_ticket:
        #         if self.ask_ticket.Status == OrderStatus.Filled:
        #             self.ask_ticket = self.LimitOrder(self.symbol, -1, ask_p)
        #         else:
        #             response = self.ask_ticket.UpdateLimitPrice(ask_p) 
        #     else:
        #         self.ask_ticket = self.LimitOrder(self.symbol, -1, ask_p)

        # if bid_p != 0:
        #     if self.bid_ticket:
        #         if self.bid_ticket.Status == OrderStatus.Filled:
        #             self.bid_ticket = self.LimitOrder(self.symbol, 1, bid_p)
        #         else:
        #             response = self.bid_ticket.UpdateLimitPrice(bid_p)
        #     else:
        #         self.bid_ticket = self.LimitOrder(self.symbol, 1, bid_p)

    def OnWarmupFinished(self) -> None:
        self.Log(self.Time)
        self.Log("OnWarmupFinished Done")
        self.L_bid = []
        self.L_ask = []
        self.arrive_dist = []

    def OnEndOfDay(self) -> None:
        self.estimateArriveRate()
        self.estimateVol()
        self.L_bid = []
        self.L_ask = []
        self.arrive_dist = []

    def OnOrderEvent(self, orderEvent):
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"{self.Time}: {order.Type}: {orderEvent}")
            return

    def plotting(self):
        if self.L_bid and self.L_ask and self.Time.hour < 14 and self.Time.hour >8:
            self.Debug(self.Time)
            
            self.Plot("Price", "Bid2", -self.L_bid[1][0])
            self.Plot("Price", "Bid1", -self.L_bid[0][0])
            self.Plot("Price", "Ask1", self.L_ask[0][0])
            self.Plot("Price", "Ask2", self.L_ask[1][0])
            self.Plot("Price", "MidPrice", (-self.L_bid[0][0]+self.L_ask[0][0])/2)
            
            Lb = [[float(Decimal("%.2f"%(-item[0]))) , int(Decimal("%.2f"%(item[1])))] for item in self.L_bid]
            Ls = [[float(Decimal("%.2f"%(item[0]))) , int(Decimal("%.2f"%(item[1])))] for item in self.L_ask]
            sumVolBid = sum(Vb*W for Vb,W in zip(Lb[0:self.Nlevel][1],self.W))
            sumVolAsk = sum(Va*W for Va,W in zip(Ls[0:self.Nlevel][1],self.W))
            imbalanceRatio = (sumVolBid-sumVolAsk)/(sumVolBid+sumVolAsk)
            self.Plot("Imbalance Ratio", "Value", imbalanceRatio)
            
            self.Debug(str(Lb[self.Nlevel-1::-1])+"   "+str(Ls[0:self.Nlevel]))
        if self.Time.hour > 14:
            self.Quit()
    

    def build_LOB(self,data):
        for tick in data.Ticks[self.symbol]:
            if tick.TickType == TickType.Trade:
                if self.L_bid and self.L_ask:
                    
                    if tick.Price in [-x for x in self.L_bid[:][0]]:
                        p, v, d = tick.Price, tick.Quantity, 1
                        order=[[p,v,d]]
                        self.L_bid, self.L_ask = self.reconstructLOB(order, self.L_bid, self.L_ask)
                    
                    if tick.Price in self.L_ask[:][0]:
                        p, v, d = tick.Price, tick.Quantity, 0
                        order=[[p,v,d]]
                        self.L_bid, self.L_ask = self.reconstructLOB(order, self.L_bid, self.L_ask)

            if tick.TickType == TickType.Quote:
                if tick.AskSize>1:
                    p, v, d = tick.AskPrice, tick.AskSize, 1
                else:
                    p, v, d = tick.BidPrice, tick.BidSize, 0
                order=[[p,v,d]]
                self.L_bid, self.L_ask = self.reconstructLOB(order, self.L_bid, self.L_ask)

    def reconstructLOB(self,orders,bid_book,ask_book):
        trade_ps = []
        mid_ps = []
        for x in orders: # iterate all the orders received
            price,number,direction = x[0],x[1],x[2]
            n = number
            if direction == 0: # buy
                while n > 0 and ask_book:
                    p,num = heapq.heappop(ask_book) # pop minimum ask
                    # if minimum ask is higher than trade price, put the ask back and return
                    if p > price: 
                        heapq.heappush(ask_book, [p, num]) 
                        break
                    # if minimum ask is lower than trade price, check the number sit on the book
                    if num <= n: # the trade ont only happens at best ask level
                        n -= num # reduce the trade number
                        if bid_book:
                            trade_ps.append(p) # add the price to trade_ps list
                            mid_ps.append((p-bid_book[0][0])/2)   # get mid price, bid price is negative                  
                        while n>0: # if we still have to trade
                            p,num = heapq.heappop(ask_book) # pop the next level
                            if num <= n:
                                n -= num
                                if bid_book:
                                    trade_ps.append(p)
                                    mid_ps.append((p-bid_book[0][0])/2) 
                            else:
                                heapq.heappush(ask_book, [p, num-n])
                                n = 0
                    # the trade only happens at best ask level, put the remaining ask back
                    else:
                        heapq.heappush(ask_book, [p, num-n]) 
                        n = 0 
                # If we take all the liquidity of the ask side then we just show the remaining at the trade price
                # This also considers the situation that we just want to add limited order
                if n > 0: 
                    heapq.heappush(bid_book,[-price,n])
            else: # sell
                while n > 0 and bid_book: 
                    p, num = heapq.heappop(bid_book)
                    # if maximum bid is lower than trade price, put the bid back and return
                    if -p < price:
                        heapq.heappush(bid_book, [p, num])
                        break
                    # if maximum bid is higher than trade price, check the number sit on the book
                    if num <= n: # the trade ont only happens at best bid level
                        n -= num
                        if ask_book:
                            trade_ps.append(-p)
                            mid_ps.append((-p+ask_book[0][0])/2)
                        while n>0 and len(bid_book)>0:
                            p,num = heapq.heappop(bid_book)
                            if num <= n:
                                n -= num
                                if ask_book:
                                    trade_ps.append(-p)
                                    mid_ps.append((-p+ask_book[0][0])/2)
                            else:
                                heapq.heappush(bid_book, [p, num-n]) # fill all that order then we have num-n volume
                                n = 0
                    else:
                        heapq.heappush(bid_book, [p, num-n])
                        n = 0
                # If we take all the liquidity of the bid side then we just show the remaining at the trade price
                # This also considers the situation that we just want to add limited order
                if n > 0:
                    heapq.heappush(ask_book,[price,n])
        
        # Add up the number at same price
        bu=[[k, sum(v for _, v in g)] for k, g in groupby(sorted(bid_book), key = lambda x: x[0])]
        se=[[k, sum(v for _, v in g)] for k, g in groupby(sorted(ask_book), key = lambda x: x[0])]
        # heapify the list
        heapq.heapify(bu)
        heapq.heapify(se)
        self.getDistance(trade_ps, mid_ps)
        return bu, se

    def getDistance(self,trade_ps,mid_ps):
        for trade_p,mid_p in zip(trade_ps,mid_ps):
            self.arrive_dist.append(abs(trade_p-mid_p)/self.tick_sz)

    def estimateArriveRate(self):
        #  Estimate second Arrive rate
        # y = A*exp(-k*x)
        if self.arrive_dist:
            data = np.array(self.arrive_dist)
            bins = np.linspace(0, 20, 100)
            digitized = np.digitize(data, bins)
            bin_estimated = np.array([len(data[digitized == i])/(6.5*60*60) for i in range(1, len(bins)+1)])
            idx = np.nonzero(bin_estimated)
            x = bins[idx]*self.tick_sz
            y = np.log(bin_estimated[idx]) # log(y)=log(A)-k*x
            regressor = stats.linregress(x, y)
            self.A = np.exp(regressor.intercept)
            self.k = -regressor.slope
            self.Log(f'k: {self.k}')

    def estimateVol(self):
        #  Estimate second Vol
        df = self.History(self.symbol,timedelta(days=1),Resolution.Minute)
        if len(df)!=0:
            diff5M = df['close'].unstack(0).resample('5Min').apply(lambda x: x[-1]-x[0])
            vol = (diff5M**2).sum()/len(diff5M)/(5*60)
            self.vol = vol[0]
    