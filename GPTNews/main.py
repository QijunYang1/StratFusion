#region imports
from AlgorithmImports import *
import numpy as np
import openai
import pandas as pd
from gpt_prompt import get_sentiment_general, get_sentiment_general_parallel, get_sentiment_general_sequence
import time
from io import StringIO
from datetime import timedelta
#endregion





class TiingoNewsDataAlgorithm(QCAlgorithm):

    def __init__(self):
        self.n = 9
        self.alphas = [0,-0.5,0.2,0.2,0,0.5,0,0,0]
        self.holding_times = [60] * 9
        self.long_thres = 1
        self.short_thres = -1

    def Initialize(self) -> None:
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2022, 12, 31)
        
        self.SetStartDate(self.start_date)
        self.SetEndDate(self.end_date)
        self.SetCash(1000000)
        
        self.current_holdings = 0
        self.target_holdings = 0
        self.sample_size = 50
        self.SetWarmup(-1)

        self.timers = [[] for _ in range(9)]
        
        # Requesting data
        self.aapl = self.AddEquity("AAPL", Resolution.Minute).Symbol
        self.tiingo_symbol = self.AddData(TiingoNews, self.aapl, resolution=Resolution.Minute).Symbol

        self.num_bull = 0
        self.num_neu = 0
        self.num_bear = 0
        
        self.prev_price = 0
        self.sentiments = []
        
        file = self.Download("https://www.dropbox.com/scl/fi/1xwdv4zv3l70djvnaai22/news_value_apple_2022_all.csv?rlkey=vvt0phcdhaar5bw3erk2q6yre&dl=1")
        self.score_df = pd.read_csv(StringIO(file))

        self.score_dic = self.score_df.set_index('date')['score'].to_dict()

        for hour in range(9, 17):  # Assuming trading hours from 10:00 to 16:00
            self.Schedule.On(self.DateRules.EveryDay("AAPL"), 
                            self.TimeRules.At(hour, 0),
                            self.rebalance)
        
        
    def OnData(self, slice: Slice) -> None:
        cur_time = self.Time
        for i in range(9):
            timer = self.timers[i]
            if timer:
                to_sell = 0
                for t in timer:
                    if (cur_time - t) > timedelta(minutes=self.holding_times[i]):
                        to_sell += 1
                self.timers[i] = self.timers[i][to_sell:]
                self.target_holdings -= self.alphas[i] * to_sell

        if slice.ContainsKey(self.tiingo_symbol) and self.Securities[self.aapl].Price != None:
            title_words = slice[self.tiingo_symbol].Title + slice[self.tiingo_symbol].Description
            cur_time = f"{self.Time}"[:19]
            if cur_time in self.score_dic:
                score = self.score_dic[cur_time]
                index = score - 1
                self.target_holdings += self.alphas[index]
                self.Debug(f"{score} {self.target_holdings}")
                self.timers[index].append(self.Time)
            
    def rebalance(self):
        self.target_holdings = round(self.target_holdings, 2)
        self.real_holdings = self.target_holdings
        if self.target_holdings > 1:
            self.real_holdings = 1
        if self.target_holdings < -1:
            self.real_holdings = -1
        self.Debug(f"{self.current_holdings} {self.target_holdings} {self.Time}")
        self.SetHoldings(self.aapl,  self.real_holdings)
        self.current_holdings = self.target_holdings