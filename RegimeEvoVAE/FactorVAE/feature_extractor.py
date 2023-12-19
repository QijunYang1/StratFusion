#region imports
from AlgorithmImports import *
#endregion


### Library classes are snippets of code/classes you can reuse between projects. They are
### added to projects on compile.
###
### To import this class use the following import with your values subbed in for the {} sections:
### from {libraryProjectName} import {libraryFileName}
### 
### Example using your newly imported library from 'Library.py' like so:
###
### from {libraryProjectName} import Library
### x = Library.Add(1,1)
### print(x)
###

from numba import njit
import numpy as np
import pandas as pd

@njit(fastmath=True)
def trend_ratio(series):
    return  np.sum(series) / np.sum(np.abs(series))

@njit(fastmath=True)
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

@njit(fastmath=True)
def up_var(series):
    temp = series[series>0]
    return np.sqrt(np.sum(temp**2))
    
@njit(fastmath=True)
def down_var(series):
    temp = series[series<0]
    return np.sqrt(np.sum(temp**2))

def calc_other_feature(df,window):
    feat1 = df['close'][df['close']>0].groupby('symbol').rolling(window).apply(trend_ratio, engine='numba', raw=True).reset_index(level=0,drop=True)
    feat2 = df['close'][df['close']>0].groupby('symbol').rolling(window).apply(realized_volatility, engine='numba', raw=True).reset_index(level=0,drop=True)
    feat3 = df['close'][df['close']>0].groupby('symbol').rolling(window).apply(up_var, engine='numba', raw=True).reset_index(level=0,drop=True)
    feat4 = df['close'][df['close']>0].groupby('symbol').rolling(window).apply(down_var, engine='numba', raw=True).reset_index(level=0,drop=True)
    res = pd.concat([feat1,feat2,feat3,feat4],axis=1)
    res.columns = [ i+f'_{window}' for i in ['trend_ratio','realized_volatility','up_var','down_var']]
    return res

def calc_close_volume_corr(df, window):
    def rolling_corr(x):
        return pd.DataFrame(x['close'].rolling(window=5).corr(x['volume']))
    corr = df.groupby('symbol', group_keys=True)[['close','volume']].apply(rolling_corr).reset_index(level=0,drop=True)
    corr.columns = ['pv_corr_'+f'{window}']
    return corr


@njit
def calc_autocorr_numba(a):
    return np.corrcoef(a[:-1], a[1:])[0, 1]

def calc_autocorr(df, window):
    close_autocorr = df.groupby('symbol')['close'].rolling(window).apply(calc_autocorr_numba, engine='numba', raw=True).reset_index(level=0,drop=True)
    volume_autocorr = df.groupby('symbol')['volume'].rolling(window).apply(calc_autocorr_numba, engine='numba', raw=True).reset_index(level=0,drop=True)
    res = pd.concat([close_autocorr,volume_autocorr],axis=1)
    res.columns = [ i+f'_{window}' for i in ['close_autocorr','volume_autocorr']]
    return res

def calc_rt_rv(rt,window):
    rv = rt.groupby('symbol').rolling(window).apply(realized_volatility, engine='numba', raw=True).reset_index(level=0,drop=True)
    rv.columns = [i+'_rt_rv'+f'_{window}' for i in ['close', 'high', 'low', 'open', 'volume']]
    return rv

def calc_ts_feature(history, window):
    normalization = history.groupby('symbol').rolling(window).rank(pct=True).reset_index(level=0,drop=True)
    normalization.columns = [i+f'_norm_{window}' for i in history.columns]
    stats = history.groupby('symbol')[['close','close_rt','volume']].rolling(window).agg(['std','kurt','skew']).reset_index(level=0,drop=True)
    stats.columns = ['_'.join(col)+f'_{window}' for col in stats.columns]
    other_feature = calc_other_feature(history,window)
    rt = history[[i+f'_rt' for i in ['close', 'high', 'low', 'open', 'volume']]]
    rv = calc_rt_rv(rt,window)
    pv_corr = calc_close_volume_corr(history, window)
    autocorr = calc_autocorr(history, window)
    return pd.concat([normalization,stats,other_feature,rv,pv_corr,autocorr],axis=1)

def calc_features(history):
    rt = history.groupby('symbol').pct_change()
    rt.columns = [i+'_rt' for i in ['close', 'high', 'low', 'open', 'volume']]
    rt_pct = rt.groupby('symbol').pct_change()
    rt_pct.columns = [i+'_rt_pct' for i in ['close', 'high', 'low', 'open', 'volume']]
    trend = (history-history.shift())/(history+history.shift())
    trend.columns = [i+'_trend' for i in ['close', 'high', 'low', 'open', 'volume']]
    history = pd.concat([history,rt,rt_pct,trend],axis=1)
    ts_feature_20 = calc_ts_feature(history, 20)
    ts_feature_40 = calc_ts_feature(history, 40)
    ts_feature_60 = calc_ts_feature(history, 60)
    ts_feature_80 = calc_ts_feature(history, 80)
    history = pd.concat([history,ts_feature_20,ts_feature_40,ts_feature_60,ts_feature_80],axis=1)
    history.fillna(0,inplace=True)
    history.replace([-np.inf,np.inf],0,inplace=True)
    return history