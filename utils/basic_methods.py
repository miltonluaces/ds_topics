import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def PlotFcst(hist, fcst):
    tot = list(hist) + list(fcst)
    plt.figure(figsize=(20, 10))
    plt.plot(tot, color='r')
    plt.plot(hist, color='b')
    plt.show()
 
    
def Naive(ts, hor):
    fcst = np.full(shape=hor, fill_value=ts[len(ts)-1])
    return fcst

def SimpleAvg(ts, hor):
    mean = ts.mean()
    fcst = np.full(shape=hor, fill_value=mean)
    return fcst

def MovingAvg(ts, hor, mw):
    mean = ts[-mw:].mean()
    fcst = np.full(shape=hor, fill_value=mean)
    return fcst
 
def ExpSmooth(ts, hor, sl):
    fit = SimpleExpSmoothing(ts).fit(smoothing_level=sl, optimized=False)
    fcst = fit.forecast(hor)
    return fcst

def HoltWintersLin(ts, hor, sl, ss):
    fit = Holt(ts).fit(smoothing_level = sl,smoothing_slope = ss)
    fcst = fit.forecast(hor)
    return fcst

def HoltWinters(ts, hor, sp, trd, sea):
    fit = ExponentialSmoothing(ts ,seasonal_periods=sp ,trend=trd, seasonal=sea,).fit()
    fcst = fit.forecast(hor)
    return fcst
  
def Arima(ts, hor):
    fit = statsmodels.tsa.statespace.sarimax.SARIMAX(ts, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
    fcst = fit.predict(dynamic=True)
    return fcst