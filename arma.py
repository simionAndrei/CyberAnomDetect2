from time import sleep

import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.graphics.tsaplots
import statsmodels.stats.stattools
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa
from statsmodels.graphics.api import qqplot
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

import scipy.stats

# Plots the autocorrelation graphs
from statsmodels.tsa.stattools import acf


def plot_autocorrelations(df):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsaplots.plot_acf(df.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsaplots.plot_pacf(df, lags=40, ax=ax2)


# Creates an ARMA model with the specified parameters
def create_model(df, p, q, r=0):
    return ARIMA(df, order=(p, q, r)).fit()


# Prints the stats for an ARMA model
def stats(arma_mod):
    print("AIC: %f\tBIC: %f\tHQIC: %f\tDurbin_Watson: %f" %
          (arma_mod.aic, arma_mod.bic, arma_mod.hqic, sm.stats.stattools.durbin_watson(arma_mod.resid.values)))


def plot_model(arma_mod):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax = arma_mod.resid.plot(ax=ax)


def analise_model(arma_mod):
    plot_model(arma_mod)

    resid = arma_mod.resid

    print(scipy.stats.normaltest(resid))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)


def extract_resid_stats(arma_mod):
    resid = arma_mod.resid
    r, q, p = acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


def plot_predictions(df, arma_mod):
    startI = int(len(df) * 0.98)

    start = df.index[startI]
    end = df.index[-1]

    predict_points = arma_mod.predict(start=start, end=end)

    plt.figure(figsize=(10,5))

    times = df[startI:].index
    plt.plot(times, df[startI:], color = 'blue', label = "True")
    plt.plot(times, predict_points, label='Prediction', color = 'gold')
    plt.savefig("test.png", dpi = 120, bbox_inches='tight')

    '''
    ax = df[startI:].plot(figsize=(12, 8))
    ax = predict_points.plot(ax=ax, style='r--', label='Prediction')
    ax.legend()
    '''

    original = df[startI:]

    print("Mean Forecast Error: {} Mean Absolute Error: {} Mean Square error:  {}".format(
        mean_forecast_err(original, predict_points),
        mean_absolute_err(original, predict_points),
        mean_squared_error(original, predict_points)))

    # ax.axis((-20.0, 38.0, -4.0, 200.0))


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat))  # or percent error = * 100


def answer(series, field, p, q, r=0):
    df1 = series[field]
    plot_autocorrelations(df1)

    # %%
    df1_model = create_model(df1, p, q, r)
    stats(df1_model)
    # %%
    analise_model(df1_model)
    # %%

    plot_autocorrelations(df1_model.resid)
    # %%
    extract_resid_stats(df1_model)
    # %%
    plot_predictions(df1, df1_model)
