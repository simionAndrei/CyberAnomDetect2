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
from logger import Logger
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample, _arma_predict_in_sample

from sklearn.metrics import mean_squared_error

import scipy.stats

# Plots the autocorrelation graphs
from statsmodels.tsa.stattools import acf


# perform a grid search to find the optimar ARIMA params
def grid_search(df, terms, ps, ds, qs):
    all_results = []
    for term in terms:
        df1 = df[term]
        results = []

        for p in ps:
            for q in qs:
                for d in ds:
                    try:
                        model = create_model(df1, p, d, q)
                        results.append([model, p, d, q, term])

                    except:
                        pass

        results.sort(key=lambda e: e[0].aic)

        all_results.append(results)

    return all_results


# plots the auto correlation
def plot_autocorrelations(df, title="Original Series"):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsaplots.plot_acf(df.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsaplots.plot_pacf(df, lags=40, ax=ax2)


# Creates an ARMA model with the specified parameters
def create_model(df, p, q, d=0):
    return ARIMA(df, order=(p, d, q)).fit(disp=-1)


# wrapper for creating a model
def create_model_no_fit(df, p, q, d=0):
    return ARIMA(df, order=(p, d, q))


# Prints the stats for an ARMA model
def stats(arma_mod):
    print("AIC: %f\tBIC: %f\tHQIC: %f\tDurbin_Watson: %f" %
          (arma_mod.aic, arma_mod.bic, arma_mod.hqic, sm.stats.stattools.durbin_watson(arma_mod.resid.values)))


# plots the model and the residuals
def plot_model(arma_mod):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax = arma_mod.resid.plot(ax=ax)


# prints the stats regarding an ARMA model
def analise_model(arma_mod):
    plot_model(arma_mod)

    resid = arma_mod.resid

    print(scipy.stats.normaltest(resid))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)


# prints the stats regarding the residuals
def extract_resid_stats(arma_mod):
    resid = arma_mod.resid
    r, q, p = acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


# creates a plot with the predictions vs the truth
def plot_predictions(df, arma_mod, file="test.png"):
    startI = int(len(df) * 0.98)

    start = df.index[startI]
    end = df.index[-1]

    predict_points = arma_mod.predict(start=start, end=end)

    plt.figure(figsize=(10, 5))

    times = df[startI:].index
    plt.plot(times, df[startI:], color='blue', label="True")
    plt.plot(times, predict_points, label='Prediction', color='gold')
    plt.title("Prediction")
    plt.legend()
    plt.savefig(file, dpi=120, bbox_inches='tight')

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


# plots multiple model prediction in the same plot
def plot_predictions_cmp(df, arma_mods, file="test.png"):
    startI = int(len(df) * 0.98)

    start = df.index[startI]
    end = df.index[-1]

    plt.figure(figsize=(10, 5))
    original = df[startI:]

    times = df[startI:].index
    plt.plot(times, df[startI:], color='blue', label="True")

    for (arma_mod, p, q) in arma_mods:
        predict_points = arma_mod.predict(start=start, end=end)
        plt.plot(times, predict_points, label='Prediction p({}) q({})'.format(p, q))

        print("Mean Forecast Error: {} Mean Absolute Error: {} Mean Square error:  {}".format(
            mean_forecast_err(original, predict_points),
            mean_absolute_err(original, predict_points),
            mean_squared_error(original, predict_points)))

    plt.title("Prediction")
    plt.legend()
    plt.savefig(file, dpi=120, bbox_inches='tight')

    '''
    ax = df[startI:].plot(figsize=(12, 8))
    ax = predict_points.plot(ax=ax, style='r--', label='Prediction')
    ax.legend()
    '''


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat))  # or percent error = * 100


# wrapper for doing everything regarding some parameters p and q :)
def answer(series, field, p, q, r=0, file="test.png"):
    df1 = series[field]
    plot_autocorrelations(df1)

    # %%
    df1_model = create_model(df1, p, q, r)
    stats(df1_model)
    # %%
    analise_model(df1_model)
    # %%

    plot_autocorrelations(df1_model.resid, "Residuals")
    # %%
    extract_resid_stats(df1_model)
    # %%
    plot_predictions(df1, df1_model, file)

    return df1_model


# if __name__ == '__main__':
#     def parser(x):
#         return pd.datetime.strptime(x, '%d/%m/%y %H')
#
#
#     logger = Logger(show=True, html_output=True, config_file="config.txt")
#
#     series = pd.read_csv(logger.get_data_file(logger.config_dict['TRAIN_FILE']), header=0, parse_dates=[0], index_col=0,
#                          squeeze=True, date_parser=parser)
#
#     # answer(series, 'L_T1', 2, 0)
#     # answer(series, 'L_T2', 2, 0)
#     # answer(series, 'L_T7', 2, 0)
#
#     grid_search_result = grid_search(series, ['L_T1', 'L_T2'], range(0, 5), range(0, 1), range(0, 1))
#
#     print("DONE")
