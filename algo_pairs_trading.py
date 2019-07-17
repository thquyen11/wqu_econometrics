import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from statsmodels.stats.stattools import jarque_bera
from sklearn.model_selection import train_test_split
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn


def find_cointegrated_pairs(data, columns, plot=False):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pvalue_min = np.inf
    alpha = 0.05
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            return_S1 = data[columns[i]]
            return_S2 = data[columns[j]]
            res_coint = coint(return_S1, return_S2)
            score_matrix[i, j] = res_coint[0]
            pvalue_matrix[i, j] = res_coint[1]
            print(
                f'pvalue: {res_coint[1]} of pairs: {(columns[i], columns[j])}')
            if res_coint[1] < alpha and res_coint[1] < pvalue_min:
                pvalue_min = res_coint[1]
                pairs = [columns[i], columns[j]]

    if plot is True:
        # Illustrate the co-integrated pairs
        seaborn.heatmap(pvalues, xticklabels=keys,
                        yticklabels=keys, cmap='RdYlGn_r',
                        mask=(pvalues >= 0.98))
        plt.show()
        plt.plot(panel_data[pairs[0]], color='red')
        plt.plot(panel_data[pairs[1]], color='green')
        plt.show()

    return score_matrix, pvalue_matrix, pairs


def normalize_data(data):
    mean = np.mean(data)
    sigma = np.std(data)
    data = (data - mean)/sigma
    return data


def get_arima_model(timeseries):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_range = range(5)
    d_range = range(2)
    for p in pq_range:
        for d in d_range:
            for q in pq_range:
                try:
                    tmp_mdl = ARIMA(timeseries, order=(p, d, q)).fit(
                        method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (p, d, q)
                        best_mdl = tmp_mdl
                except:
                    continue

    print(f"ARIMA - aic: {best_aic}, order: {best_order}")
    return best_aic, best_order, best_mdl


def _check_heteroskedastic_behavior(residuals):
    score, pvalue_jb, _, _ = jarque_bera(residuals)
    lbvalue, pvalue_lb = acorr_ljungbox(
        residuals**2, lags=[20], boxpierce=False)

    if pvalue_jb < 0.05:
        print("We have reason to suspect that the residuals are not normally distributed")
    else:
        print("The residuals seem normally distributed")

    if pvalue_lb < 0.05:
        print("We have reason to suspect that the residuals are autocorrelated")
        return True
    else:
        print("The residuals seem like white noise")
        return False


def get_garch_model(residuals):
    pq_range = range(5)
    o_range = range(2)
    best_aic = np.inf
    best_model = None
    best_order = None

    for p in pq_range:
        for o in o_range:
            for q in pq_range:
                try:
                    tmp_model = arch_model(residuals, p=p, o=o, q=q, dist='StudentsT').fit(
                        update_freq=5, disp='off')
                    tmp_aic = tmp_model.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_model = tmp_model
                        best_order = (p, o, q)
                except:
                    continue

    print(f'GARCH - aic: {best_aic}, order: {best_order}')
    return best_aic, best_order, best_model


def predict_future_trend(data, plot=False):
    windowLength = 50
    foreLength = len(data) - windowLength
    signal = 0*data[-foreLength:]
    forecast = pd.DataFrame(index=signal.index, columns=['Forecast'])
    forecast_output = []

    for d in range(foreLength):
        TS = data[d:(windowLength+d)]
        out = forecast_nextday(TS)
        forecast_output.append(out)
        signal.iloc[d] = np.sign(out)

    if plot is True:
        returns = pd.DataFrame(index=signal.index, columns=[
                               'Buy and Hold', 'Strategy'])
        returns['Buy and Hold'] = return_S1[-foreLength:]
        returns['Strategy'] = signal*returns['Buy and Hold']
        eqCurves = pd.DataFrame(index=signal.index, columns=[
                                'Buy and Hold', 'Strategy'])
        eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum()+1
        eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        eqCurves['Strategy'].plot(figsize=(10, 8))
        eqCurves['Buy and Hold'].plot()
        plt.legend()
        plt.show()

    return forecast


def forecast_nextday(timeseries):
    res_setup = get_arima_model(timeseries)
    aic_arima = res_setup[0]
    order_arima = res_setup[1]
    mdl_arima = res_setup[2]

    if _check_heteroskedastic_behavior(mdl_arima.resid):
        res_garch = get_garch_model(mdl_arima.resid)
        aic_garch = res_garch[0]
        order_garch = res_garch[1]
        mdl_garch = res_garch[2]
        # print(mdl_garch.summary())
        out = mdl_garch.forecast(horizon=1, start=None, align='origin')
        return out.mean['h.1'].iloc[-1]

    else:
        return mdl_arima.forecast()[0]


def initiate_trading_signal(series_1, series_2):
    spread = return_S1 - return_S2
    mean = spread.mean()
    upper_threshold = mean + 2*spread.std()
    lower_threshold = mean - 2*spread.std()
    buy_signal = spread.copy()
    sell_signal = spread.copy()

    buy_signal[spread > upper_threshold] = 0
    sell_signal[spread < lower_threshold] = 0

    return buy_signal, sell_signal


if __name__ == '__main__':
    start_date = '2018-03-01'
    # start_date = '2018-01-01'
    end_date = '2018-08-31'
    tickers = ['BTC-USD', 'ETH-USD', 'EOS-USD', 'LTC-USD', 'XMR-USD',
               'NEO-USD', 'ZEC-USD', 'BNB-USD', 'TRX-USD']

    panel_data = data.DataReader(
        tickers, 'yahoo', start_date, end_date)['Adj Close']
    keys = panel_data.keys()

    # Find the best co-integrated pairs
    scores, pvalues, pairs = find_cointegrated_pairs(
        panel_data, keys)
    print(f'The best cointegrated pairs is: {pairs}')

    # TRADING STRATEGY: flag TRADE if spread > 2 sigma
    return_S1 = np.log(panel_data[pairs[0]]/panel_data[pairs[0]].shift(1)
                       ).replace([np.inf, -np.inf], np.nan).dropna()
    return_S2 = np.log(panel_data[pairs[1]]/panel_data[pairs[1]].shift(1)
                       ).replace([np.inf, -np.inf], np.nan).dropna()

    buy_signal, sell_signal = initiate_trading_signal(return_S1, return_S2)


    # Graph: buy sell signals
    S1 = panel_data[pairs[0]][1:]
    S2 = panel_data[pairs[1]][1:]
    buyR = 0*S1
    sellR = 0*S1
    # When buying the ratio, buy S1 and sell S2
    buyR[buy_signal != 0] = S1[buy_signal != 0]
    sellR[buy_signal != 0] = S2[buy_signal != 0]
    # When selling the ratio, sell S1 and buy S2
    buyR[sell_signal != 0] = S2[sell_signal != 0]
    sellR[sell_signal != 0] = S1[sell_signal != 0]

    S1.plot(color='b')
    S2.plot(color='c')
    buyR.plot(color='g', linestyle='None', marker='^')
    sellR.plot(color='r', linestyle='None', marker='^')
    plt.legend(['Series 1', 'Series 2', 'Buy Signal', 'Sell Signal'])
    plt.show()
