# !/usr/bin/env python
# -*- coding:utf-8 -*-

import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   # acf pacf
from statsmodels.tsa.stattools import adfuller   # adf检验


# 示例数据
date = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
        6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304,
        9355, 10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707,
        10767, 12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
        13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
        9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
        11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]

date_index = pd.date_range('2017-01-01', periods=90, Freq='D')
date = pd.Series(date, index=date_index)
timeSeries = date


def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()


def draw_acf_pacf(timeSeries, lags=31):
    '''acf pacf 默认为31阶差分 '''
    # bestdiff = best_diff(timeSeries)
    # 差分运算
    # ts_diff = timeSeries.diff(int(bestdiff))
    # ts_diff.dropna(inplace=True)

    # 绘图
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(timeSeries, lags=lags, ax=ax1)

    ax2 = f.add_subplot(212)
    plot_pacf(timeSeries, lags=lags, ax=ax2)
    plt.show()


def outliers(timeSeries, threshold=3):
    '''异常值处理 采用移动中位数方法'''
    ts_rolling = rolling_median(timeSeries, window=3, center=True)
    ts_dropna = ts_rolling.fillna(method='bfill').fillna(method='ffill')

    return ts_dropna


def best_diff(timeSeries, maxdiff=8):
    '''判断最优差分'''
    p_set = {}
    for i in range(0, maxdiff):
        temp = pd.DataFrame(timeSeries).copy()  # 每次循环前，重置
        if i == 0:
            temp['diff'] = temp[temp.columns[0]]
        else:
            temp['diff'] = temp[temp.columns[0]].diff(i)
            temp = temp.drop(temp.iloc[:i].index)  # 差分后，前几行的数据会变成nan，所以删掉

        pvalue = adfuller(temp['diff'])[1]
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']

    i = 0
    while i < len(p_df):
        if p_df['p_value'][i] < 0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff


def draw_trend(timeSeries, size=12):
    '''移动平均图'''
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    # ggplot style
    import matplotlib
    matplotlib.style.use('ggplot')

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


def testStationarity(timeSeries):
    # adf 检验
    dftest = adfuller(timeSeries)
    # 语义描述
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value',
                         '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def diff(timeSeries):
    '''ADF检验时间序列不是平稳序列 进行差分运算 并绘制图形'''
    date = timeSeries

    fig = plt.figure(facecolor='white', figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    date_plot = date.plot()

    # diff_1
    ax2 = fig.add_subplot(222)
    diff_1 = timeSeries.diff(1)
    date_diff_1_plot = diff_1.plot()

    # diff_2
    ax3 = fig.add_subplot(223)
    diff_2 = timeSeries.diff(2)
    date_diff_2_plot = diff_2.plot()

    # diff_3
    ax4 = fig.add_subplot(224)
    diff_3 = timeSeries.diff(3)
    date_diff_3_plot = diff_3.plot()
    plt.show()

    return diff_1, diff_2, diff_3


# 数据平稳后，需要对模型定阶，即确定p、q的阶数
def proper_model(data_ts, maxLag):
    '''对于个数不多的时序数据，我们可以通过观察自相关图和偏相关图来进行模型识别.
    对于数量较多的时序数据，依据BIC准则识别模型的p, q值，通常认为BIC值越小的模型相对更优。
    BIC准则，它综合考虑了残差大小和自变量的个数，残差越小BIC值越小，自变量个数越多BIC值越大'''
    import sys
    from statsmodels.tsa.arima_model import ARMA
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel


def disassemble(data_ts):
    # 分解
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(data_ts, model='additive')
    # 趋势项
    trend = decomposition.trend
    teend_plot = decomposition.trend.plot(color='black', label='Trend')
    # 季节因素项
    seasonal = decomposition.seasonal
    seasonal_plot = decomposition.seasonal.plot(color='red', label='Seasonal')
    # 剩余项
    residual = decomposition.resid
    residual_plot = decomposition.resid.plot(color='blue', label='Residual')

    plt.show()

    return trend, seasonal, residual


'''观察法，通俗的说就是通过观察序列的趋势图与相关图是否随着时间的变化呈现出某种规律。
所谓的规律就是时间序列经常提到的周期性因素，现实中遇到得比较多的是线性周期成分，这类
周期成分可以采用差分或者移动平均来解决，而对于非线性周期成分的处理相对比较复杂，
需要采用某些分解的方法。下图为航空数据的线性图，可以明显的看出它具有年周期成分和长期趋势成分。
平稳序列的自相关系数会快速衰减，下面的自相关图并不能体现出该特征，
所以我们有理由相信该序列是不平稳的。'''


if __name__ == "__main__":
    # 1.可视化时间序列
    draw_ts(timeSeries)

    # 2.可视化 acf pacf
    draw_acf_pacf(timeSeries)

    # 3.异常值处理 判断最优差分
    timeSeries = outliers(timeSeries)
    maxLag = best_diff(timeSeries)

    # 模型识别
    # rol_mean = ts.rolling(window=12).mean()
    # rol_mean.dropna(inplace=True)

    # diff_1
    # ts_diff_1 = rol_mean.diff(1)
    # ts_diff_1.dropna(inplace=True)

    # diff_2
    # ts_diff_2 = ts_diff_1.diff(1)
    # ts_diff_2.dropna(inplace=True)

    # 2阶差分 adf
    # testStationarity(ts_diff_2)

    # 拟合ARMA
    order = st.arma_order_select_ic(timeSeries, max_ar=3, max_ma=3,
                                    ic=['aic', 'bic', 'hqic'])
    order.bic_min_order
    model = ARMA(ts_diff_2, order=(1, 1))
    result_arma = model.fit(disp=-1, method='css')

    # 模型拟合完后，我们就可以对其进行预测了
    predict_ts = result_arma.predict()

    # 预测的Y值 还原
    '''由于ARMA拟合的是经过相关预处理后的数据，故其
    预测值需要通过相关逆变换进行还原（根据相关的变换情况）'''

    # 一阶差分还原
    diff_shift_ts = ts_diff_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)

    # 再次一阶差分还原
    rol_shift_ts = rol_mean.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)

    # 移动平均还原
    rol_sum = ts_log.rolling(window=11).sum()
    rol_recover = diff_recover * 12 - rol_sum.shift(1)

    # 对数还原
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)

    # 使用均方根误差（RMSE）来评估模型样本内拟合的好坏
    ts = ts[log_recover.index]  # 过滤没有预测的记录
    plt.figure(facecolor='white')
    log_recover.plot(color='blue', label='Predict')
    ts.plot(color='red', label='Original')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover - ts)**2) / ts.size))
    plt.show()
