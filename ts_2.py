# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from pandas import rolling_median
import matplotlib
matplotlib.style.use('ggplot')  # ggplot style
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   # acf pacf
from statsmodels.tsa.stattools import adfuller   # adf检验
from statsmodels.tsa.arima_model import ARMA


def draw_trend(timeSeries, size=12):
	'''移动平均图'''
	f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


def outliers(timeSeries, threshold=3):
	'''异常值处理 采用移动中位数方法'''
    ts_rolling = rolling_median(timeSeries, window=3, center=True)
    ts_dropna = ts_rolling.fillna(method='bfill').fillna(method='ffill')
    
    f=plt.figure(facecolor='white')
    ts_dropna.plot(color='blue', label='Original')

    plt.show()

    return ts_dropna


def testStationarity():
	ts=outliers(timeSeries)
	
	# adf 平稳性检验 
	dftest = adfuller(ts)
	# 语义描述
	dfoutput = pd.Series(dftest[0:4], index=[
	                     'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value (%s)' % key] = value
	return dfoutput


def best_diff(timeSeries, maxdiff = 8):
	'''判断最优差分'''
    p_set = {}
    for i in range(0, maxdiff):
        temp = pd.DataFrame(timeSeries).copy() #每次循环前，重置
        if i == 0:
            temp['diff'] = temp[temp.columns[0]]
        else:
            temp['diff'] = temp[temp.columns[0]].diff(i)
            temp = temp.drop(temp.iloc[:i].index)  #差分后，前几行的数据会变成nan，所以删掉

        pvalue = adfuller(temp['diff'])[1]
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df['p_value'][i]<0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff


def draw_acf_pacf(lags=31):
	'''acf pacf 默认为31阶'''
	bestdiff=best_diff(timeSeries)
	# 差分运算
	ts_diff = timeSeries.diff(int(bestdiff))
	ts_diff.dropna(inplace=True)
	# 绘图
	f = plt.figure(facecolor='white')
	ax1 = f.add_subplot(211)
	plot_acf(ts_diff, lags=40, ax=ax1)

	ax2 = f.add_subplot(212)
	plot_pacf(ts_diff, lags=40, ax=ax2)
	plt.show()

	return ts_diff


def adftest():
	ts=draw_acf_pacf()
	
	# adf 平稳性检验 
	dftest = adfuller(ts)
	# 语义描述
	dfoutput = pd.Series(dftest[0:4], index=[
	                     'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value (%s)' % key] = value
	return dfoutput


#
data_ts=draw_acf_pacf()
maxLag=best_diff(timeSeries, maxdiff = 8)

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
	teend_plot=decomposition.trend.plot(color='black', label='Trend')
	# 季节因素项
	seasonal = decomposition.seasonal
	seasonal_plot = decomposition.seasonal.plot(color='red', label='Seasonal')
	# 剩余项
	residual = decomposition.resid
	residual_plot = decomposition.resid.plot(color='blue', label='Residual')

	plt.show()

	return trend,seasonal,residual


# 拟合ARMA
import statsmodels.tsa.stattools as st 

order = st.arma_order_select_ic(data_ts,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order

model=ARMA(data_ts,order=order.bic_min_order)
result_arma = model.fit(disp=-1, method='css')


# 模型拟合完后，我们就可以对其进行预测了
predict_ts = result_arma.predict('2017-03-20','2017-03-31')


# 预测的Y值 还原
'''由于ARMA拟合的是经过相关预处理后的数据，故其
预测值需要通过相关逆变换进行还原（根据相关的变换情况）'''

# 一阶差分还原
diff_shift_ts = data_ts.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)

# 再次一阶差分还原
rol_mean=outliers(timeSeries, threshold=3)

rol_shift_ts = rol_mean.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)

# 移动平均还原
rol_sum = ts_log.rolling(window=11).sum()
rol_recover = diff_recover * 12 - rol_sum.shift(1)

# # 对数还原
# log_recover = np.exp(rol_recover)
# log_recover.dropna(inplace=True)

# 使用均方根误差（RMSE）来评估模型样本内拟合的好坏
ts = ts[diff_recover.index]  # 过滤没有预测的记录

plt.figure(facecolor='white')
diff_recover.plot(color='blue', label='Predict')

ts.plot(color='red', label='Original')

plt.legend(loc='best')
plt.title('RMSE: %.4f' % np.sqrt(sum((diff_recover_1 - ts)**2) / ts.size))
plt.show()


