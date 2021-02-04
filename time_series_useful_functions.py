# load a pandas with time index
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)#squeez create a series instead of a dataframe

#if the data doesn't come with the frequency it has to be added
result = seasonal_decompose(series, model='additive', freq=1)

#if the data doesn't have the time you can add it
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)

#read specific time period
print(series['1959-01'])

#create new indexes with the time information
dataframe = DataFrame()
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]

#create a shifted column
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
dataframe.columns = ['t', 't+1']

#plot line
series.plot()
pyplot.show()

#plot dot
series.plot(style='k.')
pyplot.show()

#multy plot with grouped with a specific frequency
groups = series.groupby(pd.Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.plot(subplots=True, legend=False)
pyplot.show()

#histogram
series.hist()
pyplot.show()

#density plot
series.plot(kind='kde')
pyplot.show()

#box plot data groupped with a specif frequency
one_year = series['1990']
groups = one_year.groupby(pd.Grouper(freq='M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
pyplot.show()

#heatmap of a selected frequency
groups = series.groupby(pd.Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()

#lag scatter plot
lag_plot(series)
pyplot.show()

#autocorrelation plot
autocorrelation_plot(series)
pyplot.show()

#upsampling data
upsampled = series.resample('D').mean()#changeing the frequency at a higher than the sampling frequency you end up with a upsampling series with nans

#interpolation
interpolated = upsampled.interpolate(method='linear')#linear
interpolated = upsampled.interpolate(method='spline', order=2)#polinomial

#downsampling with the mean
resample = series.resample('Q')
quarterly_mean_sales = resample.mean()

#transformation (the goal is to obtain a time series with a histo plot gaussian like)
dataframe['passengers'] = sqrt(dataframe['passengers'])#square root
dataframe['passengers'] = log(dataframe['passengers'])#log (for values positive and non 0, comon approach is to add a constant to the variables)

#boxcox (generalized transformation)
#  lambda = -1.0 is a reciprocal transform.
#  lambda = -0.5 is a reciprocal square root transform.
#  lambda = 0.0 is a log transform.
#  lambda = 0.5 is a square root transform.
#  lambda = 1.0 is no transform.

dataframe['passengers'] = boxcox(dataframe['passengers'], lmbda=0.0)#user selected transformation
dataframe['passengers'], lam = boxcox(dataframe['passengers'])#transformation selected with a fit
print('Lambda: %f' % lam)


# ███████ ███████  █████  ███████  ██████  ███    ██  █████  ██      ██ ████████ ██    ██
# ██      ██      ██   ██ ██      ██    ██ ████   ██ ██   ██ ██      ██    ██     ██  ██
# ███████ █████   ███████ ███████ ██    ██ ██ ██  ██ ███████ ██      ██    ██      ████
#      ██ ██      ██   ██      ██ ██    ██ ██  ██ ██ ██   ██ ██      ██    ██       ██
# ███████ ███████ ██   ██ ███████  ██████  ██   ████ ██   ██ ███████ ██    ██       ██


#automatic decomposition (simple and risky)
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
series = ...

# seasonal_decompose function generate 4 arrays
result = seasonal_decompose(series, model='additive')#model can be 'additive' or 'multiplicative'
print(result.trend)#trend present in the original data
print(result.seasonal)#seasonality present in the original data
print(result.resid)#original data without trend and seasonality
print(result.observed)#original data untouched
result.plot()
pyplot.show()

#differenziating can remove trend
series = series.diff(num_diff)
series.fillna(method='bfill', inplace=True)

#differenziating may be more powerful if used on data from the same day but on a different year, and more stable if the value comes from the averege over a month from the previous year
#seasonality can be removed as well, if the differenziating has the same frequency of the season
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
diff = list()
days_in_year = 365
for i in range(days_in_year, len(X)):
    month_str = str(series.index[i].year-1)+'-'+str(series.index[i].month)
    month_mean_last_year = series[month_str].mean()
    value = X[i] - month_mean_last_year
    diff.append(value)


    # ██ ███████     ███████ ████████  █████  ████████ ██  ██████  ███    ██  █████  ██████  ██    ██
    # ██ ██          ██         ██    ██   ██    ██    ██ ██    ██ ████   ██ ██   ██ ██   ██  ██  ██
    # ██ ███████     ███████    ██    ███████    ██    ██ ██    ██ ██ ██  ██ ███████ ██████    ████
    # ██      ██          ██    ██    ██   ██    ██    ██ ██    ██ ██  ██ ██ ██   ██ ██   ██    ██
    # ██ ███████     ███████    ██    ██   ██    ██    ██  ██████  ██   ████ ██   ██ ██   ██    ██


from random import seed
from random import random
from statsmodels.tsa.stattools import adfuller
import pandas as pd
# generate random walk
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)

# statistical test
# se il ADF stat e' maggiore dei valori critici la serie non e' stazionaria
#come nell'esempio seguente
result = adfuller(random_walk)
print('ADF Statistic: %f' % result[0])# maggiore dei valori critici non e' stazionario
print('p-value: %f' % result[1])# p-value > 0.05 non e' stazionario
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

series = pd.read_csv('daily-total-female-births-CA.csv', header=0, index_col=0, parse_dates=True,squeeze=True)

# statistical test
# se il ADF stat e' minore dei valori critici la serie e' stazionaria
#come nell'esempio seguente
result = adfuller(series)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


    # ██████  ███████ ███████ ██ ██████  ██    ██ ██
    # ██   ██ ██      ██      ██ ██   ██ ██    ██ ██
    # ██████  █████   ███████ ██ ██   ██ ██    ██ ██
    # ██   ██ ██           ██ ██ ██   ██ ██    ██ ██
    # ██   ██ ███████ ███████ ██ ██████   ██████  ██


#genera una statistica dei residui e il grafico
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot
def residuals_eval(y_true, y_pred):
    y_res = y_true - y_pred
    df_res = DataFrame(y_res)
    print(df_res.describe())
    df_res.plot()#line
    pyplot.show()
    df_res.hist()#hist
    pyplot.show()
    df_res.plot(kind='kde')#density plot
    pyplot.show()
    qqplot(numpy.array(y_res), line='r')
    pyplot.show()
    # A signicant autocorrelation in the residual plot suggests that the model could be doing
    # a better job of incorporating the relationship between observations and lagged observations,
    # called autoregression
    autocorrelation_plot(residuals)
    pyplot.show()
