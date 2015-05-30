from pandas.io.data import DataReader
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

MSFT = DataReader("MSFT",  "yahoo", datetime(2014,3,11), datetime(2015,3,11))

## Plot
# MSFT["Volume"].plot(kind="bar")

## slice with boolean, column, and row
is_volatile = MSFT['High']-MSFT['Low'] > 1
is_active = MSFT['Volume'] > 5e7
print MSFT[is_active & is_volatile][['High','Low','Volume']][1:5]

## integer division in python
print sum(MSFT[is_active]['Volume']) / sum(MSFT['Volume'])
print sum(MSFT[is_active]['Volume']) / sum(MSFT['Volume']).astype(float)
ActVolPct = MSFT[is_active]['Volume'] / sum(MSFT['Volume']).astype(float)
# ActVolPct.plot(kind='bar')
# plt.show()

## Load .csv files
# bikes = pd.read_csv('../data/bikes.csv', 
# 	sep=';', encoding='latin1', parse_dates=['Date'], 
#	dayfirst=True, index_col='Date')

## Weekday and aggregating
MSFT['weekday'] = MSFT.index.weekday
print MSFT.groupby('weekday').aggregate(np.sum)
print MSFT.groupby('weekday').agg(sum) # equivalent results with np

## save to .csv
MSFT[['Volume', 'weekday']].groupby('weekday').agg(np.sum).\
	to_csv('MSFT_WkdVol.csv')

## aggregate by month using .resample
# is_snowing = weather_description.str.contains('Snow') # string
print MSFT['Volume'].astype(float).resample('M', how = np.mean)

MSFT[['Close','Volume']].astype(float).resample('M', how = np.mean).\
	plot(kind = 'bar', subplots = True)
# plt.show()

## unique elements in an array
print MSFT['weekday'].unique()

## combined plots

# top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
# top.plot(MSFT.index, MSFT["Close"])
# plt.title('Microsoft Price - 1 Year')

# bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
# bottom.bar(MSFT.index, MSFT['Volume'])
# plt.title('Microsoft Trading Volume')

# plt.gcf().set_size_inches(15,8)
# plt.show()

## density
MSFT.Volume.plot(kind="kde")
# plt.show()

## melt
MSFT_melt = pd.melt(MSFT, id_vars = "weekday")
MSFT






















