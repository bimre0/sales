# -*- coding: utf-8 -*-

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
import itertools
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller



train_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
test_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
oil_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv')
transaction_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv')
stores_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
holiday_event_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')


train_df = train_df.merge(stores_df, on ='store_nbr')
train_df = train_df.merge(oil_df, on ='date', how='left')
holiday_event_df = holiday_event_df.rename(columns={'type': 'holiday_type'})
train_df = train_df.merge(holiday_event_df, on='date', how='left')



avg_sales = train_df.groupby('date').agg({'sales': 'mean'}).reset_index()
#daily_avg_sales['weekly_avg_sales'] = daily_avg_sales['sales'].rolling(window=7).mean()
avg_sales['weekly_avg_sales'] = avg_sales['sales'].ewm(span=7, adjust=False).mean()


transaction_df['date'] = pd.to_datetime(transaction_df['date'])

# Extract the year from the 'date' column
transaction_df['year'] = transaction_df['date'].dt.year


# Extract the month from the 'date' column
transaction_df['month'] = transaction_df['date'].dt.month

train_df.to_csv('traindf_processed.csv')

table = pa.Table.from_pandas(train_df)

pq.write_table(table, 'train.parquet', compression='GZIP')