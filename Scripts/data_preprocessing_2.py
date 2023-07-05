
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


train_df = pd.read_csv('/kaggle/working/traindf_processed.csv')

"""# Missing Values Detection"""

#percentage of missing values in train_df

missing_percentages = train_df.isnull().sum()/ len(train_df) * 100

# remove columns that are having more than 30% missing values

columns_to_delete = missing_percentages[missing_percentages > 40].index

train_df = train_df.drop(columns=columns_to_delete)

"""# Duplicates"""

dupes=train_df.duplicated()

#dropping duplicate values

train_df = train_df.drop_duplicates()

train_df.duplicated().any()

test_df.duplicated().any()

"""# Check for missing values

# The dataset for dcoilwtico exhibits missing values. According to the provided plot, a suitable approach for addressing this issue would be to employ the backwards fill method.
"""

# Filling missing values with bfill
train_df['dcoilwtico_filled'] = train_df['dcoilwtico'].fillna(method='bfill')

train_df.to_csv('traindf_preprocessed.csv')

table = pa.Table.from_pandas(train_df)

pq.write_table(table, 'train.parquet', compression='GZIP')