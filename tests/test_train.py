import pandas as pd
import pytest

@pytest.fixture(scope='module')
def df():
    # Load the train.parquet file
    return pd.read_parquet('train.parquet')

def test_columns_exist(df):
    # Check if the required columns exist in the dataset
    required_columns = ['store_nbr', 'onpromotion', 'month', 'year', 'day_of_week']
    for column in required_columns:
        assert column in df.columns

def test_columns_data_type(df):
    # Check if the data types of the required columns are correct
    column_data_types = {
        'store_nbr': int,
        'onpromotion': bool,
        'month': int,
        'year': int,
        'day_of_week': int
    }
    for column, data_type in column_data_types.items():
        assert df[column].dtype == data_type
