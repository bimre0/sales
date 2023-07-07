import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
import pickle
import argparse
import logging

# Configure the logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


parser = argparse.ArgumentParser(description='Sales Forecasting')
# Add arguments
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for the forecast (default: 0.5)')
parser.add_argument('--beta', type=float, default=0.3, help='Beta value for the forecast (default: 0.3)')
parser.add_argument('--gamma', type=float, default=0.2, help='Gamma value for the forecast (default: 0.2)')
parser.add_argument('--split_val', type=float, default=0.8, help='Split value for training and testing (default: 0.9)')

# Parse the arguments
args = parser.parse_args()

# Access the values of the arguments
alpha = args.alpha
beta = args.beta
gamma = args.gamma
split_val = args.split_val

# Read the train.parquet file
df = pd.read_parquet('train.parquet')

# Convert date column to separate month, year, and day of week columns
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek

# Perform forecasting using Linear Regression model
model = LinearRegression()
X = df[['store_nbr', 'onpromotion', 'month', 'year', 'day_of_week']
y = df['sales']
split_index = int(len(df) * split_val)  # Use the last 10% of data as the test set

X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Fit the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
forecast_train = model.predict(X_train)
forecast_test = model.predict(X_test)

# Benchmarking metrics on the train set
train_mae = mean_absolute_error(y_train, forecast_train)
train_rmse = mean_squared_error(y_train, forecast_train, squared=False)
train_r2 = r2_score(y_train, forecast_train)

# Benchmarking metrics on the test set
test_mae = mean_absolute_error(y_test, forecast_test)
test_rmse = mean_squared_error(y_test, forecast_test, squared=False)
test_r2 = r2_score(y_test, forecast_test)

# Print benchmarking metrics
logging.debug("Train set MAE:", train_mae)
logging.debug("Train set RMSE:", train_rmse)
logging.debug("Train set R^2:", train_r2)
logging.debug("Test set MAE:", test_mae)
logging.debug("Test set RMSE:", test_rmse)
logging.debug("Test set R^2:", test_r2)

# Save evaluation results to a file
with open('evaluation_results.txt', 'w') as file:
    file.write("Train set MAE: {}\n".format(train_mae))
    file.write("Train set RMSE: {}\n".format(train_rmse))
    file.write("Train set R^2: {}\n".format(train_r2))
    file.write("Test set MAE: {}\n".format(test_mae))
    file.write("Test set RMSE: {}\n".format(test_rmse))
    file.write("Test set R^2: {}\n".format(test_r2))

# Write the forecasted data to a parquet file
forecast_test.to_parquet('forecast.parquet')

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully.")
