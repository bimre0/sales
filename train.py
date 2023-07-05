import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import yaml
import pickle


# Read the train.parquet file
df = pd.read_parquet('train.parquet')

# Convert date column to separate month, year, and day of week columns
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek

# Fix family column as categorical feature
df['family'] = df['family'].astype('category')

# Perform one-hot encoding on the family column
encoder = OneHotEncoder(sparse=False)
family_encoded = encoder.fit_transform(df[['family']])
family_encoded_df = pd.DataFrame(family_encoded, columns=encoder.get_feature_names_out(['family']))
df = pd.concat([df, family_encoded_df], axis=1)

# Load hyperparameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Extract hyperparameters
alpha = params['alpha']
beta = params['beta']
gamma = params['gamma']
split_val = params['split_val']
# Perform forecasting using Linear Regression model
model = LinearRegression()
X = df[['store_nbr', 'onpromotion', 'month', 'year', 'day_of_week'] + list(encoder.get_feature_names_out(['family']))]
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
print("Train set MAE:", train_mae)
print("Train set RMSE:", train_rmse)
print("Train set R^2:", train_r2)
print("Test set MAE:", test_mae)
print("Test set RMSE:", test_rmse)
print("Test set R^2:", test_r2)

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