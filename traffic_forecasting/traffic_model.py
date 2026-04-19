import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("----- Traffic Forecasting using Real Dataset -----")

# 1. Load Data
df = pd.read_csv("traffic.csv")
print("\nData loaded successfully.")

# Print column names and first few rows for clarity
print("\nColumns in dataset:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# 2. Convert DateTime column to datetime format and sort data
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values(by='DateTime')

# 3. Perform feature engineering
# Extracting useful time-based features from DateTime
df['hour'] = df['DateTime'].dt.hour
df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month
df['day_of_week'] = df['DateTime'].dt.dayofweek

# 4. Handle categorical column (Junction) using one-hot encoding
# drop_first=True helps avoid the dummy variable trap
df = pd.get_dummies(df, columns=['Junction'], drop_first=True)

# 5. Define features (X) and target (Vehicles)
# ID and DateTime are not needed for our model's predictions
X = df.drop(columns=['Vehicles', 'DateTime', 'ID'])
y = df['Vehicles']

# 6. Split data chronologically (80% train, 20% test)
# Since it's time series style, we do not randomly shuffle
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# 7. Train TWO models
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Training Random Forest Regressor (this might take a little bit)...")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# 8. Evaluate both models using RMSE and MAE
# Make predictions on the test set
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Calculate metrics for Linear Regression
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)

# Calculate metrics for Random Forest Regressor
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)

# 9. Print comparison of both models clearly
print("\n--- Model Evaluation Results ---")

print("\nLinear Regression:")
print(f"RMSE: {round(lr_rmse, 2)}")
print(f"MAE:  {round(lr_mae, 2)}")

print("\nRandom Forest Regressor:")
print(f"RMSE: {round(rf_rmse, 2)}")
print(f"MAE:  {round(rf_mae, 2)}")

print("\nDone! Comparing the models, the one with lower RMSE and MAE performed better.")