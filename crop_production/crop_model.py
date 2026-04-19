import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    # 1. Load dataset using pandas
    print("Loading dataset...")
    try:
        # Assuming the CSV is in the same directory as this script, or we use a relative/absolute path
        df = pd.read_csv('crop_production.csv')
    except FileNotFoundError:
        print("Error: 'crop_production.csv' not found. Please ensure it's in the same directory.")
        return

    print(f"Data loaded successfully. Initial shape: {df.shape}")

    # 2. Handle missing values and clean data
    # The target variable is 'Production', so we drop rows where it is missing.
    df = df.dropna(subset=['Production'])
    
    # Fill or drop other missing values (e.g., 'Area'). Handling via dropping for simple clean data
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    # 3. Convert categorical columns (State_Name, Season, Crop) using encoding
    print("Encoding categorical columns...")
    categorical_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
    
    # We use Label Encoding to keep things simple and avoid a massive number of columns
    # which would occur with get_dummies (One-Hot Encoding) due to high cardinality of districts/crops.
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            # Strip whitespace to avoid treating "Kharif" and "Kharif " as different.
            df[col] = df[col].astype(str).str.strip()
            
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # 4. Perform basic feature engineering if needed
    # There are no obviously beneficial simple features to add here without 
    # risking data leakage (like yield=prod/area), so we proceed with the cleaned generic features.

    # 5. Define features (X) and target (Production)
    # We use all remaining columns to try and predict 'Production'
    X = df.drop(columns=['Production'])
    y = df['Production']

    # 6. Split dataset into train and test sets (80% train, 20% test)
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Train TWO models: Linear Regression and Random Forest
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    print("Training Random Forest Regressor (this may take a moment)...")
    # Using 100 trees, random state set for reproducibility
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 8. Evaluate models using RMSE and MAE
    def evaluate(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return rmse, mae

    print("\nEvaluating models...")
    lr_rmse, lr_mae = evaluate(lr_model, X_test, y_test)
    rf_rmse, rf_mae = evaluate(rf_model, X_test, y_test)

    # 9. Print comparison clearly
    print("\n" + "="*45)
    print("         MODEL PERFORMANCE COMPARISON        ")
    print("="*45)
    print("Linear Regression:")
    print(f"  RMSE: {lr_rmse:,.2f}")
    print(f"  MAE:  {lr_mae:,.2f}")
    print("-" * 45)
    print("Random Forest Regressor:")
    print(f"  RMSE: {rf_rmse:,.2f}")
    print(f"  MAE:  {rf_mae:,.2f}")
    print("="*45)

if __name__ == "__main__":
    main()
