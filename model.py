import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("data/housing.csv")

# Handle missing values
df = df.fillna(df.mean())

# Features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------
# Model 1: Linear Regression
# -------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

# -------------------
# Model 2: Random Forest
# -------------------
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

# -------------------
# Compare models
# -------------------
print("Model Comparison")
print("----------------")
print(f"Linear Regression -> R2: {lr_r2:.3f}, MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}")
print(f"Random Forest     -> R2: {rf_r2:.3f}, MAE: {rf_mae:.3f}, RMSE: {rf_rmse:.3f}")

# Select best model (based on R2)
best_model = rf if rf_r2 > lr_r2 else lr

# Save best model
with open("data/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ Best model saved as data/model.pkl")
