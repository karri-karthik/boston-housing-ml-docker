import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("data/housing.csv")

# Handle missing values (IMPORTANT)
df = df.fillna(df.mean())

# Features & Target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully")

import os
print("Saved model at:", os.path.abspath("model.pkl"))
