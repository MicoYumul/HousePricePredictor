import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("train.csv")

features = [
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "BedroomAbvGr",
    "FullBath",
    "HalfBath",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea",
    "GrLivArea"
]

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

joblib.dump(
    {
        "model": model,
        "features": features,
        "mae": mae
    },
    "model.pkl"
)

print("Model saved successfully!")
print(f"MAE: ${mae:,.2f}")