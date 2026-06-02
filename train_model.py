import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("train.csv")

# Features used for prediction
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

# Inputs and target
X = df[features]
y = df["SalePrice"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Show sample predictions
print("=== FIRST 5 PREDICTIONS ===")
for pred, actual in zip(predictions[:5], y_test.iloc[:5]):
    print(f"Predicted: ${pred:,.2f} | Actual: ${actual:,.2f}")

# Calculate accuracy
mae = mean_absolute_error(y_test, predictions)

print("\n=== MODEL PERFORMANCE ===")
print(f"Mean Absolute Error: ${mae:,.2f}")

# Feature importance
print("\n=== FEATURE IMPORTANCE ===")
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# Predict a custom house
new_house = pd.DataFrame([{
    "LotArea": 10000,
    "OverallQual": 8,
    "OverallCond": 5,
    "YearBuilt": 2015,
    "YearRemodAdd": 2018,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "HalfBath": 1,
    "TotRmsAbvGrd": 8,
    "GarageCars": 2,
    "GarageArea": 500,
    "GrLivArea": 2000
}])

predicted_price = model.predict(new_house)

print("\n=== CUSTOM HOUSE PREDICTION ===")
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")