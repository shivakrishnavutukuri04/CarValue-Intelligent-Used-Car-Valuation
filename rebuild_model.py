import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Ridge

from model_utils import IQRClipper


# Load dataset
data = pd.read_csv("training/Used_Car_Price_Prediction.csv")

# Basic cleaning
data = data.drop_duplicates()
data["service_history"] = data["service_history"].fillna("Unknown")


# Features
numeric_cols = [
    "make_year",
    "mileage_kmpl",
    "engine_cc",
    "owner_count",
    "accidents_reported"
]

categorical_cols = [
    "fuel_type",
    "brand",
    "transmission",
    "color",
    "service_history",
    "insurance_valid"
]

X = data[numeric_cols + categorical_cols]
y = data["price_usd"]


# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("Clipper", IQRClipper()),
    ("Scaler", MinMaxScaler())
])


# Full preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_cols
    )
])


# Model
model = Pipeline([
    ("Preprocessor", preprocessor),
    (
        "Model",
        Ridge(alpha=0.15686508206728317)
    )
])


# Train
print("Training model...")
model.fit(X, y)
print("Training completed.")


# Save model
joblib.dump(model, "model.pkl")
print("New model.pkl saved successfully.")


# Test model
test_data = pd.DataFrame({
    "make_year": [2015],
    "mileage_kmpl": [18.0],
    "engine_cc": [1500],
    "fuel_type": ["Petrol"],
    "owner_count": [1],
    "brand": ["Toyota"],
    "transmission": ["Manual"],
    "color": ["White"],
    "service_history": ["Full"],
    "accidents_reported": [0],
    "insurance_valid": ["Yes"]
})


loaded_model = joblib.load("model.pkl")

prediction = loaded_model.predict(test_data)

print("Test prediction:", prediction)