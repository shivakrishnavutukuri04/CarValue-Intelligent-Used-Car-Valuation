import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

# Name of the experiment
mlflow.set_experiment("KNN_CarPrice_Pipeline")

# Load Dataset
data = pd.read_csv("Used_Car_Price_Prediction.csv")

# Data Cleaning
data = data.drop_duplicates()
data['service_history'] = data['service_history'].fillna('Unknown')

# Segregate features and Target
numeric_cols = ['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count', 'accidents_reported']
categorical_cols = ['fuel_type', 'brand', 'transmission', 'color', 'service_history', 'insurance_valid']

X = data[numeric_cols + categorical_cols]
y = data['price_usd']

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Custom IQR Clipper Transformer
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = {}
        self.upper_ = {}
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_[col] = Q1 - self.factor * IQR
            self.upper_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(self.lower_[col], self.upper_[col])
        return X.values

# Numeric Preprocessing Pipeline
numeric_pipeline = Pipeline(
    [
        ('Clipper', IQRClipper()),
        ('Scaler', StandardScaler())
    ]
)

# Column Transformer
preprocessor = ColumnTransformer(
    [
        ('num', numeric_pipeline, numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Define Pipeline
pipeline_1 = Pipeline(
    [
        ('Preprocessor', preprocessor),
        ('Model', KNeighborsRegressor())
    ]
)

# Define Objective
def objective(trial):
    # Hyperparameters suggested by Optuna
    scaler_type = trial.suggest_categorical("Preprocessor__num__Scaler__type", ["standard", "minmax"])
    n_neighbors = trial.suggest_int("Model__n_neighbors", 3, 21, 2)
    p = trial.suggest_int("Model__p", 1, 3)
    weights = trial.suggest_categorical("Model__weights", ["uniform", "distance"])

    # Set pipeline params for this trial
    pipeline_1.set_params(
        Preprocessor__num__Scaler=StandardScaler() if scaler_type == "standard" else MinMaxScaler(),
        Model__n_neighbors=n_neighbors,
        Model__p=p,
        Model__weights=weights
    )

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline_1, X_train, y_train, scoring="r2", cv=kf).mean()

    return cv_score

mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="cv_r2"
)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, callbacks=[mlflow_callback])

best_params = study.best_trial.params
print("Best hyperparameters:", best_params)
print("Best CV R2:", study.best_trial.value)

# Autolog final sklearn model
mlflow.sklearn.autolog()

# Training with Best parameters
scaler_type = best_params['Preprocessor__num__Scaler__type']
Scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
pipeline_1.set_params(Preprocessor__num__Scaler=Scaler)
pipeline_1.set_params(**{'Model__n_neighbors': best_params['Model__n_neighbors'], 'Model__p': best_params['Model__p'], 'Model__weights': best_params['Model__weights']})
pipeline_1.fit(X_train, y_train)
score = pipeline_1.score(X_train, y_train)
print('Training score', score)

# Testing the model
y_pred_test = pipeline_1.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print('Testing R2', test_r2)
print('Testing MAE', test_mae)
print('Testing RMSE', test_rmse)

# Log metrics to MLflow
mlflow.log_metric("train_r2", score)
mlflow.log_metric("test_r2", test_r2)
mlflow.log_metric("test_mae", test_mae)
mlflow.log_metric("test_rmse", test_rmse)
