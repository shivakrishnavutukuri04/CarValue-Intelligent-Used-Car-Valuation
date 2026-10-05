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
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

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
    X, y, test_size=0.3
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
pipeline = Pipeline(
    [
        ('Preprocessor', preprocessor),
        ('Model', KNeighborsRegressor())
    ]
)

# Linear Regression Objective
def objective_lr(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(
        Model=LinearRegression(
            fit_intercept=trial.suggest_categorical('fit_intercept', [True, False]),
            positive=trial.suggest_categorical('positive', [True, False])
        )
    )
    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# Ridge Objective
def objective_ridge(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(
        Model=Ridge(
            alpha=trial.suggest_float('alpha', 0.01, 200, log=True),
            solver=trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sag'])
        )
    )
    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# KNN Objective
def objective_knn(trial):
    # Define Hyperparameters
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(Model=KNeighborsRegressor())
    pipeline.set_params(Model__n_neighbors=trial.suggest_int('n_neighbors', 3, 21, 2))
    pipeline.set_params(Model__weights=trial.suggest_categorical('weights', ['uniform', 'distance']))
    pipeline.set_params(Model__p=trial.suggest_int('p', 1, 3))
    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# Decision Tree Objective
def objective_dt(trial):
    # Define hyperparameters
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(Model=DecisionTreeRegressor())
    pipeline.set_params(
        Model__criterion=trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson']),
        Model__max_depth=trial.suggest_int('max_depth', 2, 30),
        Model__min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        Model__min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        Model__max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']))
    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# Random Forest Objective
def objective_rf(trial):
    # Scaler (optional for Random Forest, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())

    # Random Forest hyperparameters
    pipeline.set_params(
        Model=RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            criterion=trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson']),
            max_depth=trial.suggest_int('max_depth', 5, 40),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            random_state=42,
            n_jobs=-1
        )
    )

    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# Gradient Boosting Objective
def objective_gb(trial):
    # Scaler (optional for Gradient Boosting, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())

    # Gradient Boosting hyperparameters
    pipeline.set_params(
        Model=GradientBoostingRegressor(
            criterion='squared_error',
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    )

    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# XGBoost Objective
def objective_xgb(trial):
    # Scaler (optional for XGBoost, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler())

    # XGBoost hyperparameters
    pipeline.set_params(
        Model=XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            gamma=trial.suggest_float('gamma', 0, 0.5),
            random_state=42,
            n_jobs=-1
        )
    )

    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()
    return score

# Map model names to objective functions
objectives = {
    "LinearRegression": objective_lr,
    "Ridge": objective_ridge,
    "DecisionTree": objective_dt,
    "RandomForest": objective_rf,
    "GradientBoosting": objective_gb,
    "KNN": objective_knn,
    "XGBoost": objective_xgb
}

# Set experiment
mlflow.set_experiment("CarPrice_PL_RUNS")

results = {}
model_dict = {}
scaler_dict = {}

# Loop through each algorithm
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    with mlflow.start_run(run_name=model_name):

        # # Disable autologging once
        # mlflow.sklearn.autolog(disable = True)

        mlflow_cb = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),  # Log into the active experiment
            metric_name="cv_r2",                     # Primary metric (str)
            create_experiment=False,                 # Reuse the active experiment instead of creating a new one
            mlflow_kwargs={                           # **MLflow start_run() kwargs**
                "nested": True                        # Child runs under parent
            }
        )

        # Create Optuna study
        study = optuna.create_study(direction="maximize")

        # Train the final model
        start_fit = time.time()
        study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
        fit_time = time.time() - start_fit

        print(f"Best CV R2 for {model_name}: {study.best_value:.4f}")
        best_params = study.best_params
        results[model_name] = {"best_params": best_params, "best_cv_r2": study.best_value}

        # Fit the pipeline with the best parameters
        if model_name == "LinearRegression":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=LinearRegression(
                    fit_intercept=best_params["fit_intercept"],
                    positive=best_params["positive"]
                )
            )
        elif model_name == "Ridge":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=Ridge(
                    alpha=best_params["alpha"],
                    solver=best_params["solver"]
                )
            )
        elif model_name == "KNN":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=KNeighborsRegressor(
                    n_neighbors=best_params["n_neighbors"],
                    weights=best_params["weights"],
                    p=best_params["p"]
                )
            )
        elif model_name == "DecisionTree":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=DecisionTreeRegressor(
                    criterion=best_params["criterion"],
                    max_depth=best_params["max_depth"],
                    min_samples_split=best_params["min_samples_split"],
                    min_samples_leaf=best_params["min_samples_leaf"],
                    max_features=best_params["max_features"]
                )
            )
        elif model_name == "RandomForest":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=RandomForestRegressor(
                    n_estimators=best_params["n_estimators"],
                    criterion=best_params["criterion"],
                    max_depth=best_params["max_depth"],
                    min_samples_split=best_params["min_samples_split"],
                    min_samples_leaf=best_params["min_samples_leaf"],
                    max_features=best_params["max_features"],
                    bootstrap=best_params["bootstrap"],
                    random_state=42,
                    n_jobs=-1
                )
            )
        elif model_name == "GradientBoosting":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=GradientBoostingRegressor(
                    criterion='squared_error',
                    n_estimators=best_params["n_estimators"],
                    learning_rate=best_params["learning_rate"],
                    max_depth=best_params["max_depth"],
                    min_samples_split=best_params["min_samples_split"],
                    min_samples_leaf=best_params["min_samples_leaf"],
                    max_features=best_params["max_features"],
                    subsample=best_params["subsample"],
                    random_state=42
                )
            )
        elif model_name == "XGBoost":
            pipeline.set_params(
                Preprocessor__num__Scaler=StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler(),
                Model=XGBRegressor(
                    n_estimators=best_params["n_estimators"],
                    learning_rate=best_params["learning_rate"],
                    max_depth=best_params["max_depth"],
                    subsample=best_params["subsample"],
                    colsample_bytree=best_params["colsample_bytree"],
                    gamma=best_params["gamma"],
                    random_state=42,
                    n_jobs=-1
                )
            )

        # # Enable autologging once
        # mlflow.sklearn.autolog()

        # Train the final model
        pipeline.fit(X_train, y_train)

        # Evaluate on test data
        start_test = time.time()
        y_pred = pipeline.predict(X_test)
        test_time = time.time() - start_test

        train_r2 = pipeline.score(X_train, y_train)
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{model_name} Training R2: {train_r2:.4f}, Testing R2: {test_r2:.4f}")
        print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.2f}s")

        # Save model manually to track model size
        model_path = f"{model_name}_final_model.pkl"
        joblib.dump(pipeline, model_path)
        model_size = os.path.getsize(model_path)
        for i, model in enumerate(objectives.keys()):
            model_dict[model] = i

        for i, scaler_type in enumerate(['standard', 'minmax']):
            scaler_dict[scaler_type] = i

        mlflow.log_metric(f"model_id", model_dict[model_name])
        mlflow.log_metric(f"Scalar_id", scaler_dict[best_params["scaler_type"]])
        mlflow.log_metric(f"train_r2", train_r2)
        mlflow.log_metric(f"test_r2", test_r2)
        mlflow.log_metric(f"test_mae", test_mae)
        mlflow.log_metric(f"test_rmse", test_rmse)
        mlflow.log_metric(f"train_time", fit_time)
        mlflow.log_metric(f"test_time", test_time)
        mlflow.log_metric(f"model_size", model_size)
        mlflow.sklearn.log_model(pipeline, name=f"{model_name}_car_model", serialization_format="cloudpickle")  # , registered_model_name = f"CarPrice_{model_name}_Best")
        os.remove(model_path)

        results[model_name].update({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "fit_time": fit_time,
            "test_time": test_time,
            "model_size_bytes": model_size
        })

# Summary
print("\n--- Summary ---")
for model_name, res in results.items():
    print(f"{model_name}: CV R2={res['best_cv_r2']:.4f}, Train R2={res['train_r2']:.4f}, "
          f"Test R2={res['test_r2']:.4f}, Test MAE={res['test_mae']:.2f}, Test RMSE={res['test_rmse']:.2f}, "
          f"Fit Time={res['fit_time']:.2f}s, Model Size={res['model_size_bytes']} bytes")