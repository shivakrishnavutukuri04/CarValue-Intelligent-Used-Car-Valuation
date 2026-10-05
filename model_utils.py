import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.lower_ = {}
        self.upper_ = {}

        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1

            self.lower_[col] = q1 - self.factor * iqr
            self.upper_[col] = q3 + self.factor * iqr

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            X[col] = X[col].clip(
                self.lower_[col],
                self.upper_[col]
            )

        return X.values