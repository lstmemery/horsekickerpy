import pandas as pd
from statsmodels.genmod.families import Poisson
from statsmodels.api import GLM
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from functools import partial
from horsekickerpy.horsekick import clean_horse_kicks
import numpy as np
from sklearn.externals import joblib


df = pd.read_csv("data/VonBort.csv")

# https://stackoverflow.com/a/48949667/2687504
class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


PoissonRegressor = partial(GLM, family=Poisson())
horse_kick_pipeline = make_pipeline(OneHotEncoder(sparse=False),
                                    SMWrapper(PoissonRegressor, fit_intercept=True))

clean_df = clean_horse_kicks(df)

horse_kick_pipeline.fit(np.asarray(clean_df), np.asarray(df["deaths"]))

print(horse_kick_pipeline.predict(clean_df))

joblib.dump(horse_kick_pipeline, "results/horse_kick_model.pkl")

