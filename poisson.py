from functools import partial

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.api import GLM
from statsmodels.genmod.families import Poisson

from horsekickerpy.horsekick import clean_horse_kicks
from horsekickerpy.wrapper import SMWrapper

df = pd.read_csv("data/VonBort.csv")


PoissonRegressor = partial(GLM, family=Poisson())
horse_kick_pipeline = make_pipeline(OneHotEncoder(sparse=False),
                                    SMWrapper(PoissonRegressor, fit_intercept=True))

clean_df = clean_horse_kicks(df)

horse_kick_pipeline.fit(np.asarray(clean_df), np.asarray(df["deaths"]))

print(horse_kick_pipeline.predict(clean_df))

joblib.dump(horse_kick_pipeline, "model/horse_kick_model.pkl")
