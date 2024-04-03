import polars as pl
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

trn = pl.read_csv("./playground_series/s4e1/data/train.csv")
tst = pl.read_csv("./playground_series/s4e1/data/test.csv")

trn.head()

model = XGBClassifier()
X = trn.drop(["CustomerId", "Exited"])
y = trn["Exited"]

# todo
# onehot encode string cols

# fit model
model.fit(X, y)
