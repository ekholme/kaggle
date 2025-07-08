import polars as pl
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

trn = pl.read_csv("playground_series/s5e5/data/train.csv")
tst = pl.read_csv("playground_series/s5e5/data/test.csv")

# subset to only the predictors I currently care about
feats = ["Duration", "Heart_Rate", "Body_Temp"]
X = trn.select(feats)
y = trn["Calories"]

# subset test data
X_tst = tst.select(feats)

poisreg = PoissonRegressor()
ss = StandardScaler()

pipe = Pipeline([("scaler", ss), ("poisson_reg", poisreg)])

pipe.fit(X, y)

# get predictions for test data
preds = pipe.predict(X_tst)

# write submission out to file
sub = pl.DataFrame({"id": tst["id"], "Calories": preds})

sub.write_csv("playground_series/s5e5/submissions/poisreg_baseline.csv")
# score is 0.26 here -- considerably better
