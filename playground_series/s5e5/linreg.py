import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# going to import the whole training set right now and make a prediction to kaggle
trn = pl.read_csv("playground_series/s5e5/data/train.csv")
tst = pl.read_csv("playground_series/s5e5/data/test.csv")

feats = ["Body_Temp", "Heart_Rate", "Duration"]

X = trn.select(feats)
y = trn["Calories"]

# all of the X variables are floats, so I don't need to do any dummy'ing or handling of anything categorical

# implement a simple pipeline that scales all of the continuous predictors in X
linreg = Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())])

linreg.fit(X, y)

# let's predict on the test data
X_tst = tst.select(feats)

preds = linreg.predict(X_tst)

preds_nz = [0.1 if x < 0.1 else x for x in preds]

sub = pl.DataFrame({"id": tst["id"], "Calories": preds_nz})

# write out to file
sub.write_csv("playground_series/s5e5/submissions/linreg.csv")

# score is 0.50
# there's probably a better approach to fitting this model to handle the negative/low predictions
