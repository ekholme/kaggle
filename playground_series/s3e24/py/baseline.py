import polars as pl
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

trn = pl.read_csv("./playground_series/s3e24/data/train.csv")
tst = pl.read_csv("./playground_series/s3e24/data/test.csv")

trn.head()

y = trn["smoking"]
X = trn.drop(["smoking", "id"])

# check column types
X.schema
# ok so this is all numeric, which is good for now

# preprocess data
scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)

# train model
logreg = LogisticRegression()

logreg.fit(X_scaled, y)

# predict
X_tst = tst.drop(["id"])

X_tst_scaled = scaler.transform(X_tst)

y_pred = logreg.predict_proba(X_tst_scaled)

y_pred2 = y_pred[:, 1]

# save predictions
sub = pl.DataFrame({"id": tst["id"], "smoking": y_pred2})

sub.write_csv("./playground_series/s3e24/submissions/baseline_logreg.csv")

# check performance