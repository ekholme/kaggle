import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

seed = 48

# load data

trn = pl.read_csv("playground_series/s5e7/data/train.csv")
tst = pl.read_csv("playground_series/s5e7/data/test.csv")

trn.columns

# ok, so the naive process here is going to be:
# - median impute and numeric values
# - mode impute any strings
# - dummy out string columns

# show column types for each column in the df
[(i, j) for i, j in zip(trn.columns, trn.dtypes)]

y = trn["Personality"]
cat_cols = ["Stage_fear", "Drained_after_socializing"]
num_cols = [i for i in trn.columns if i not in cat_cols + ["Personality", "id"]]

X = trn.select(cat_cols + num_cols)

# categorical transformation steps
cat_transform = Pipeline(
    [
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("oh_encoder", OneHotEncoder(drop="first")),
    ]
)

# numerical transformation steps
num_transform = Pipeline(
    [("num_imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    [("cats", cat_transform, cat_cols), ("nums", num_transform, num_cols)]
)

# define a logistic regression model with a lasso penalty
model = LogisticRegression(penalty="l1", solver="liblinear", random_state=seed)

pipe = Pipeline([("preprocessor", preprocessor), ("logreg", model)])

pipe.fit(X, y)

y_hat = pipe.predict(X)

# evaluate against ground truth y values
accuracy_score(y, y_hat)

# get predictions for test data
X_tst = tst.select(cat_cols + num_cols)
y_hat_tst = pipe.predict(X_tst)

sub = pl.DataFrame(
    {
        "id": tst["id"],
        "Personality": y_hat_tst,
    }
)

sub.write_csv("playground_series/s5e7/submissions/baseline_logreg.csv")
