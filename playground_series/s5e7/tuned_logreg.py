import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

seed = 48

# load data

trn = pl.read_csv("playground_series/s5e7/data/train.csv")
tst = pl.read_csv("playground_series/s5e7/data/test.csv")

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
model = LogisticRegression(penalty="elasticnet", solver="saga", random_state=seed)

pipe = Pipeline([("preprocessor", preprocessor), ("logreg", model)])

# define parameters to tune
param_space = {
    "logreg__C": loguniform(0.001, 10),
    "logreg__l1_ratio": loguniform(0.001, 1),
}

# define a random search
random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_space,
    n_iter=50,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
    verbose=1,
    random_state=seed,
)

# fit the random search
random_search.fit(X, y)

# get the best model
best_model = random_search.best_estimator_

# get predictions for test data
X_tst = tst.select(cat_cols + num_cols)
y_hat_tst = best_model.predict(X_tst)

sub = pl.DataFrame(
    {
        "id": tst["id"],
        "Personality": y_hat_tst,
    }
)

sub.write_csv("playground_series/s5e7/submissions/tuned_logreg.csv")
# this ends up being slightly worse than the naive/untuned model
