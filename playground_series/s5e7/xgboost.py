import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

seed = 48

trn = pl.read_csv("playground_series/s5e7/data/train.csv")
tst = pl.read_csv("playground_series/s5e7/data/test.csv")

y = trn["Personality"]
y_bool = [int(i == "Extrovert") for i in y]
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

# preprocess the training data, then pass the transformed data into an xgboost classifer
model = xgb.XGBClassifier(random_state=seed)

pipe = Pipeline([("preprocessor", preprocessor), ("xgb", model)])

pipe.fit(X, y_bool)

# predict training data
y_pred = ["Extrovert" if i == 1 else "Introvert" for i in pipe.predict(X)]
accuracy_score(y, y_pred)

# get predictions for test data
X_tst = tst.select(cat_cols + num_cols)
y_hat_tst = pipe.predict(X_tst)
y_class_tst = ["Extrovert" if i == 1 else "Introvert" for i in y_hat_tst]

sub = pl.DataFrame(
    {
        "id": tst["id"],
        "Personality": y_class_tst,
    }
)

sub.write_csv("playground_series/s5e7/submissions/baseline_xgb.csv")
