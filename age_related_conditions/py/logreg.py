import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer_names
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

trn = pl.read_csv("./age_related_conditions/data/trn.csv")
val = pl.read_csv("./age_related_conditions/data/validation.csv")

imp = SimpleImputer(missing_values=np.nan, strategy='median')
yj = preprocessing.PowerTransformer()
oh = preprocessing.OneHotEncoder(drop='first')
pca = PCA(n_components=42)
lr = LogisticRegression()


#finagle X a little bit
X = trn.select(pl.exclude(['Id', 'Class']))
y = trn['Class']

#defining a preprocessing pipeline

cat_feats = ['EJ']
num_feats = [i for i in X.columns if i not in set(cat_feats)]

cat_transformer = Pipeline(
    [
        ('cat_imputer', SimpleImputer(
            strategy='constant', fill_value='missing'
        )),
        ('onehot', preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore'))
    ]
)

num_transformer = Pipeline(
    [
        ('num_imputer', SimpleImputer(strategy='median', missing_values=np.nan)),
        ('yj', preprocessing.PowerTransformer()),
        ('pca', PCA())
    ]
)

preprocessor = ColumnTransformer(
    [
        ('cats', cat_transformer, cat_feats),
        ('nums', num_transformer, num_feats)
    ],
    remainder='drop'
)

#unfortunately, I think sklearn pipelines want pandas dataframes
#so I need to do this conversion before running?
Xpd = X.to_pandas()


#make a pipeline with the preprocessing and logreg steps
pipe = Pipeline(
    [
        ('preprocess', preprocessor),
        ('lr', lr)
    ]
)

#set up parameters for gridsearch
params = {
    'lr__solver': ['liblinear'],
    'lr__C': [.01, .1, .2, .5, .9, .99],
    'lr__penalty': ['l1', 'l2'],
    'lr__random_state': [408]
}

folds = RepeatedStratifiedKFold(n_splits = 5, n_repeats=5, random_state= 408)

cv = GridSearchCV(
    pipe,
    params,
    cv = folds,
    scoring = 'neg_log_loss'
)

cv.fit(Xpd, y)

cv.best_score_
cv.best_params_

Xt = val.select(pl.exclude(['Id', 'Class'])).to_pandas()
yt = val['Class']

cv.score(Xt, yt)

preds = cv.predict_proba(Xt)

sub = pl.DataFrame(
    {
        "Id": val['Id'],
        "class_0": preds[:, 0],
        "class_1": preds[:, 1]
    }
)