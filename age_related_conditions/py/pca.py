import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

trn = pl.read_csv("./age_related_conditions/data/trn.csv")
val = pl.read_csv("./age_related_conditions/data/validation.csv")

imp = SimpleImputer(missing_values=np.nan, strategy='median')
yj = preprocessing.PowerTransformer()
oh = preprocessing.OneHotEncoder(drop='first')
pca = PCA()
lr = LogisticRegression(C=.9)


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

preprocessor.fit_transform(Xpd)

ev_pct = preprocessor.named_transformers_['nums'].named_steps['pca'].explained_variance_ratio_.cumsum()


out_pcs = sum(ev_pct < .95)
#ok, so we want to keep this many components, I guess


#see this article for more on preprocessing pipelines in sklearn
#https://medium.com/analytics-vidhya/how-to-apply-preprocessing-steps-in-a-pipeline-only-to-specific-features-4e91fe45dfb8