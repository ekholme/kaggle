import polars as pl
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

trn = pl.read_csv("./age_related_conditions/data/trn.csv")
val = pl.read_csv("./age_related_conditions/data/validation.csv")

#define pipeline steps
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
zscore = preprocessing.StandardScaler()
lr = LogisticRegression(C=.1, max_iter=2000)

pipe = make_pipeline(imp, zscore, lr)

X = trn.select(pl.exclude(['Id', 'Class']))
X = X.select(pl.col(pl.Float64))
y = trn['Class']

pipe.fit(X, y)

#cleaning up test data
Xt = val.select(pl.exclude(['Id', 'Class']))
Xt = Xt.select(pl.col(pl.Float64))
yt = val['Class']

pipe.score(Xt, yt)

preds = pipe.predict_proba(Xt)

ll = log_loss(val['Class'], preds)
#ll of .244, which is better than the baseline
#this also uses really strong regularization

sub = pl.DataFrame(
    {
        "Id": val['Id'],
        "class_0": preds[:, 0],
        "class_1": preds[:, 1]
    }
)
#this is basically what I want I think
#next step is to put it an an actual kaggle notebook