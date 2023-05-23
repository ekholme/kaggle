import polars as pl
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

trn = pl.read_csv("./age_related_conditions/data/trn.csv")
val = pl.read_csv("./age_related_conditions/data/validation.csv")

imp = SimpleImputer(missing_values=np.nan, strategy='median')
yj = preprocessing.PowerTransformer()
oh = preprocessing.OneHotEncoder(drop='first')
pca = PCA()
lr = LogisticRegression(C=.9)

pre_pipe = make_pipeline(imp, yj, oh, pca)

#finagle X a little bit
X = trn.select(pl.exclude(['Id', 'Class']))
y = trn['Class']

#fitting pre-pipe -- want to get a sense of how many PCA components I should keep

pre_pipe.fit(X)

#see this article for more on preprocessing pipelines in sklearn
#https://medium.com/analytics-vidhya/how-to-apply-preprocessing-steps-in-a-pipeline-only-to-specific-features-4e91fe45dfb8