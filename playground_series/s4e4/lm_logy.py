import polars as pl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

trn = pl.read_csv("data/train.csv")
tst = pl.read_csv("data/test.csv")

#feat eng
#1. log the outcome
#2. Box-Cox numerics, PCA and extract 1 component
#3. dummy-code sex
#4. interact the PCA component with the sex dummies
#we'll try without the interaction first

# Structure Data -----------

y = trn["Rings"].to_list()
X = trn.drop(["id", "Rings"]).to_pandas()

#we'll use the log of y
ln_Y = np.log(y)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, ln_Y, test_size = 0.2, random_state=408)

# Categorical Transformations --------

cat_feats = ["Sex"]

cat_transform = Pipeline([
    ("oh_encode", preprocessing.OneHotEncoder(drop="first"))
])

# Continuous Transformations ----------

cont_feats = ["Length", "Diameter", "Height", "Whole weight", "Whole weight.1", "Whole weight.2", "Shell weight"]

pca = PCA(n_components=1)

cont_transform = Pipeline([
    ("yj", preprocessing.PowerTransformer("yeo-johnson")),
    ("pca", pca)
])

# Make Preprocessor and Full Pipeline -------

preprocessor = ColumnTransformer([
    ("cat", cat_transform, cat_feats),
    ("cont", cont_transform, cont_feats)
])

lm = LinearRegression()

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("lm", lm)
])

#fit model
pipe.fit(X_trn, y_trn)

#getting predictions
y_hat = np.exp(pipe.predict(tst))

#create submission
sub = pl.DataFrame({
    "id": tst["id"],
    "Rings": y_hat
})

sub.write_csv("submissions/lm_logy_yj_pca.csv")