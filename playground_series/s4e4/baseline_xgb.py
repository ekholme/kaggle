import polars as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import ensemble
import numpy as np

trn = pl.read_csv("data/train.csv")
tst = pl.read_csv("data/test.csv")

trn.describe()

y = trn["Rings"].to_list()
trn_no_y = trn.drop(["id", "Rings"])

#train test split
X = trn_no_y.to_pandas()

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=408)

#one-hot encode the Sex variable
cat_feats = ["Sex"]

cat_transform = Pipeline([
    ('oh_encode', preprocessing.OneHotEncoder(drop='first'))
])

#and standardize the continuous variables
cont_feats = ["Length", "Diameter", "Height", "Whole weight", "Whole weight.1", "Whole weight.2", "Shell weight"]

cont_transform = Pipeline([
    ("standardizer", preprocessing.StandardScaler())
])

#packaging the above steps up into a pipeline
preprocessor = ColumnTransformer([
    ('cat', cat_transform, cat_feats),
    ('cont', cont_transform, cont_feats)
])

#setting up the xgb model
#we'll just use the default values
xgb_mod = ensemble.GradientBoostingRegressor()

#define the full pipeline
pipe = Pipeline([('preprocessor', preprocessor), ('xgb', xgb_mod)])

#and fit the pipeline
pipe.fit(X_trn, y_trn)

#predicting from the model
y_hat = np.round(pipe.predict(X_tst), decimals = 0)

tst_err = root_mean_squared_log_error(y_tst, y_hat)
print(f"Test error: {tst_err}")

#predict on the actual test set
preds = np.round(pipe.predict(tst), decimals = 0)

#create submission df
sub = pl.DataFrame({
    "id": tst["id"],
    "Rings": preds
})

sub.write_csv("submissions/baseline_xgb.csv")
#score was 0.15486
#next step is to probably use BoxCox scaling for the X values
#potentially also interact sex with size vars?