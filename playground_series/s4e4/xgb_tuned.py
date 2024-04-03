import polars as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
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

#and power transform the continuous variables
cont_feats = ["Length", "Diameter", "Height", "Whole weight", "Whole weight.1", "Whole weight.2", "Shell weight"]

cont_transform = Pipeline([
    ("yj", preprocessing.PowerTransformer("yeo-johnson")),
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
pipe = Pipeline([('preprocessor', preprocessor), ('model', xgb_mod)])

# set up grid search
param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__learning_rate': [0.1, 0.01, 0.001],
    'model__max_depth': [3, 5, 8],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# perform randomized search cv using the pipeline and the param_grid
grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring='neg_root_mean_squared_log_error')

# fit the grid search
grid_search.fit(X_trn, y_trn)

# print the best parameters
print(grid_search.best_params_)

# predict on the actual test set
preds = grid_search.predict(tst)

#create submission df
sub = pl.DataFrame({
    "id": tst["id"],
    "Rings": preds
})

sub.write_csv("submissions/xgb_tuned.csv")
#0.1494 -- next step is probably to see where we're getting bad predictions