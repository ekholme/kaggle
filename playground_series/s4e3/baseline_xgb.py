import polars as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb

trn = pl.read_csv("./data/train.csv")
tst_raw = pl.read_csv("./data/test.csv")

#the goal here is to do a generic baseline xgboost model, mostly to recall how to do it in python

nms = trn.columns

targets = trn.columns[(len(nms)-7):len(nms)]

tars = trn[targets]

tar_cols_sum = tars.sum(axis=1)

# get columns and rows I want
X = trn.drop(targets)
X = X.drop(columns=['id'])

x = X.filter(tar_cols_sum <= 1)
y = tars.filter(tar_cols_sum <= 1)

#scale x
scaler = preprocessing.StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

#also scale testing data
tst = tst_raw.drop(columns=['id'])
tst = scaler.transform(tst)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=408)

# set up xgb model
evalset = [(X_train, y_train), (X_test,y_test)]
xgb.set_config(verbosity=0)

xgb_model = xgb.XGBClassifier(tree_method="hist", multi_strategy="multi_output_tree", n_estimaters=5000)

xgb_model.fit(X_train, y_train, eval_set=evalset, eval_metric="logloss", early_stopping_rounds=100)

y_pred = xgb_model.predict(X_test)

# get roc auc
roc_auc = roc_auc_score(y_test, y_pred)

print(roc_auc)

# predict tst values
tst_pred = xgb_model.predict_proba(tst)

# make submission
tst_pred = pl.DataFrame(tst_pred)
tst_pred.columns = targets

sub = tst_pred.with_columns(id = tst_raw["id"])

sub.write_csv("./submissions/baseline_xgb.csv")