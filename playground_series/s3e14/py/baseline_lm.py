import polars as pl
from sklearn.linear_model import LinearRegression

trn = pl.read_csv("./playground_series/s3e14/data/train.csv")
tst = pl.read_csv("./playground_series/s3e14/data/test.csv")

trn.shape
trn.columns

X = trn.select([pl.exclude(['id', 'yield'])])

#checking datatypes to mke sure we can lm
X.dtypes

y = trn.select([pl.col('yield')])

lm = LinearRegression()

lm.fit(X, y)

lm.coef_

#get X_test
X_test = tst.select([pl.exclude('id')])

preds = lm.predict(X_test)

#write to file
sub = pl.DataFrame(
    
)