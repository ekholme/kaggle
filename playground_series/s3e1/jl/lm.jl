using CSV
using DataFrames
using GLM

trn = CSV.read("./playground_series/s3e1/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e1/data/test.csv", DataFrame)

#a simple approach is just to get all of the data and fit a simple linear model

X = Matrix(trn[:, 2:9])
X_tst = Matrix(tst[:, 2:9])

y = trn.:MedHouseVal

model = lm(X, y)

preds = predict(model, X_tst)

sub = DataFrame(
    id = tst.:id,
    MedHouseVal = preds
)

CSV.write("./playground_series/s3e1/submissions/base_lm.csv", sub)