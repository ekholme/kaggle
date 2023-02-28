using CSV
using DataFrames
using GLM
using Statistics
using XGBoost
using MLJ

trn = CSV.read("./playground_series/s3e6/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e6/data/test.csv", DataFrame)

#coerce data
X = trn[:, 2:end-1]
y = Float64.(trn.:price)

#not sure why xgboost isn't showing up here
models(matching(X, y))

#fit model
b = xgboost((X, y), num_round = 100, max_depth = 5, η = .1)

#make predictions
preds = XGBoost.predict(b, tst[:, 2:end])

#write out submission
sub = DataFrame(
    id = tst.:id,
    price = preds
)

CSV.write("./playground_series/s3e6/submissions/xgb_base.csv", sub)