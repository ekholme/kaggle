using CSV
using DataFrames
using Statistics
using MLJ

trn = CSV.read("./playground_series/s3e5/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e5/data/test.csv", DataFrame)

y, X = unpack(trn, ==(:quality))
X = X[:,2:12]
y = coerce(string.(y), Multiclass)

scitype(y)

models(matching(X, y)) |> DataFrame

xgb = @load XGBoostClassifier pkg=XGBoost

#we'll just use the defaults for now
XGB = xgb()

xgbm = machine(XGB, X, y)

fit!(xgbm)

#get predictions
p = predict_mode(xgbm, tst[:,2:12])

sub = DataFrame(
    Id = tst.:Id,
    quality = p
)

CSV.write("./playground_series/s3e5/submissions/xgb_baseline_sub.csv", sub)
#interesting that it's worse than naive bayes, but it's completely untuned so it sort of makes sense