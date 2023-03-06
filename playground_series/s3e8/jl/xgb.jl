using Parquet2
using DataFrames
using GLM
using Statistics
using CSV
using XGBoost

trn = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/train.parquet"))
tst = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/test.parquet"))

# coerce data
target = trn.:price

#dummy out all of the binary cols
ts = eltype.(eachcol(trn))

cat_feats = names(trn)[ts .== String]

xs = Vector{Any}(undef, 3)

for i in 1:lastindex(cat_feats)

    x = trn[:, cat_feats[i]]

    tmp = select(
    trn,
    [Symbol(cat_feats[i]) => ByRow(isequal(v)) => Symbol(v) for v in unique(x)]
)

xs[i] = tmp
end

crt = trn[:, :carat]
X = hcat(xs...)

X_cont = trn[:, [:carat, :depth, :table, :x, :y, :z]] 
X = hcat(X, X_cont, makeunique = true)

#fit the model
mod = xgboost((X, target), num_round = 100, max_depth = 5, η = .1)

# onehot test df
x_tst = Vector{Any}(undef, 3)

for i in 1:lastindex(cat_feats)

    x = tst[:, cat_feats[i]]

    tmp = select(
    tst,
    [Symbol(cat_feats[i]) => ByRow(isequal(v)) => Symbol(v) for v in unique(x)]
)

x_tst[i] = tmp
end

X_tst_cat = hcat(x_tst...)
X_tst_cont = tst[:, [:carat, :depth, :table, :x, :y, :z]] 
X_tst_mm = hcat(X_tst_cat, X_tst_cont) 

preds = XGBoost.predict(mod, X_tst_mm)

sub = DataFrame(
    id = tst.:id,
    price = preds
)

CSV.write("./playground_series/s3e8/submissions/xgb_base.csv", sub)
#interesting that this is worse than the model that just uses carat
