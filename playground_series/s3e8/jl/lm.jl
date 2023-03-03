using Parquet2
using GLM
using Statistics
using CSV

trn = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/train.parquet"))
tst = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/test.parquet"))

#let's see how far we can get with just carat

target = log.(trn[:, :price])

x = trn[:, :carat]

X = hcat(ones(length(x)), x)

mod = X \ target

#get predictions
preds = hcat(ones(length(tst[:, :carat])), tst[:, :carat]) * mod

preds = exp.(preds)

#write submission
sub = DataFrame(
    id = tst.:id,
    price = preds
)

CSV.write("./playground_series/s3e8/submissions/just_carat_lm.csv", sub)
#hmm so this actually isn't very good
#i assume it really misclassifies some smaller ideal cuts, etc

# Take 2 ------------

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
X = hcat(ones(nrow(trn)), hcat(xs...))
X = hcat(X, trn.:carat, makeunique = true)

mod = Matrix(X) \ target

#onehot tst

x_tst = Vector{Any}(undef, 3)

for i in 1:lastindex(cat_feats)

    x = tst[:, cat_feats[i]]

    tmp = select(
    tst,
    [Symbol(cat_feats[i]) => ByRow(isequal(v)) => Symbol(v) for v in unique(x)]
)

x_tst[i] = tmp
end

X_tst = hcat(ones(nrow(tst)), hcat(x_tst...))
X_tst = Matrix(hcat(X_tst, tst.:carat, makeunique = true))

preds = exp.(X_tst * mod)

sub = DataFrame(
    id = tst.:id,
    price = preds
)

CSV.write("./playground_series/s3e8/submissions/lm_onehot.csv", sub)