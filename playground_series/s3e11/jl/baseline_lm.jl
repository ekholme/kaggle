using CSV
using GLM
using DataFrames

trn = CSV.read("./playground_series/s3e11/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e11/data/test.csv", DataFrame)

#get types of cols
show(stack(mapcols(x -> eltype(x), trn), 1:ncol(trn)), allrows = true)
#note -- make this a function in eemisc

#training a baseline lm model
𝐗 = Matrix(hcat(ones(nrow(trn)), trn[:, 2:end-1]))
y = trn[:, end]

mod = lm(𝐗, y)

#coercing test data
X_tst = Matrix(hcat(ones(nrow(tst)), tst[:, 2:end]))

preds = predict(mod, X_tst)

sub = DataFrame(
    id = tst.:id,
    cost = preds
)

CSV.write("./playground_series/s3e11/submissions/lm_baseline.csv", sub)