using CSV
using DataFrames
using GLM

trn = CSV.read("./playground_series/s3e23/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e23/data/test.csv", DataFrame)

# get X and y
X = Matrix(hcat(ones(nrow(trn)), trn[:, 2:end-1]))
y = trn.:defects

#fit model
mod = glm(X, y, Binomial(), LogitLink())

#predict
preds = predict(mod, Matrix(hcat(ones(nrow(tst)), tst[:, 2:end])))

sub = DataFrame(
    id=tst.:id,
    defects=preds
)

CSV.write("./playground_series/s3e23/submissions/logreg_baseline.csv", sub)