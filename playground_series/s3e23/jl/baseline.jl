using CSV
using DataFrames
using GLM
using Statistics

trn = CSV.read("./playground_series/s3e23/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e23/data/test.csv", DataFrame)

# get avg bug rate
μ = mean(trn.:defects)

preds = DataFrame(
    id=tst.:id,
    defects=μ
)

#write out to file
CSV.write("./playground_series/s3e23/submissions/baseline.csv", preds)