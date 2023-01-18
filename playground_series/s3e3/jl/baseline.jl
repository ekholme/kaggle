using CSV
using DataFrames
using Statistics

trn = CSV.read("./playground_series/s3e3/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e3/data/test.csv", DataFrame)

pred = mean(trn.:Attrition)

sub = DataFrame(
    id = tst.:id,
    Attrition = pred
)

CSV.write("./playground_series/s3e3/submissions/baseline.csv", sub)