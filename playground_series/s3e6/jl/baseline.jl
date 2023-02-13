using CSV
using DataFrames
using Statistics

trn = CSV.read("./playground_series/s3e6/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e6/data/test.csv", DataFrame)

pred = mean(trn.:price)

sub = DataFrame(
    id = tst.:id,
    price = pred
)

CSV.write("./playground_series/s3e6/submissions/baseline.csv", sub)