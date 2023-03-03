using CSV
using Statistics
using DataFrames

trn = CSV.read("./playground_series/s3e8/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e8/data/test.csv", DataFrame)

#so this is the diamonds dataset

pred = mean(trn.:price)

sub = DataFrame(
    id = tst.:id,
    price = pred
)

CSV.write("./playground_series/s3e8/submissions/baseline.csv", sub)