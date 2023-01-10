using CSV
using DataFrames
using Statistics

trn = CSV.read("./playground_series/s3e2/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e2/data/test.csv", DataFrame)

#take a peek at dimensions
size(trn)

pred = mean(trn.:stroke)

sub = DataFrame(
    id = tst.:id,
    stroke = pred
)

#write out
CSV.write("./playground_series/s3e2/submissions/baseline_mean.csv", sub)
