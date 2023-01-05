using CSV
using DataFrames
using Statistics

trn = CSV.read("./playground_series/s3e1/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e1/data/test.csv", DataFrame)

#take a peek at dimensions
size(trn)

pred = mean(trn.:MedHouseVal)

sub = DataFrame(
    id = tst.:id,
    MedHouseVal = pred
)

#write out
CSV.write("./playground_series/s3e1/submissions/baseline_mean.csv", sub)