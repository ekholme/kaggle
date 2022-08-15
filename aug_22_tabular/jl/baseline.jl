using CSV
using DataFrames
using Statistics

trn = CSV.read("./aug_22_tabular/data/train.csv", DataFrame)
tst_id = CSV.read("./aug_22_tabular/data/test.csv", DataFrame).id

pred = mean(trn.failure)

sub = DataFrame(
    id = tst_id,
    failure = pred
)