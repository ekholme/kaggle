using CSV
using DataFrames
using Statistics

trn = CSV.read("./sept_22_tabular/data/train.csv", DataFrame)
tst = CSV.read("./sept_22_tabular/data/test.csv", DataFrame)
sample = CSV.read("./sept_22_tabular/data/sample_submission.csv", DataFrame)

pred = Int(round(mean(trn.:num_sold)))

sub = DataFrame(
    row_id = tst.:row_id,
    num_sold = pred
)

#write out
CSV.write("./sept_22_tabular/submissions/baseline.csv", sub)