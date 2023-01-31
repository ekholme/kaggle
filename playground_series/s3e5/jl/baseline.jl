using CSV
using DataFrames
using Statistics
using StatsBase

trn = CSV.read("./playground_series/s3e5/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e5/data/test.csv", DataFrame)

n = StatsBase.counts(trn.:quality)

pred = getindex(levels(trn.:quality), findmax(n)[2])

sub = DataFrame(
    Id = tst.:Id,
    quality = pred
)

CSV.write("./playground_series/s3e5/submissions/baseline_jl.csv", sub)