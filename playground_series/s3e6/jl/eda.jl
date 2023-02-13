using CSV
using DataFrames
using Statistics
using CairoMakie
using Chain
trn = CSV.read("./playground_series/s3e6/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e6/data/test.csv", DataFrame)

# overall dims --------------

size(trn)

# missingness ----------------

mapcols(x -> sum(ismissing, x), trn)

# target column --------------
target = trn.:price

##RESUME HERE