using CSV
using GLM
using DataFrames

trn = CSV.read("./playground_series/s3e11/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e11/data/test.csv", DataFrame)

#get types of cols
show(stack(mapcols(x -> eltype(x), trn), 1:ncol(trn)), allrows = true)
#note -- make this a function in eemisc
