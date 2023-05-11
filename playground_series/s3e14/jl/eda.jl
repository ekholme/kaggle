using DataFrames
using CSV
using Statistics
using Plots

trn = CSV.read("./playground_series/s3e14/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e14/data/test.csv", DataFrame)

#Examine shape

size(trn)

# get element types
show(eltype.(eachcol(trn)))

#check missingness
show(mapcols(x -> sum(ismissing, x), trn), allcols = true)

# Explore Target --------------

mean(trn.:yield)

std(trn.:yield)

histogram(trn.:yield)
#slight left skew

# Explore Predictors -----------

#getting predictors
X = select(trn, Not(["id", "yield"]))

