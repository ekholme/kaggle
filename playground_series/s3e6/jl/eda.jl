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

mean(target)

#and let's get a histogram of the target
hist(target)
#ok, so a fairly uniform distribution

# binary features --------------

binary_features = [:hasYard, :hasPool, :isNewBuilt, :hasStormProtector, :hasStorageRoom]

trn_bin = copy(trn[:, binary_features])

mapcols(x ->sum(x) / nrow(trn), trn_bin)

#ok, so about 45-47% of all houses have any given one of these features.

#faceted histograms
#see https://docs.makie.org/stable/tutorials/layout-tutorial/ for a start
f = Figure()
ax = Axis(f[1,1])

demo_cols = [:squareMeters, :numberOfRooms, :basement]

for i in demo_cols
    hist!(ax, trn[:, i])
end

f
#this doesn't work yet