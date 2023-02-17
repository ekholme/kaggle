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

#non-binary features -------------

cont_feats = setdiff(Symbol.(names(trn)), binary_features)[2:end-1]

f2 = Figure()

ax2 = [Axis(f2[i, j]) for i in 1:4, j in 1:3][1:length(cont_feats)]

for (i, d) in enumerate(cont_feats)
    ax2[i].title = string(d)
    hist!(ax2[i], trn[:, d])
end

f2
#ok so this works finally
#i don't know exactly what city code is, but it feels like a grouping variable

#next step is to correlate features with price

# correlations ----------------------
cm = cor(Matrix(trn))

#just cors with price
price_cors = collect(zip(names(trn), cm[:, end]))
show(price_cors)
#ok so the only one that really seems to matter much is square footage. garage has a small negative value
#it's possible there are some nonlinearities, though

#let's treat city code as categorical
length(unique(trn.:cityCode))
#ok just kidding -- there are way too many unique values

#what about city part range
length(unique(trn.:cityPartRange))

part_range_mns = @chain trn begin
    transform(:cityPartRange => string )
end
#this doesn't work like i want it to