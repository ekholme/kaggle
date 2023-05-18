#splitting training data into an 80-20 train/test to give something to evaluate on
using CSV
using DataFrames
using Random

trn = CSV.read("./age_related_conditions/data/train.csv", DataFrame)

#setting a random seed so this is replicable
Random.seed!(0408)

#splitting function
function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    r = ids .<= (nrow(df) * pct)
    # return r
    return df[r, :], df[.!r, :]
end

train, val = splitdf(trn, .8)

#write out new files
CSV.write("./age_related_conditions/data/trn.csv", train)
CSV.write("./age_related_conditions/data/validation.csv", val)