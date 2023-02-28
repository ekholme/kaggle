using CSV
using DataFrames
using Statistics
using Chain
using CairoMakie
using Parquet2

#first convert to parquet
# trn = CSV.read("./playground_series/s3e8/data/train.csv", DataFrame)
# tst = CSV.read("./playground_series/s3e8/data/test.csv", DataFrame)

# Parquet2.writefile("./playground_series/s3e8/data/train.parquet", trn)
# Parquet2.writefile("./playground_series/s3e8/data/test.parquet", tst)

#load in data
trn = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/train.parquet"))
tst = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/test.parquet"))

# Check Missingness -------------

mapcols(x -> sum(ismissing, x), trn)

# Explore Target ---------------

target = trn.:price 

mean(target)

hist(target)
#yikes -- ok. super right skewed

#let's try logging it
hist(log.(target))
#logging makes it look somewhat better, but that might not
#be the best approach for modeling?