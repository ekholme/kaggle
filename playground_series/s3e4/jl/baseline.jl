using CSV
using DataFrames
using Statistics
using Parquet2

#read parquet
#note that I already converted these to parquet
trn = DataFrame(Parquet2.Dataset("./playground_series/s3e4/data/train.parquet"); copycols = false)
tst = DataFrame(Parquet2.Dataset("./playground_series/s3e4/data/test.parquet"); copycols = false)

names(trn)

pred = mean(trn.:Class)

sub = DataFrame(
    id = tst.:id,
    Class = pred
)

CSV.write("./playground_series/s3e4/submissions/baseline.csv", sub)