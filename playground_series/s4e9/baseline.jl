using CSV
using DataFrames
using Statistics

trn = CSV.read("playground_series/s4e9/data/train.csv", DataFrame)
tst = CSV.read("playground_series/s4e9/data/test.csv", DataFrame)

avg_price = mean(trn.:price)

sub = DataFrame(
    id=tst.:id,
    price=avg_price
)

CSV.write("playground_series/s4e9/submissions/baseline.csv", sub)