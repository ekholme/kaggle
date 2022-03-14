#Setup & Load Data

using CSV
using MLJ
using DataFrames

train = CSV.read("data/train.csv", DataFrame)
test = CSV.read("data/test.csv", DataFrame)

#get mean congestion
trn_avg = mean(train.congestion)

#make new df
sub = DataFrame(row_id = test.row_id,
congestion = trn_avg)

#write out baseline sub
CSV.write("submissions/baseline.csv", sub)