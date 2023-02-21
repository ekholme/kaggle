using CSV
using DataFrames
using GLM
using Statistics

trn = CSV.read("./playground_series/s3e6/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e6/data/test.csv", DataFrame)

#get a basic training set
X = Matrix(trn[:, 2:end-1])
y = trn.:price

#estimate using \, why not
mod = X \ y

#make predictions
X_test = Matrix(tst[:, 2:end])

preds = X_test * mod

#write out submission

