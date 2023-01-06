using CSV
using DataFrames
using MLJ

trn = CSV.read("./playground_series/s3e1/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e1/data/test.csv", DataFrame)

#unpack data
y_trn, X_trn = unpack(trn[:,2:ncol(trn)], ==(:MedHouseVal), rng = 0408)

#check models matching data structure
show(models(matching(X_trn, y_trn)))

rf = @load RandomForestRegressor pkg=DecisionTree

rand_forest = rf()

#create a machine
mach = machine(rand_forest, X_trn, y_trn)

fit!(mach)

#get predictions
ŷ = predict(mach, tst[:, 2:ncol(tst)])

sub = DataFrame(
    id = tst.:id,
    MedHouseVal = ŷ
)

CSV.write("./playground_series/s3e1/submissions/base_rf.csv", sub)
