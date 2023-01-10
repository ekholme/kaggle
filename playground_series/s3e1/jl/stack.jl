using CSV
using DataFrames
using MLJ
using Lasso
using Statistics

trn = CSV.read("./playground_series/s3e1/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e1/data/test.csv", DataFrame)

#unpack data
y_trn, X_trn = unpack(trn[:,2:ncol(trn)], ==(:MedHouseVal), rng = 0408)
y_tst, X_tst = unpack(tst[:, 2:ncol(tst)], ==(:MedHouseVal), rng = 0408)
#transform data

std_zer = Standardizer()

transform_machine = machine(std_zer, X_trn)

fit!(transform_machine)

X_trans = MLJ.transform(transform_machine, X_trn)

#fit lasso
m = fit(LassoModel, Matrix(X_trn), y_trn, Normal())

#get lasso predictions
lasso_preds = Lasso.predict(m, Matrix(X_tst))

#fit rf model
rf = @load RandomForestRegressor pkg=DecisionTree

randf = rf()

rf_machine = machine(randf, X_trn, y_trn)

fit!(rf_machine)

rf_preds = MLJ.predict(
    rf_machine,
    X_tst
)

#take avg of preds
function pred_means(y1, y2)
    (y1 + y2) / 2
end

p = pred_means.(lasso_preds, rf_preds)

sub = DataFrame(
    id = tst.:id,
    MedHouseVal = p
)

CSV.write("./playground_series/s3e1/submissions/rf_lasso_stack.csv", sub)
