using CSV
using DataFrames
using MLJ
using XGBoost

trn = CSV.read("./playground_series/s3e23/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e23/data/test.csv", DataFrame)

# get X and y
y, X = unpack(trn, ==(:defects))
X = X[:, 2:end]

y_cat = coerce(y, Multiclass)

mm = models(matching(X, y_cat)) |> DataFrame

rf = @load RandomForestClassifier pkg = DecisionTree

RF = rf()

rf_m = machine(RF, X, y_cat)

fit!(rf_m)

p = MLJ.predict(rf_m, tst[:, 2:end])

pt = pdf.(p, true)

sub = DataFrame(
    id=tst.:id,
    defects=pt
)

CSV.write("./playground_series/s3e23/submissions/rf_baseline.csv", sub)
