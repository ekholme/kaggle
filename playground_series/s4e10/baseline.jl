using CSV
using DataFrames
using GLM
using MLJ
using Chain

trn = CSV.read("playground_series/s4e10/data/train.csv", DataFrame)
tst = CSV.read("playground_series/s4e10/data/test.csv", DataFrame)

#getting column types
show(eltype.(eachcol(trn)))

show(MLJ.schema(trn))

#need to coerce all textual data to multiclass
trn_c = coerce(trn, Textual => Multiclass, :loan_status => Multiclass)

#split into train/validate sets
train, val = partition(trn_c, 0.8, shuffle=true, rng=0408)

#unpack X and y in train
y, X, id = unpack(train, ==(:loan_status), !=(:id))

#i won't standardize for this baseline approach, but i do need to one hot encode
oh = OneHotEncoder(drop_last=true)

xgb = @load XGBoostClassifier
xgb_model = xgb()

p = Pipeline(oh, xgb_model)

mach = machine(p, X, y)

fit!(mach)

#getting X test set
tst_c = coerce(tst, Textual => Multiclass)
X_tst, id = unpack(tst_c, !=(:id))

ŷ = MLJ.predict(mach, X_tst)

preds = pdf.(ŷ, 1)

#write out submission
sub = DataFrame(
    id=id,
    loan_status=preds
)

CSV.write("playground_series/s4e10/submissions/baseline_xgb.csv", sub)