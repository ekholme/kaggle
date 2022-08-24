using CSV
using DataFrames
using Statistics
using MLJ
using Lasso
using CategoricalArrays

trn = CSV.read("./aug_22_tabular/data/train.csv", DataFrame)
tst = CSV.read("./aug_22_tabular/data/test.csv", DataFrame)

#transform data

#remove id column
trn = trn[:, Not(:id)]
tst_id = tst.:id
tst = tst[:, Not(:id)]

#unpack
y, X = unpack(trn, ==(:failure))

y = CategoricalArray(y)

imputer = FillImputer()
oh_encoder = OneHotEncoder(drop_last = true)
std_zer = Standardizer()

#create a transformation pipeline
trans_pipe = (trn -> coerce(trn, Textual => Multiclass)) |> imputer |> std_zer |> oh_encoder

trans_mach = machine(trans_pipe, X)

fit!(trans_mach)

X_trans = MLJ.transform(trans_mach, X)

#so this finally works
log_fit = fit(LassoModel, Matrix(X_trans), y, Binomial())

#next step is to get the test data in the right shape and then predict
X_tst = MLJ.transform(trans_mach, tst)
#ok so running into an issue here -- I need to figure out a way to "step_other" the test data
size(X)
size(tst)

