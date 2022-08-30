using CSV
using DataFrames
using Statistics
using MLJ
using Lasso
using CategoricalArrays

trn = CSV.read("./aug_22_tabular/data/train.csv", DataFrame)
tst = CSV.read("./aug_22_tabular/data/test.csv", DataFrame)

#get target column out
target = trn[:, :failure]
target = CategoricalArray(target)

#remove id and target columns
tst_id = tst.:id
tst = tst[:, Not(:id)]

trn_id = trn[:, :id]
trn = trn[:, Not([:id, :failure])]

#concatenate dfs together
X = [trn; tst]

#define transform steps
imputer = FillImputer()
oh_encoder = OneHotEncoder(drop_last = true)
std_zer = Standardizer()

#create a transformation pipeline
trans_pipe = (X -> coerce(X, Textual => Multiclass)) |> imputer |> std_zer |> oh_encoder

trans_mach = machine(trans_pipe, X)

fit!(trans_mach)

X_trans = MLJ.transform(trans_mach, X)


trn_trans = X_trans[1:length(target),:]
tst_trans = X_trans[length(target)+1:end, :]

#get columns with no variance
nms = names(trn_trans)

unique_vals = []

for i in nms
    r = length(unique(trn_trans[:, i]))

    push!(unique_vals, r)
    print(i * "has " * string(r) * " unique values \n")
end

exclude_vars = nms[unique_vals .== 1]

select!(trn_trans, Not(exclude_vars))
select!(tst_trans, Not(exclude_vars))

#fit model
m = fit(LassoModel, Matrix(trn_trans), target, Binomial())

preds = Lasso.predict(m, Matrix(tst_trans))

sub = DataFrame(
    id = tst_id,
    failure = preds
)

CSV.write("./aug_22_tabular/submission/lasso_jl.csv")