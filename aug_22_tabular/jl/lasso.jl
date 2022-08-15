using CSV
using DataFrames
using Statistics
using MLJ
using Lasso

trn = CSV.read("./aug_22_tabular/data/train.csv", DataFrame)
tst = CSV.read("./aug_22_tabular/data/test.csv", DataFrame)

#transform data

#remove id column
trn = trn[:, Not(:id)]

#unpack
y, X = unpack(trn, ==(:failure))

y = string.(y)

imputer = FillImputer()
oh_encoder = OneHotEncoder(drop_last = true)
std_zer = Standardizer()

#let's just try the above

#fitting our imputation machine
mach_impute = fit!(machine(imputer, X))

trn_trans = MLJ.transform(mach_impute, X)

#now doing the standardizing with no missing data
mach_std = fit!(machine(std_zer, trn_trans))

X_trans = MLJ.transform(mach_std, trn_trans)

##TODO
#need to get the string column names to use in the OH encoder above