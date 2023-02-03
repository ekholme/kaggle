using CSV
using DataFrames
using Statistics
using StatsBase
using MLJ
# see https://medium.com/@liza_p_semenova/ordered-logistic-regression-and-probabilistic-programming-502d8235ad3f
#for using Turing to estimate this
trn = CSV.read("./playground_series/s3e5/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e5/data/test.csv", DataFrame)


#getting dv counts
cs_zip = collect(zip(levels(trn.:quality), StatsBase.counts(trn.:quality)))

show(schema(trn))
#get X and Y values
X_trn = Matrix(trn[:, 2:12])
y_trn = coerce(trn.:quality, Multiclass)

models(matching(X_trn, y_trn)) |> DataFrame

nb = @load GaussianNBClassifier pkg=NaiveBayes
nbc = nb()

mach_nbc = machine(nbc, X_trn, y_trn)
fit!(mach_nbc)

#get test data
X_tst = Matrix(tst[:, 2:12])
#and get predictions
p = MLJ.predict_mode(mach_nbc, X_tst)

sub = DataFrame(
    Id = tst.:Id,
    quality = p
)

#checking counts
StatsBase.counts(int(p))
#sure why not

CSV.write("./playground_series/s3e5/submissions/naive_bayes_sub.csv", sub)

#next, write a model to do ordinal logistic reg using Turing
# see https://medium.com/@liza_p_semenova/ordered-logistic-regression-and-probabilistic-programming-502d8235ad3f
#see 