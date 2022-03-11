using CSV
using MLJ
using DataFrames
using MLJLinearModels


train = CSV.read("data/train.csv", DataFrame)
test = CSV.read("data/test.csv", DataFrame)

#make a pipeline and model
onehot = MLJ.OneHotEncoder()
stand = MLJ.Standardizer()
cont_encode = ContinuousEncoder()
lm = MLJLinearModels.LinearRegressor()

X = train[!, 3:5]
y = convert(Vector{Float64}, train[!, 6])

pipe = Pipeline(X-> coerce(X, :x=>Continuous,:y=>Continuous), cont_encode, onehot, stand, lm)

mach = machine(pipe, X, y)

fit!(mach)

preds = predict(mach, test)

#write out results
sub = DataFrame(row_id = test.row_id,
congestion = preds)

CSV.write("submissions/baseline_lm.csv", sub)