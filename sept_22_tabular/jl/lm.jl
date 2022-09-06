using CSV
using DataFrames
using Statistics
using GLM
using Dates

trn = CSV.read("./sept_22_tabular/data/train.csv", DataFrame)
tst = CSV.read("./sept_22_tabular/data/test.csv", DataFrame)

target = trn.:num_sold

#create converter function to get days since first day as an integer
function convert_date(x)
    m = trn.:date[begin]
    tmp = x .- m
    ret = Dates.value.(tmp)

    return ret
end

X = convert_date(trn.:date)

#run a simple little model where num_sold ~ date
data = DataFrame(X = X, y = target)

ols_res = lm(@formula(y ~ X), data)

#convert test data and get predictions
X_tst = convert_date(tst.:date)

preds = Int.(round.(predict(ols_res, DataFrame(X = X_tst))))

sub = DataFrame(
    row_id = tst.:row_id,
    num_sold = preds
)

#write out predictions
CSV.write("./sept_22_tabular/submissions/baseline_lm_date_only.csv", sub)