using CSV
using DataFrames
using Statistics
using GLM
using Chain
using BSplineKit

trn = CSV.read("playground_series/s4e9/data/train.csv", DataFrame)
tst = CSV.read("playground_series/s4e9/data/test.csv", DataFrame)

#i'll want to do a more thorough eda, but my bet is that just using year, mileage, and whether there was an accident will get us a decent model

#replace missing accidents in train with 'none reported' for now
replace!(trn.:accident, missing => "None reported")
replace!(tst.:accident, missing => "None reported")

trn = @chain trn begin
    transform(:accident => ByRow(x -> x == "None reported" ? 0 : 1) => :accident_bool)
end

X = Matrix(select(trn, [:model_year, :milage, :accident_bool]))
y = trn.:price
# might eventually want to try a spline, but let's just do a linear model for now

β = X \ y

# getting testing data to the correct format
transform!(
    tst,
    :accident => ByRow(x -> x == "None reported" ? 0 : 1) => :accident_bool
)

X_tst = Matrix(select(tst, [:model_year, :milage, :accident_bool]))

preds = X_tst * β

#write out submission
sub = DataFrame(
    id=tst.:id,
    price=preds
)

CSV.write("playground_series/s4e9/submissions/baseline_lm.csv", sub)