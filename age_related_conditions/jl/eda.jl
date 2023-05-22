using CSV
using DataFrames
using CairoMakie
using Statistics
using Chain

trn = CSV.read("./age_related_conditions/data/train.csv", DataFrame)
greeks = CSV.read("./age_related_conditions/data/greeks.csv", DataFrame) 

# General shape -------------------

size(trn)

# Missingness ---------------------

show(mapcols(x -> sum(ismissing, x), trn), allcols = true)

#BQ and EL have moderate missingness (~10%); a few other cols have like 1 or 2 missing records


# Element Types -------------------

tps = eltype.(eachcol(trn))
tp_counts = [(i, count(==(i), tps)) for i in unique(tps)]

#find the col that is a string
ind = eltype.(eachcol(trn)) .== String1
names(trn)[ind]
#i'm not sure if this is the best way to do this, but w/e
#EJ is our categorical column

unique(trn.:EJ)
#we'll revisit this later

# Explore Target ----------------------

sum(trn.:Class)/ nrow(trn)

#17.5 % of cases are 'yes'

#let's dig a little bit more in the greeks file
#if Alpha == 'A', then Class == 0; otherwise B, D, and G represent 3 different outcomes

out_classes = [(i, count(==(i), greeks.:Alpha)) for i in unique(greeks.:Alpha)]
#ok, so B is much more common then D and G (more common than both combined)
#possible option moving forward is to model each separately

# Explore Numerical Columns -----------------

X = select(trn, Not([:Id, :EJ, :Class]))


f = Figure();

ax = [Axis(f[i, j]) for i in 1:7, j in 1:8]

for i in 1:ncol(X)
    tmp = X[:, i]

    comp = collect(skipmissing(tmp))

    ax[i].title = names(X)[i]

    hist!(
        ax[i],
        comp
        )
end

f
#ok i can't actually really see this but it works
#from what I can see, there's lots of non-normal, very skewed data here
#something like yeo-johnson or box-cox transformation could be helpful

# Explore Counts of Categorical Column -------------

cat_counts = [(i, count(==(i), trn.:EJ)) for i in unique(trn.:EJ)]

cat_outs = @chain trn begin
    select(:EJ, :Class)
    groupby(:EJ)
    combine(
        :Class => sum => :sum_class,
        nrow => :count
        )
    transform([:sum_class, :count] => ByRow(/) => :pct)
end
#B's are over-represented as positive cases relative to A's