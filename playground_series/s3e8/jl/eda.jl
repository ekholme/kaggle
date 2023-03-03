using CSV
using DataFrames
using Statistics
using Chain
using CairoMakie
using Parquet2
using NamedArrays

#first convert to parquet
# trn = CSV.read("./playground_series/s3e8/data/train.csv", DataFrame)
# tst = CSV.read("./playground_series/s3e8/data/test.csv", DataFrame)

# Parquet2.writefile("./playground_series/s3e8/data/train.parquet", trn)
# Parquet2.writefile("./playground_series/s3e8/data/test.parquet", tst)

#load in data
trn = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/train.parquet"))
tst = DataFrame(Parquet2.Dataset("./playground_series/s3e8/data/test.parquet"))

# Check Missingness -------------

mapcols(x -> sum(ismissing, x), trn)
#no missing data

# Explore Target ---------------

target = trn.:price 

mean(target)

hist(target)
#yikes -- ok. super right skewed

#let's try logging it
hist(log.(target))
#logging makes it look somewhat better, but that might not
#be the best approach for modeling?

# Explore Categorical Features ------------
ts = eltype.(eachcol(trn))

cat_feats = names(trn)[ts .== String]

#pivot wider to make a combined histogram
X_cat = select(trn, vcat("id", cat_feats, "price"))

X_cat_long = @chain X_cat begin
    stack(2:4)
    groupby([:variable, :value])
    combine(:price => mean)
end

#making a bar chart with avg price for each
f2 = Figure()

ax2 = [Axis(f2[i, 1]) for i in 1:length(cat_feats)]


for i in 1:length(cat_feats)
    tmp = X_cat_long[X_cat_long.:variable .== cat_feats[i], :]
    l = 1:length(tmp[:, :value])

    ax2[i].xticks = (l, tmp.:value)
    ax2[i].title = cat_feats[i]
    
    barplot!(
        ax2[i],
        l,
        tmp[:, :price_mean]
        # bar_labels = tmp[:, :value] 
    )
end

f2
#ok, so, this tells me that there are differences, but
#this is obv going to be confounded with other things like carat, etc

# Continuous Variables ------------

#get continuous features (plus target) and not id
cont_feats = setdiff(names(trn), cat_feats)[2:end]

X_cont = trn[:, cont_feats]

cm = cor(Matrix(X_cont))

cm = NamedArray(cm, (names(X_cont), names(X_cont)))

p_cors = cm[:, 7]
show(p_cors)
#so basically price is super highly correlated with size
#which makes sense...
#but there will obv be an interaction btwn size and stuff like cut, color, and clarity