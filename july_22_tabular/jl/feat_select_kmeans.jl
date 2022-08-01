using CSV
using DataFrames
using Clustering

df = CSV.read("july_22_tabular/data/data.csv", DataFrame)

tmp_cols = ["f_07", "f_08", "f_09"]
other_cols = [collect(10:13); collect(22:28)]

aa = []

for col in other_cols
    i = "f_" * string(col)
    push!(aa, i)
end

keep_cols = [tmp_cols; aa]

# Subset Dataframe

X = select(df, keep_cols)

# Kmeans

C = Clustering.kmeans(Matrix(X)', 7)

#write submission
sub = DataFrame(
    Id=df[!, :id],
    Predicted=C.assignments
)

CSV.write("july_22_tabular/submissions/feat_jl.csv", sub)