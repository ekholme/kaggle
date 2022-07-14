using Revise
using CSV
using DataFrames
using Clustering

#read in data
df = CSV.read("july_22_tabular/data/data.csv", DataFrame)

#get rid of id
X = df[:, 2:ncol(df)]

#kmeans apparently expects a transposed matrix, like:
# Matrix(X)'

#estimate kmeans using 7 clusters
C = Clustering.kmeans(Matrix(X)', 7)

#write submission
sub = DataFrame(
    Id = df[!, :id],
    Predicted = C.assignments
)

CSV.write("july_22_tabular/submissions/baseline_jl.csv", sub)