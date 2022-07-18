using CSV
using DataFrames
using MultivariateStats
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
X_mat = Matrix(X)

# Perform PCA
PC_fit = fit(PCA, X_mat'; pratio = .9)

#apply PCA model back to data
X_pcs = predict(PC_fit, X_mat')

#apply kmeans to reduced feature space
C = Clustering.kmeans(X_pcs, 7)

#get submission
sub = DataFrame(
    Id = df[!, :id],
    Predicted = C.assignments
)

CSV.write("july_22_tabular/submissions/pca_jl.csv", sub)