using CSV
using Flux
using Flux: train!
using DataFrames
using StatsBase

trn = CSV.read("./playground_series/s3e11/data/train.csv", DataFrame)
tst = CSV.read("./playground_series/s3e11/data/test.csv", DataFrame)

𝐗 = trn[:, 2:end-1]
y = trn[:, end]

#get z score, convert to matrix, and convert to float 32 for flux fitting
Xz = Float32.(Matrix(mapcols(zscore, 𝐗)))

#fit a flux model ----------------

#define the model
mod = Dense(size(Xz, 2) => 1)

#define a loss function
my_loss(x, y) = Flux.Losses.mse(mod(x)', y)

#use Adam as optimizer with .01 lr
opt = Adam(.01)

#define data
#I think I need the transpose of Xz
data = [(Xz', y)]

#isolate parameters
θ = Flux.params(mod)

#get the initial loss
my_loss(Xz', y)

#do one training round
train!(my_loss, θ, data, opt)

#and look at the loss again
my_loss(Xz', y)
#ok so this is in the right direction

#now let's do a bunch of training iterations
n_epoch = 4_000

for epoch in 1:n_epoch
    train!(my_loss, θ, data, opt)
end

θ
my_loss(Xz', y)
#ok so this works, but I doubt it's a great model
#but let's get predictions anyway

𝐗_tst = Float32.(Matrix(mapcols(zscore, tst[:, 2:end])))
#this actually isn't ideal bc i'm computing z-scores on different data sets
#but we'll just assume they have the same distributions and yolo for now

preds = vec(mod(𝐗_tst')')
#so these predictions are going to be terrible bc of the data types
#there are several binary columns, so z-scoring those is bad news
#but w/e

sub = DataFrame(
    id = tst.:id,
    cost = preds
)

CSV.write("./playground_series/s3e11/submissions/flux_fiddling.csv", sub)
