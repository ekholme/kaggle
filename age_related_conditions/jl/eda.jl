using CSV
using DataFrames
using CairoMakie
using Statistics

trn = CSV.read("./age_related_conditions/train.csv")

# General shape -------------------

size(trn)

# Missingness ---------------------

show(mapcols(x -> sum(ismissing, x), trn), allcols = true)