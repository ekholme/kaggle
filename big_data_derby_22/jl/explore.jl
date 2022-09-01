using CSV
using DataFrames

#see here for more on the data: https://www.kaggle.com/competitions/big-data-derby-2022/data

#read data in

df = CSV.read("./big_data_derby_22/data/nyra_2019_complete.csv", DataFrame)

#so we have 5.2m rows and 17 cols

show(names(df))