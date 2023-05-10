import polars as pl
import statistics

trn = pl.read_csv("./playground_series/s3e14/data/train.csv")
tst = pl.read_csv("./playground_series/s3e14/data/test.csv")

# Examine Data

trn.shape

trn.head()