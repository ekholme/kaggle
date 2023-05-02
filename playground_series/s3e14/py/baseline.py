import polars as pl

trn = pl.read_csv("./playground_series/s3e14/data/train.csv")
tst = pl.read_csv("./playground_series/s3e14/data/test.csv")
