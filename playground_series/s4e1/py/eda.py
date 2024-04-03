# todo
import polars as pl
import statistics
import numpy as np

trn = pl.read_csv("./playground_series/s4e1/data/train.csv")
tst = pl.read_csv("./playground_series/s4e1/data/test.csv")
