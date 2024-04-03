import polars as pl
import statistics
import numpy as np

trn = pl.read_csv("./playground_series/s4e1/data/train.csv")
tst = pl.read_csv("./playground_series/s4e1/data/test.csv")

trn.head()

trn.shape

# get target
y = trn["Exited"]

# write code to take the mean of y
p = statistics.mean(y)

# encode predictions in a new dataframe
sub = pl.DataFrame({"id": tst["id"], "Exited": p})

# write to csv
sub.write_csv("./playground_series/s4e1/submissions/baseline.csv")
