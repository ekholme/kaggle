import polars as pl
import numpy as np
import statistics

trn = pl.read_csv("playground_series/s5e1/data/train.csv")
tst = pl.read_csv("playground_series/s5e1/data/test.csv")

#let's just take the rounded mean for now
#although I should probably come up with a better way to handle the nulls
y = trn['num_sold'].drop_nulls()

avg = round(statistics.mean(y))

preds = pl.DataFrame(
    {
        "id": tst["id"],
        "num_sold": avg
    }
)

preds.write_csv("playground_series/s5e1/submissions/baseline.csv")