import polars as pl
import statistics

trn = pl.read_csv("./playground_series/s3e14/data/train.csv")
tst = pl.read_csv("./playground_series/s3e14/data/test.csv")

trn.shape

#getting mean yield
pred = statistics.mean(trn['yield'])

#creating submission
sub = pl.DataFrame(
    {
        "id": tst['id'],
        "yield": pred,
    }
)

#write out
sub.write_csv("./playground_series/s3e14/submissions/baseline_py.csv")