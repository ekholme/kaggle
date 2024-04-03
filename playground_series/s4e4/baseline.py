import statistics
import polars as pl

trn = pl.read_csv("data/train.csv")
tst = pl.read_csv("data/test.csv")

y = trn["Rings"]

pred = round(statistics.mean(y))

sub = pl.DataFrame({
    "id": tst["id"],
    "Rings": pred
})

sub.write_csv("submissions/baseline.csv")