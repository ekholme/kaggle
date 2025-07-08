import polars as pl
import seaborn as sns
import statistics

trn = pl.read_csv("playground_series/s5e5/data/train.csv")
tst = pl.read_csv("playground_series/s5e5/data/test.csv")

y = trn["Calories"]
pred = y.mean()

sub = pl.DataFrame({"id": tst["id"], "Calories": pred})

# write out
sub.write_csv("playground_series/s5e5/submissions/baseline.csv")
