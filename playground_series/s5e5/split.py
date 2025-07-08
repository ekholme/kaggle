# the purpose of this script is to split the training data into train and validation sets that I actually have the outcome data for
import polars as pl

trn_full = pl.read_csv("playground_series/s5e5/data/train.csv")
trn_full.shape  # 750k rows

# set the fraction to sample
p = 0.8
seed = 408

trn2 = trn_full.sample(fraction=p, with_replacement=False, seed=seed)
val = trn_full.filter(~pl.col("id").is_in(trn2["id"]))

# write out
trn2.write_csv("playground_series/s5e5/data/train2.csv")
val.write_csv("playground_series/s5e5/data/validation.csv")
