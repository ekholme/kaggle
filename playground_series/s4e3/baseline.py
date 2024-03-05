import polars as pl
import statistics
import numpy as np

#read in data
trn = pl.read_csv("./data/train.csv")
tst = pl.read_csv("./data/test.csv")

#check cols in trn
trn.head()

#get col names
nms = trn.columns

#get target cols
targets = trn.columns[(len(nms)-7):len(nms)]

#get the target columns from trn
tar_cols = trn[targets]

#calculate the row-wise sum of tar_cols
tar_cols_sum = tar_cols.sum(axis=1)

#count the unique values in tar_cols_sum and include the counts of each unique entry
tar_cols_counts = tar_cols_sum.value_counts()
#ok, so most entries have only 1 defect, but some have none and some have multiple defects

#for a simple baseline, I'll just take the mean of each target column and use that as the predicted value
preds = tar_cols.mean(axis=0)

xx = [pl.select(pl.repeat(preds[i], tst.shape[0])).to_series().alias(i) for i in targets]

column_dict = {s.name: s for s in xx}
pred_df = pl.DataFrame(column_dict)

#horizontally concatenate preds with tst["id"]
submission = pl.concat([tst.select("id"), pred_df], how="horizontal")

#write out to csv
submission.write_csv("./submissions/baseline.csv")