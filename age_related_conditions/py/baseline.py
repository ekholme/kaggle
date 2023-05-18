import polars as pl
import statistics
from sklearn.metrics import log_loss

trn = pl.read_csv("./age_related_conditions/data/trn.csv")
val = pl.read_csv("./age_related_conditions/data/validation.csv")

trn.shape

pred = statistics.mean(trn['Class'])

preds = [pred] * len(val['Class'])

#calculate log loss
ll = log_loss(val['Class'], preds)
#log loss of .3747 -- this probably isn't good