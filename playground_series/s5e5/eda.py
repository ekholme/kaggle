import polars as pl
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

trn = pl.read_csv("playground_series/s5e5/data/train2.csv")

trn.shape

# take a look at the data
trn.head()

# and the column names
trn.columns

# and a null count
trn.null_count()

# distribution of y
sns.histplot(trn["Calories"])

plt.show()
# ok, so, super skewed distribution with a long right tail

# let's see the log distribution
sns.histplot(trn["Calories"].log())
plt.show()
# eh, not really a lognormal distribution
# we can figure it out more as we go

# make a faceted histogram showing distributions for all of the quantitative variables in trn
trn_long = trn.select(pl.exclude("Sex")).unpivot(index="id")

g = sns.FacetGrid(
    trn_long.to_pandas(), col="variable", col_wrap=3, sharex=False, sharey=True
)
g.map(sns.histplot, "value")
plt.show()

# ok, now what if we look at the bivariate correlations between calories and everything else
trn = trn.with_columns(male=pl.col("Sex") == "male").select(pl.exclude("Sex"))

df_mat = trn.select(pl.exclude("id")).to_numpy()

np.corrcoef(df_mat, rowvar=False)
# eh, this is hard to read

cor_res = trn.select(pl.exclude("id")).corr()
# ok so the largest correlations are with body temp, heart rate, and duration. other correlations are pretty small.
# they might have some interactions, but a reasonable place to start is just with a linear model where those 3 are predictors of calories
