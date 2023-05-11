import polars as pl
import seaborn as sns
import statistics

#apply default seaborn theme
sns.set_theme()

trn = pl.read_csv('./playground_series/s3e14/data/train.csv')
tst = pl.read_csv('./playground_series/s3e14/data/test.csv')

# Examine ------------------
trn.shape

trn.head()

#get column types
trn.dtypes
#ok so everything is numeric

# Target --------------
trn['yield'].mean()
trn['yield'].std()

sns.displot(trn, x='yield')
