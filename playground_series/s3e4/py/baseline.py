import pandas as pd
import statistics

path = "./playground_series/s3e4/data/"

trn = pd.read_csv(path + 'train.csv')
tst = pd.read_csv(path + 'test.csv')

trn.head(1)

pred = statistics.mean(trn.Class)

data = {
    'id' : trn.id,
    'Class' : pred
}


sub = pd.DataFrame(data = data)

sub.to_csv("./playground_series/s3e4/submissions/baseline_py.csv")
    
