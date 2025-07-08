import polars as pl

stub = "playground_series/s5e7/"

trn = pl.read_csv(stub + "data/train.csv")
tst = pl.read_csv(stub + "data/test.csv")

y = trn["Personality"]


def count_classes(x):
    res = {}
    for i in x:
        if i in res:
            res[i] += 1
        else:
            res[i] = 1
    return res


counts = count_classes(y)

mode_class = max(counts, key=counts.get)
# this also works: mode_class = sorted(counts)[0]

# writing out a submission using the modal data
sub = pl.DataFrame(
    {
        "id": tst["id"],
        "Personality": mode_class,
    }
)

sub.write_csv(stub + "submissions/baseline_mode.csv")
