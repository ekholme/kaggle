library(tidyverse)

# Setup -------------------
df <- read_csv(here::here("july_22_tabular/data/data.csv"))

ids <- df$id

mtrx <- as.matrix(df[, 2:ncol(df)])

# K-Means ----------------

#setting function parameters
n_start <- 20
ks <- 2:10

res <- map(ks, ~kmeans(mtrx, centers = .x, nstart = n_start))

err_tbl <- tibble(
    k = ks,
    err = map_dbl(res, ~ pluck(., 5))
)

# take a look at scree plot
ggplot(err_tbl, aes(x = ks, y = err)) +
    geom_point() +
    geom_line() +
    theme_minimal()
#ok well this isn't definitive but let's go with 6

# Getting Preds -------------

sub <- tibble(
    Id = ids,
    Predicted = res[[9]]$cluster - 1
)

write_csv(sub, here::here("july_22_tabular/submissions/baseline.csv"))
