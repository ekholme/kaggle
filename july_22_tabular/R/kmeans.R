library(tidyverse)
library(skimr)
library(tidymodels)

# Setup -------------------
df <- read_csv(here::here("july_22_tabular/data/data.csv"))

ids <- df$id

# looking at data
skim(df)

# Processing -------------------

X <- df[, 2:ncol(df)]

rec <- recipe(~., data = X) |>
    step_normalize(all_predictors()) |>
    prep()

Xs <- bake(rec, new_data = NULL)
# Kmeans ------------------------

n_start <- 1
ks <- 2:10

res <- map(ks, ~ kmeans(Xs, centers = .x, nstart = n_start))

err_tbl <- tibble(
    k = ks,
    err = map_dbl(res, ~ pluck(., 5))
)

# take a look at scree plot
ggplot(err_tbl, aes(x = ks, y = err)) +
    geom_point() +
    geom_line() +
    theme_minimal()
#not super informative, but let's try 7

# Write Submission -----------------

sub <- tibble(
    Id = ids,
    Predicted = res[[6]]$cluster
)

write_csv(sub, here::here("july_22_tabular/submissions/normalize.csv"))