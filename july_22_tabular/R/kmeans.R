library(tidyverse)
library(skimr)
library(tidymodels)

# Setup -------------------
df <- read_csv(here::here("july_22_tabular/data/data.csv"))

ids <- df$id

# Processing -------------------

keep_vars <- paste0("f_", c("07", "08", "09"))
keep_vars <- c(keep_vars, paste0("f_", c(10:13, 22:28)))

X <- df[keep_vars]

rec <- recipe(~., data = X) |>
    step_YeoJohnson(all_predictors()) |>
    prep()

Xs <- bake(rec, new_data = NULL)
# Kmeans ------------------------

n_start <- 10
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

write_csv(sub, here::here("july_22_tabular/submissions/fewer_feats.csv"))