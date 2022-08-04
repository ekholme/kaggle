# Setup ---------------------------------

library(tidyverse)
library(tidymodels)

df <- read_csv(here::here("july_22_tabular/data/data.csv"))

ids <- df$id

# Process -------------------------------

X <- df[, 2:ncol(df)]

rec <- recipe(~., data = X) |>
    step_YeoJohnson(all_predictors()) |>
    prep()

Xs <- bake(rec, new_data = NULL)

# Fit -----------------------------------

res <- kmeans(Xs, 7)

# Explore Feats -------------------------

X2 <- broom::augment(res, Xs)

x_summed <- X2 |>
    group_by(.cluster) |>
    summarize(across(everything(), mean)) |>
    pivot_longer(
        cols = starts_with("f_"),
        names_to = "var",
        values_to = "avg"
    ) |>
    ungroup()

# plotting
x_summed |>
    ggplot(aes(x = avg, y = var, color = .cluster)) +
    geom_point(alpha = .6) +
    theme_minimal()

#ok so we want spread here -- meaning the only variables that are useful are:
#f_07:f_13, f_22:f_28