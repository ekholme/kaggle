library(tidyverse)
library(skimr)

# Setup -------------------
df <- read_csv(here::here("july_22_tabular/data/data.csv"))

ids <- df$id

# looking at data
#skim(df)

# Processing -------------------

#get integer cols to log
log_cols <- c(paste0("f_0", 7:9), paste0("f_", 10:13))

# some basic transformations
X <- df |>
    mutate(
        across(log_cols, function(x) log(x + .01)),
        across(.cols = everything(), scale)
    )

Xs <- as.matrix(X[, 2:ncol(X)])

# Kmeans ------------------------

n_start <- 20
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

