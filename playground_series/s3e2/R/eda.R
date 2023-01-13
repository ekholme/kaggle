library(tidyverse)
library(tidymodels)
library(ggridges)

theme_set(theme_minimal())
# read in data
trn <- read_csv(here::here("playground_series/s3e2/data/train.csv"))
tst <- read_csv(here::here("playground_series/s3e2/data/test.csv"))
sample_sub <-read_csv(here::here("playground_series/s3e2/data/sample_submission.csv"))  
# target var
sum(trn$stroke) / nrow(trn)
# ok, so only 4% of people are having a stroke

# let's look at nominal vars
noms <- trn |>
    select(where(is.character), stroke, id) |>
    pivot_longer(
        cols = -c("stroke", "id"),
        names_to = "key",
        values_to = "val"
    )

noms_summed <- noms |>
    group_by(key, val) |>
    summarize(pct = sum(stroke) / n()) |>
    ungroup()

# plot
ggplot(noms_summed, aes(x = val, y = pct)) +
    geom_col() +
    facet_wrap(vars(key), scales = "free_x")
# ok, so big differences for ever_married, and fairly big differences for employment type and smoking status

# and let's look at the numeric predictors
nums <- trn |>
    select(where(is.numeric)) |>
    pivot_longer(
        cols = -c("stroke", "id"),
        names_to = "key",
        values_to = "val"
    )

nums |>
    ggplot(aes(x = val, y = key, fill = as.character(stroke))) +
    geom_density_ridges(alpha = .4) 
# prob obvious, but older people are much more likely to have a stroke
# also, people with higher bmi are more likely

# and it's also probably good to look at correlations among
# numeric predictors

trn |>
    select(where(is.numeric), -id) |>
    cor()
# highest correlation here is age with bmi at .39, but this isn't all that high

# are all of these people unique
length(unique(trn$id)) / nrow(trn)
#yes