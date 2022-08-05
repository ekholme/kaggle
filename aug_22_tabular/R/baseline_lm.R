library(tidyverse)
library(tidymodels)

trn <- read_csv(here::here("aug_22_tabular/data/train.csv"))
tst <- read_csv(here::here("aug_22_tabular/data/test.csv"))

# Set Up Initial Split --------------------

splits <- initial_split(trn, prop = .8)

trn1 <- training(splits)
val <- testing(splits)

# Set Up Initial Recipe -------------------

rec <- recipe(failure ~ ., data = trn1) |>
    update_role(id, new_role = "id") |>
    step_impute_mode(all_nominal_predictors()) |>
    step_impute_median(all_numeric_predictors()) |>
    step_YeoJohnson(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors())

# Set Up Initial Model ---------------------

#will tune this later
#right now running a pure lasso model
spec <- logistic_reg(penalty = .1) |>
    set_engine("glmnet")


# Create Workflow ----------------------

wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

# Fit ---------------------------------

