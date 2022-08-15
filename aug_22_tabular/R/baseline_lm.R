library(tidyverse)
library(tidymodels)

trn <- read_csv(here::here("aug_22_tabular/data/train.csv")) |>
    mutate(failure = as.character(failure))

tst <- read_csv(here::here("aug_22_tabular/data/test.csv"))

# Set Up Initial Split --------------------

splits <- initial_split(trn, prop = .8)

trn1 <- training(splits)
val <- testing(splits)

# Set Up Initial Recipe -------------------

rec <- recipe(failure ~ ., data = trn) |>
    update_role(id, new_role = "id") |>
    step_impute_mode(all_nominal_predictors()) |>
    step_impute_median(all_numeric_predictors()) |>
    step_YeoJohnson(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors())

# Set Up Initial Model ---------------------

#will tune this later
#right now running a pure lasso model
spec <- logistic_reg(penalty = .01) |>
    set_engine("glmnet")


# Create Workflow ----------------------

wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

# Fit ---------------------------------

wf_fit <- fit(wf, data = trn)


# Predict -----------------------------

preds <- predict(wf_fit, new_data = tst, type = "prob")

sub <- tibble(
    id = tst$id,
    failure = preds$.pred_1
)

write_csv(sub, here::here("aug_22_tabular/submissions/lasso_baseline.csv"))
