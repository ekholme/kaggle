library(tidyverse)
library(tidymodels)

# read in data
trn <- read_csv(here::here("may_22_tabular/data/train.csv")) |>
    mutate(target = as.factor(target))
tst <- read_csv(here::here("may_22_tabular/data/test.csv"))

# set up resample
set.seed(0408)
splits <- initial_split(trn, prop = .8)
trn <- training(splits)
val <- testing(splits) # this will be more useful later as I experiment, but not really going to do anything with it here

# Create Recipe ---------------------------

rec <- recipe(target ~ ., data = trn) |>
    update_role(id, new_role = "id") |>
    step_mutate(f_27 = substring(f_27, 1, 3)) |>
    step_other(f_27, threshold = 20000) |>
    step_dummy(f_27) |>
    step_log(f_07:f_18, offset = 1) |>
    step_normalize(all_numeric_predictors())

# Specify Model ---------------------------

xgb_spec <- boost_tree(
    trees = 500,
    tree_depth = 6,
    learn_rate = .1
) |>
    set_engine("xgboost") |>
    set_mode("classification")

# XGB WF ----------------------------------

xgb_wf <- workflow() |>
    add_model(xgb_spec) |>
    add_recipe(rec)

# Fit -------------------------------------

xgb_fit <- xgb_wf |>
    fit(data = trn)

# Predict ---------------------------------

preds <- predict(xgb_fit, new_data = tst, type = "prob")

# Write Submission -----------------------

sub <- tibble(
    id = tst$id,
    target = preds$.pred_1
)

write_csv(sub, here::here("may_22_tabular/submissions/xgb_base.csv"))
