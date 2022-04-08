
# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)
library(stacks)

trn <- vroom::vroom(here::here("march_21_tabular/data/trn_lpa.csv")) %>%
  rename(target = ...33) %>%
  mutate(target = as_factor(target))

set.seed(0520)
split <- initial_split(trn, prop = 4/5, strata = target)

trn <- training(split)
val <- testing(split)

#set up kfolds
folds <- vfold_cv(trn, v = 5)

#set roc auc as metric
metric <- metric_set(roc_auc)


#setting up ctrl grid
ctrl_grid <- control_stack_grid()

# XGB Specification -------------------------------------------------------

xgb_spec <- boost_tree(mtry = tune(),
                       trees = tune(),
                       min_n = tune(),
                       tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

xgb_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id_var") %>%
  step_other(starts_with("cat"), threshold = .1) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors())

xgb_wf <- workflow() %>%
  add_recipe(xgb_rec) %>%
  add_model(xgb_spec)

#specifying parameters of xgb grid
params <- parameters(
  min_n(),
  finalize(mtry(), trn),
  learn_rate(),
  tree_depth(),
  loss_reduction(),
  trees()
)

set.seed(0408)
xgb_grid <- grid_max_entropy(params, size = 6) %>%
  add_row(min_n = 8, mtry = 20, learn_rate = 2.18e-10, tree_depth = 2, loss_reduction = .0000019, trees = 2000)


# Fit Model ---------------------------------------------------------------

doParallel::registerDoParallel()

#xgb
set.seed(0411)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric,
  control = ctrl_grid
)

save(xgb_res, file = here::here("march_21_tabular/data/xgb_res.RData"))
