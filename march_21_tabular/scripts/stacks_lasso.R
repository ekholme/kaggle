
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

#set base recipe
base_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id_var")

#setting up ctrl grid
ctrl_grid <- control_stack_grid()


# Lasso Specification -----------------------------------------------------

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

lasso_rec <- base_rec %>%
  step_other(starts_with("cat"), threshold = .1) %>%
  step_scale(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_corr(all_numeric()) %>%
  step_nzv(all_predictors())

lasso_wf <- workflow() %>%
  add_recipe(lasso_rec) %>%
  add_model(lasso_spec)


# Lasso Fitting -----------------------------------------------------------

doParallel::registerDoParallel()

set.seed(0409)
lasso_res <- tune_grid(
  lasso_wf,
  resamples = folds,
  grid = 4,
  metrics = metric,
  control = ctrl_grid
)

save(lasso_res, file = here::here("march_21_tabular/data/lasso_res.RData"))
