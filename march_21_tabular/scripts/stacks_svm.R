

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

# SVM Specification -------------------------------------------------------


svm_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id_var") %>%
  step_other(starts_with("cat"), threshold = .1) %>%
  step_scale(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_corr(all_numeric()) %>%
  step_nzv(all_predictors())

svm_spec <- svm_rbf(cost = tune("cost"), rbf_sigma = tune("sigma")) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wf <- workflow() %>%
  add_recipe(svm_rec) %>%
  add_model(svm_spec)


# Fit Model ---------------------------------------------------------------

doParallel::registerDoParallel()


set.seed(0410)
svm_res <- tune_grid(
  svm_wf,
  resamples = folds,
  grid = 4,
  metrics = metric,
  control = ctrl_grid
)
