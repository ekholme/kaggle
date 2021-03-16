
# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)
library(finetune)

trn <- vroom::vroom(here::here("march_21_tabular/data/train.csv")) %>%
  mutate(target = as_factor(target))
tst <- vroom::vroom(here::here("march_21_tabular/data/test.csv"))

theme_set(theme_minimal())


# Preprocess --------------------------------------------------------------

trn_folds <- vfold_cv(trn, v = 5)

#creating very minimal preprocessing
baseline_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id_var") %>%
  step_other(starts_with("cat"), threshold = .1) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors())



# Model -------------------------------------------------------------------

xgb_spec <- boost_tree(mtry = tune(),
                       trees = tune(),
                       min_n = tune(),
                       tree_depth = 2,
                       learn_rate = tune(),
                       loss_reduction = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

#create workflow
trn_wf <- workflow() %>%
  add_recipe(baseline_rec) %>%
  add_model(xgb_spec)

#tune workflow
xgb_res <- tune_race_anova(
  trn_wf,
  resamples = trn_folds
)


