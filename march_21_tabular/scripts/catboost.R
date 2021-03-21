

# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)
library(treesnip)
library(finetune)

trn <- vroom::vroom(here::here("march_21_tabular/data/train.csv")) %>%
  mutate(target = as_factor(target))
tst <- vroom::vroom(here::here("march_21_tabular/data/test.csv"))

theme_set(theme_minimal())


#splitting training from kaggle into a trn and a val set
#trn_split <- initial_split(trn, prop = 4/5, strata = target)

#trn <- training(trn_split)
#val <- testing(trn_split)

#getting resample folds
#folds <- vfold_cv(trn, v = 5)


# Preprocessing -----------------------------------------------------------

set.seed(0408)
mod_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id_var") %>%
  step_other(starts_with("cat"), threshold = .05) %>%
  step_nzv(all_predictors()) %>%
  prep()

trn_prepped <- bake(mod_rec, new_data = NULL)
tst_prepped <- bake(mod_rec, new_data = tst)

#splitting trn into folds
folds <- vfold_cv(trn_prepped, v = 5)

# Model Spec --------------------------------------------------------------

cat_spec <- boost_tree(
  trees = 500,
  min_n = tune(),
  tree_depth = tune()
) %>%
  set_mode("classification") %>%
  set_engine("catboost", nthread = 6)

#creating workflow
wf <- workflow() %>%
  add_model(cat_spec) %>%
  add_formula(target ~ .)

#creating parameters list
params <- parameters(
  min_n(),
  tree_depth()
)

params_grid <- grid_max_entropy(params, size = 10)


# Tuning Model ------------------------------------------------------------

set.seed(0408)
#doParallel::registerDoParallel()

cat_res <- tune_grid(
  wf,
  resamples = folds,
  grid = params_grid
)


# Finalizing Model --------------------------------------------------------

#show_best(cat_res)

best_params <- select_best(cat_res, "roc_auc")

#setting final model
cat_final <- finalize_model(cat_spec, best_params)

trained_final_mod <- cat_final %>%
  fit(formula = target ~ .,
      data = trn_prepped)


# Getting Test Predictions ------------------------------------------------

preds <- predict(trained_final_mod, new_data = tst_prepped, type = "prob")

sub <- tibble(
  id = tst$id,
  target = preds$.pred_1
)

write_csv(sub, here::here("march_21_tabular/submissions/baseline_catboost.csv"))