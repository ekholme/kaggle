

# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)
library(treesnip)
library(finetune)

trn <- vroom::vroom(here::here("march_21_tabular/data/trn_lpa.csv")) %>%
  rename(target = ...33) %>%
  mutate(target = as_factor(target))
tst <- vroom::vroom(here::here("march_21_tabular/data/tst_lpa.csv"))


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
  step_string2factor(all_nominal(), -all_outcomes()) %>%
  step_other(starts_with("cat"), threshold = .1) %>% 
  step_nzv(all_predictors()) %>%
  prep()

trn_prepped <- bake(mod_rec, new_data = NULL)
tst_prepped <- bake(mod_rec, new_data = tst)

#splitting trn into folds
folds <- vfold_cv(trn_prepped, v = 5)

# Model Spec --------------------------------------------------------------

cat_spec <- boost_tree(
  trees = 2500,
  tree_depth = 2,
  min_n = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_mode("classification") %>%
  set_engine("catboost")

#creating workflow
wf <- workflow() %>%
  add_model(cat_spec) %>%
  add_formula(target ~ .)

#creating parameters list
params <- parameters(
  min_n(),
  finalize(mtry(), trn_prepped),
  learn_rate()
)

set.seed(0408)
params_grid <- grid_max_entropy(params, size = 10)


# Tuning Model ------------------------------------------------------------

set.seed(0409)
#doParallel::registerDoParallel(8)

cat_res <- tune_race_anova(
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

write_csv(sub, here::here("march_21_tabular/submissions/catboost_lpa.csv"))

## next step -- try kmeans cluster or lpa with continuous variables; try boxcox on continuous preds; try smote resampling for class imbalance?
## and maybe try frequency encoding categorical features?

##and start stacking models

# i also need to start working with the validation set to see where I'm getting the worst predictions