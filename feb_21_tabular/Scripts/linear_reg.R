
# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)

trn <- vroom::vroom(here::here("feb_21_tabular/Data/train.csv"))
test <- vroom::vroom(here::here("feb_21_tabular/Data/test.csv"))


# Modeling ----------------------------------------------------------------

#resamples
trn_folds <- vfold_cv(trn, v = 5)

#create simple recipe
trn_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id") %>%
  step_normalize(starts_with("cont")) %>%
  step_pca(starts_with("cont")) %>%
  step_other(starts_with("cat"), threshold = .1) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors())

#create a linear model specification
lm_spec <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

#add to workflow
trn_wf <- workflow() %>%
  add_recipe(trn_rec) %>%
  add_model(lm_spec)

#fit model
lm_res <- tune_grid(
  trn_wf,
  resamples = trn_folds
)

#choose tuning parameters
params <- select_best(lm_res, metric = "rmse")

#finalizing workflow
fin_wf <- trn_wf %>%
  finalize_workflow(params)

#predicting
preds <- predict(fin_wf %>% fit(trn), test)

#submission
sub <- bind_cols(test$id, pull(preds)) %>%
  set_names(c("id", "target"))

##SUBMIT MODEL

write_csv(sub, here::here("feb_21_tabular/submissions/lm_sub2.csv"))
