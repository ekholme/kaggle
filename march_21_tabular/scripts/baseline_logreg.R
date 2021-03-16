

# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)
library(finetune)

trn <- vroom::vroom(here::here("march_21_tabular/data/train.csv")) %>%
  mutate(target = as_factor(target))
tst <- vroom::vroom(here::here("march_21_tabular/data/test.csv"))

theme_set(theme_minimal())

#setting up validation set
set.seed(0408)
split <- initial_split(trn, prop = 4/5)

trn2 <- training(split)
val <- testing(split)

# Preprocess --------------------------------------------------------------


trn_folds <- vfold_cv(trn2, v = 5)

#creating very minimal preprocessing
logreg_rec <- recipe(target ~ ., data = trn2) %>%
  update_role(id, new_role = "id_var") %>%
  step_other(starts_with("cat"), threshold = .1) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors())


# Model -------------------------------------------------------------------


#setting up model spec
modl_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")


#setting workflow
trn_wf <- workflow() %>%
  add_recipe(logreg_rec) %>%
  add_model(modl_spec)

#tuning model
logreg_res <- tune_race_anova(
  trn_wf,
  resamples = trn_folds
)


# Finalize & Predict ------------------------------------------------------


#finalizing workflow
params <- select_best(logreg_res, metric = "roc_auc")

#finalizing workflow
fin_wf <- trn_wf %>%
  finalize_workflow(params)

#predicting
preds <- predict(fin_wf %>% fit(trn2), tst, type = "prob")



# Make Submission ---------------------------------------------------------

sub <- tibble(
  id = tst$id,
  target = preds$.pred_1
)

write_csv(sub, here::here("march_21_tabular/submissions/baseline_logreg.csv"))
