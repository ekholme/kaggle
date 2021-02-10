
# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)

trn <- vroom::vroom(here::here("feb_21_tabular/Data/train.csv"))
test <- vroom::vroom(here::here("feb_21_tabular/Data/test.csv"))


# Modeling ----------------------------------------------------------------

#create simple recipe
trn_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id") %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors())

#create a linear model specification
lm_spec <- linear_reg() %>%
  set_mode("regression") %>%
  set_engine("lm")

#add to workflow
trn_wf <- workflow() %>%
  add_recipe(trn_rec) %>%
  add_model(lm_spec)

#fit model
lm_fit <- trn_wf %>%
  fit(data = trn)

#examine fit
lm_fit %>%
  pull_workflow_fit() %>%
  tidy()

#get preds
preds <- predict(lm_fit, test) %>% as.numeric()

#get submission
sub <- bind_cols(test$id, preds) %>%
  set_names(c("id", "target"))


write_csv(sub, here::here("feb_21_tabular/submissions/baseline_lm.csv"))
