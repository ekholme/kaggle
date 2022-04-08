
library(tidyverse)
library(tidymodels)
library(furrr)

trn_targ <- vroom::vroom(here::here("moa/Data/train_targets_scored.csv"))
trn <- vroom::vroom(here::here("moa/Data/train_features.csv"))
test <- vroom::vroom(here::here("moa/Data/test_features.csv"))
sample_sub <- vroom::vroom(here::here("moa/Data/sample_submission.csv"))

ctl_ids <- trn %>%
  filter(cp_type == "ctl_vehicle") %>%
  pull(sig_id)

test_ctl_ids <- test %>%
  filter(cp_type == "ctl_vehicle") %>%
  pull(sig_id)

#finding sums of cols
num_pos <- trn_targ %>%
  select(-sig_id) %>%
  summarize(across(everything(), sum)) %>%
  pivot_longer(cols = everything(),
               names_to = "var",
               values_to = "num_pos")

#will want to not model these
low_counts <- num_pos %>%
  filter(num_pos <= 1) %>%
  pull("var")

#getting targets to build models for
outs <- names(trn_targ[,2:207])

#making factors
trn <- trn %>%
  mutate(cp_time = as_factor(cp_time)) %>%
  filter(cp_type != "ctl_vehicle") %>%
  select(-cp_type) #getting rid of ctls bc they're all 0

trn_targ <- trn_targ %>%
  mutate(across(c(2:207), as_factor)) %>%
  filter(!(sig_id %in% ctl_ids))

test <- test %>%
  mutate(cp_time = as_factor(cp_time)) %>%
  select(-cp_type)

#global preprocessing
global_pre <- recipe(~ ., data = trn) %>%
  update_role(sig_id, new_role = "id") %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_pca(all_numeric(), threshold = .8) %>%
  step_dummy(cp_time, cp_dose) %>%
  prep()

trn_prepped <- juice(global_pre)
test_prepped <- bake(global_pre, new_data = test)


mod_fits <- function(out) {
  set.seed(0408)
  penalty <- grid_max_entropy(penalty(), size = 10)[[1,1]]
  
  if(out %in% low_counts) {
    
    return(0)
    
  } else {
  
  rec <- recipe(~ ., data = trn_prepped %>% bind_cols(trn_targ %>% select(all_of(out)))) %>%
    update_role(all_of(out), new_role = "outcome") %>%
    update_role(sig_id, new_role = "id")
  
  logreg_spec <- logistic_reg(penalty = penalty, mixture = 1) %>%
    set_engine("glmnet")
  
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(logreg_spec)
  
  set.seed(0408)
  logreg_fit <- wf %>%
    fit(trn_prepped %>% bind_cols(trn_targ %>%
                            select(c(out))))
  
  preds <- predict(logreg_fit, new_data = test_prepped, type = "prob") %>%
    select(2)
  
  return(preds)
  
  }
}

#predicting for all
yy <- future_map(outs, ~mod_fits(.x))

sub <- yy %>%
  bind_cols(test$sig_id, .) %>%
  set_names(names(sample_sub)) %>%
  mutate(across(where(is.numeric), ~if_else(sig_id %in% test_ctl_ids, 0, .)))


