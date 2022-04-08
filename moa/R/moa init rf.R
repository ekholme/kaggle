
library(tidyverse)
library(tidymodels)

trn_targ <- vroom::vroom(here::here("moa/Data/train_targets_scored.csv"))
trn <- vroom::vroom(here::here("moa/Data/train_features.csv"))
test <- vroom::vroom(here::here("moa/data/test_features.csv"))

#getting names
outs <- names(trn_targ[,2:207])

trn <- trn %>%
  mutate(cp_time = as_factor(cp_time))

trn_targ <- trn_targ %>%
  mutate(across(c(2:207), as_factor))

test <- test %>%
  mutate(cp_time = as_factor(cp_time))

#creating function
full_mod <- function(out) {
  rec <- recipe(~ ., data = trn %>% bind_cols(trn_targ %>% select(all_of(out)))) %>%
    update_role(all_of(out), new_role = "outcome") %>%
    update_role(sig_id, new_role = "id") %>%
    step_center(all_numeric()) %>%
    step_scale(all_numeric()) %>%
    step_pca(all_numeric(), num_comp = 10)
  
  rf_spec <- rand_forest() %>%
    set_engine("ranger") %>%
    set_mode("classification")
  
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(rf_spec)
  
  fit_wf <- wf %>%
    fit(trn %>% bind_cols(trn_targ %>%
                            select(c(out))))
  
  preds <- predict(fit_wf, new_data = test, type = "prob") %>%
    select(2)
  
  return(preds)
}

#getting all the preds
yy <- map(outs, ~full_mod(.x))

sub <- yy %>%
  bind_cols(test$sig_id, .) %>%
  set_names(names(trn_targ))
