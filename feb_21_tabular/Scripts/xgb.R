
# Setup -------------------------------------------------------------------


library(tidyverse)
library(tidymodels)
library(finetune)
library(glue)

trn <- vroom::vroom(here::here("feb_21_tabular/Data/train.csv"))
test <- vroom::vroom(here::here("feb_21_tabular/Data/test.csv"))


#target mean encoder
encode_means <- function(df, cat_var, targ, na.rm = TRUE) {
  
  tmp <- df %>%
    summarize(
      all_mu = mean(.data[[targ]], na.rm = na.rm),
      all_var = var(.data[[targ]], na.rm = na.rm)
    )
  
  ret <- df %>%
    group_by(.data[[cat_var]]) %>%
    summarize(
      grp_mu = mean(.data[[targ]], na.rm = na.rm),
      grp_var = var(.data[[targ]], na.rm = na.rm)
    ) %>%
    ungroup() %>%
    mutate(
      all_mu = tmp$all_mu,
      all_var = tmp$all_var,
      b = grp_var/(grp_var + all_var),
      wmean = (1-b)*grp_mu + b*all_mu 
    ) %>%
    select(.data[[cat_var]], 
           wmean)
  
  names(ret) <- (c(cat_var, paste0(cat_var, "_wmean")))
  
  return(ret)
  
}

# PreProcessing -----------------------------------------------------------

cats <- str_subset(names(trn), "cat")

trn %>%
  encode_means(cat_var = "cat0", targ = "target") %>%
  View()

match_tbls <- map(cats, ~encode_means(df = trn, cat_var = .x, targ = "target"))g

xx <- trn %>%
  left_join(match_tbls[[1]], by = "cat0") %>%
  glimpse()

##next step -- write a map that will join and keep only the necessary cols
map_encode <- function(tbls) {
  
  by <- names(tbls)[[1]]
  
}

# Modeling ----------------------------------------------------------------

#resamples
trn_folds <- vfold_cv(trn, v = 5)

#create simple recipe
trn_rec <- recipe(target ~ ., data = trn) %>%
  update_role(id, new_role = "id") %>%
  #step_normalize(starts_with("cont")) %>%
  #step_pca(starts_with("cont")) %>%
  #step_other(starts_with("cat"), threshold = .1) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors())

#setting model specification
xgb_spec <- boost_tree(mtry = tune(),
                       trees = tune(),
                       min_n = tune(),
                       tree_depth = 2,
                       learn_rate = tune(),
                       loss_reduction = tune()) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

#creating workflow
trn_wf <- workflow() %>%
  add_recipe(trn_rec) %>%
  add_model(xgb_spec)

#tune workflow
xgb_res <- tune_race_anova(
  trn_wf,
  resamples = trn_folds
)

#select best
params <- select_best(xgb_res, metric = "rmse")

#finalize model
fin_wf <- trn_wf %>%
  finalize_workflow(params)

#predict test vals
preds <- predict(fin_wf %>% fit(trn), test)

#make sub
sub <- bind_cols(test$id, pull(preds)) %>%
  set_names(c("id", "target"))

#write sub
write_csv(sub, here::here("feb_21_tabular/submissions/xgb2.csv"))
