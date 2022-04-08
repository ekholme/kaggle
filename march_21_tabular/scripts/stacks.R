

# Setup -------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(treesnip)
library(finetune)
library(stacks)

load(here::here("march_21_tabular/data/xgb_res.RData"))
load(here::here("march_21_tabular/data/lasso_res.RData"))
load(here::here("march_21_tabular/data/knn_res.RData"))

#making the stacked dataset
data_st <- stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(lasso_res) %>%
  add_candidates(xgb_res)

#using lasso to combine predictions from candidate models
model_st <- data_st %>%
  blend_predictions()

#doing the final fit
stacks_fit <- model_st %>%
  fit_members()


# Predicting --------------------------------------------------------------

#read in tst data
tst <- vroom::vroom(here::here("march_21_tabular/data/tst_lpa.csv"))

preds <- predict(stacks_fit, tst, type = "prob")

sub <- tibble(
  id = tst$id,
  target = preds$.pred_1
)

write_csv(sub, here::here("march_21_tabular/data/stacks_submission.csv"))
