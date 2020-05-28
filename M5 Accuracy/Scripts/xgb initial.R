
library(tidyverse)
library(tidymodels)
library(janitor)
library(RcppRoll)
library(xgboost)


first <- 1914
len <- 28

calendar <- vroom::vroom(here::here("M5 Accuracy/Data/calendar.csv"), col_select = -c("date", "weekday", "event_type_1", "event_type_2"))
train <- vroom::vroom(here::here("M5 Accuracy/Data/sales_train_validation.csv"), col_select = -num_range("d_", 1:1000))
sell_prices <- vroom::vroom(here::here("M5 Accuracy/Data/sell_prices.csv"))

# Setting Up Functions ----------------------------------------------------
d2int <- function(X) {
  X %>% extract(d, into = "d", "([0-9]+)", convert = TRUE)
}

demand_features <- function(X) {
  X %>% 
    group_by(id) %>% 
    mutate(lag_7 = dplyr::lag(demand, 7),
           lag_28 = dplyr::lag(demand, 28),
           roll_lag7_w7 = roll_meanr(lag_7, 7),
           roll_lag7_w28 = roll_meanr(lag_7, 28),
           roll_lag28_w7 = roll_meanr(lag_28, 7),
           roll_lag28_w28 = roll_meanr(lag_28, 28)) %>% 
    ungroup() 
}

pred_days <- paste0("d_", first:(first+2*len-1))


# Preparing Data ----------------------------------------------------------

train[, pred_days] <- NA


train <- train %>%
  mutate(id = str_remove_all(id, "_validation")) %>%
  pivot_longer(cols = starts_with("d_"),
               names_to = "d",
               values_to = "demand") %>%
  d2int() %>%
  left_join(calendar %>% d2int(),
            by = "d") %>%
  left_join(sell_prices,
            by = c("store_id", "item_id", "wm_yr_wk")) %>%
  select(-c("wm_yr_wk", "item_id")) %>%
  mutate_at(c("dept_id", "store_id", "cat_id", "state_id", "event_name_1", "event_name_2"), function(x) {
    as_factor(x) %>% as.integer(x)
  }) %>%
  demand_features() %>%
  filter(d >= first | !is.na(roll_lag28_w28))

test <- train %>%
  filter(d >= first - 56)

train <- train %>%
  filter(d < first)

rm(sell_prices, calendar)

#finishing preprocessing in a recipe
pre_rec <- recipe(demand ~ ., data = train) %>%
  update_role(id, new_role = "ID")

# Setting XGB Model Specs -------------------------------------------------
xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")


# Setting Workflow --------------------------------------------------------

xgb_wf <- workflow() %>%
  add_recipe(pre_rec) %>%
  add_model(xgb_spec)

# Setting Tuning Grid -----------------------------------------------------

params <- parameters(xgb_wf) %>%
  update(mtry = finalize(mtry(), train))

xgb_grid <- params %>%
  grid_max_entropy(size = 20)


# Setting Up Resampling ---------------------------------------------------

resamples <- vfold_cv(train, v = 5)


# Fitting Model -----------------------------------------------------------

#doParallel::registerDoParallel()
set.seed(0408)
xgb_tuned <- xgb_wf %>%
  tune_grid(
    resamples = resamples,
    grid = xgb_grid,
    metrics = metric_set(rmse)
  )