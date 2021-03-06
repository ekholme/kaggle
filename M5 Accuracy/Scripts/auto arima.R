
#this will fit an arima model for each product in the dataset

library(tidyverse)
library(tidymodels)
library(forecast)
library(janitor)
library(timetk)
library(zoo)
library(sweep)
#library(fable)
#library(tsibble)
library(lubridate)

calendar <- read_csv(here::here("M5 Accuracy/Data/calendar.csv"))
train <- vroom::vroom(here::here("M5 Accuracy/Data/sales_train_validation.csv"))

#reshaping train data to long
#taking the average number of items sold across all stores on a given day so I have to fit fewer models
train <- train %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "day",
    values_to = "sold"
  ) %>%
  left_join(calendar %>% 
              select(d, date, weekday),
            by = c("day" = "d")) %>%
  group_by(item_id, date) %>% 
  summarize(sold = mean(sold, na.rm = TRUE)) %>%
  ungroup()

#setting up function to get arimas for items
arima_preds <- function(x) {
  #safely(pb$tick()$print()) -- this doesn't work when wrapped in safely()
  
  train %>%
    filter(item_id == x) %>%
    select(date, sold) %>%
    tk_ts(start = start) %>%
    auto.arima() %>%
    forecast(h = 56) %>%
    sw_sweep(timetk_idx = TRUE) %>%
    filter(key == "forecast") %>%
    select(index, sold)
}

#define parameters for function
start <- unique(train$date)[[1]]
items <- unique(train$item_id)
#items_test <- sample(unique(train$id), size = 50)

##setting up a progress bar to use in my function
#pb <- progress_estimated(length(items))

doParallel::registerDoParallel()
set.seed(0408)
mod_preds <- map(items,
                 arima_preds) %>%
  tibble(
    item = items,
    preds = .
  ) %>%
  unnest(preds)

#getting subs
trn_ind <- vroom::vroom(here::here("M5 Accuracy/Data/sales_train_validation.csv")) %>%
  select(id, item_id) %>%
  left_join(mod_preds,
            by = c("item_id" = "item")) %>%
  mutate(id = if_else(index > as_date("2016-05-22"), str_replace_all(id, "validation", "evaluation"), id),
         f_num = paste0("F", rep(1:28, times = 20*length(items))))

val <- trn_ind %>%
  filter(str_detect(id, "validation")) %>%
  select(-c("item_id", "index")) %>%
  pivot_wider(names_from = f_num,
              values_from = sold)

eval <- trn_ind %>%
  filter(str_detect(id, "evaluation")) %>%
  select(-c("item_id", "index")) %>%
  pivot_wider(names_from = f_num,
              values_from = sold)

sub <- bind_rows(val, eval)

write_csv(sub, here::here("M5 Accuracy/Submissions/auto_arima.csv"))
