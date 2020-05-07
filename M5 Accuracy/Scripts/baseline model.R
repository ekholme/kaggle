
library(tidyverse)
library(janitor)
library(lubridate)

#This script will create a baseline model for prediction. To predict values, I'm simply going to use the value from the same date last year.
#this will serve as the baseline to compare later models against.

calendar <- read_csv(here::here("M5 Accuracy/Data/calendar.csv"))
train <- read_csv(here::here("M5 Accuracy/Data/sales_train_validation.csv"))
sample_sub <- read_csv(here::here("M5 Accuracy/Data/sample_submission.csv"))

#reshaping and joining calendar data
train <- train %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "day",
    values_to = "sold"
  ) %>%
  left_join(calendar %>% 
              select(d, date, weekday),
            by = c("day" = "d"))

#calculating the validation periods and evaluation periods
val_days <- (max(calendar$date) + days(1)) + days(0:27)
eval_days <- (max(val_days) + days(1)) + days(0:27)

val_days - years(1)

val_base <- train %>%
  filter(date %in% (val_days - years(1))) %>%
  select(id, date, sold) %>%
  pivot_wider(names_from = date,
              values_from = sold)

names(val_base) <- names(sample_sub)

eval_base <- train %>%
  filter(date %in% (eval_days - years(1))) %>%
  select(id, date, sold) %>%
  pivot_wider(names_from = date,
              values_from = sold) %>%
  mutate(id = str_replace_all(id, "validation", "evaluation"))

names(eval_base) <- names(sample_sub)

submission <- bind_rows(val_base, eval_base)

write_csv(submission, here::here("M5 Accuracy/Submissions/baseline.csv"))
