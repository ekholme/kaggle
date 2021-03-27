

# Setup -------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(finetune)
library(tidyLPA)

trn <- vroom::vroom(here::here("march_21_tabular/data/train.csv")) %>%
  mutate(target = as_factor(target))
tst <- vroom::vroom(here::here("march_21_tabular/data/test.csv"))

y_trn <- trn$target

trn <- trn %>%
  select(-target)

all_df <- bind_rows(trn, tst)


# LPA for features --------------------------------------------------------

lpa_vars <- str_subset(names(all_df), "cont")

lpa_fits <- all_df %>%
  estimate_profiles(5, variances = "varying", covariances = "zero", select_vars = lpa_vars)



#get fit data
classes <- get_data(lpa_fits)

class_df <- classes %>%
  select(-matches("CPR|_number")) %>%
  mutate(Class = str_replace_all(Class, "^", "class_"))

trn_lpa <- class_df %>%
  filter(id %in% trn$id) %>%
  bind_cols(y_trn)

tst_lpa <- class_df %>%
  filter(id %in% tst$id)

write_csv(trn_lpa, here::here("march_21_tabular/data/trn_lpa.csv"))

write_csv(tst_lpa, here::here("march_21_tabular/data/tst_lpa.csv"))
