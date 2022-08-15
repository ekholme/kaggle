library(tidyverse)

trn <- read_csv(here::here("aug_22_tabular/data/train.csv"))

tst_id <- read_csv(here::here("aug_22_tabular/data/test.csv"))$id

pred <- mean(trn$failure)

sub <- tibble(
    id = tst_id,
    failure = pred
)

write_csv(sub, here::here("aug_22_tabular/submissions/baseline_r.csv"))
