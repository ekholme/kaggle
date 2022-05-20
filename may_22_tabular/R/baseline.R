#just a simple baseline model
library(tidyverse)

trn <- read_csv(here::here("may_22_tabular/data/train.csv"))
tst <- read_csv(here::here("may_22_tabular/data/test.csv"))

pred <- sum(trn$target) / nrow(trn)

sub <- tibble(
    id = tst$id,
    target = pred
)

write_csv(sub, here::here("may_22_tabular/submissions/baseline_sub.csv"))
