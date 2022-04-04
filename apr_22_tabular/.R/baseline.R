library(tidyverse)

 trn_labs <- read_csv(here::here("apr_22_tabular/data/train_labels.csv"))
 tst <- read_csv(here::here("apr_22_tabular/data/test.csv"))
 
 x <- table(trn_labs$state)

pred <- as.integer(names(x)[x == max(x)])

sub <- tibble(
    sequence = unique(tst$sequence),
    state = pred
)

write_csv(sub, here::here("apr_22_tabular/submissions/baseline_mode.csv"))
