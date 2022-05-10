#just a simple baseline model
library(tidyverse)

trn <- read_csv(here::here("may_22_tabular/data/train.csv"))
tst <- read_csv(here::here("may_22_tabular/data/test.csv"))