# code for a relatively straightforward linear model
library(tidyverse)
library(tidymodels)

#read in data
trn <- read_csv(here::here("may_22_tabular/data/train.csv"))
tst <- read_csv(here::here("may_22_tabular/data/test.csv"))

#set up resample
splits <- initial_split(trn)