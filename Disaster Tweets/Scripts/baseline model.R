library(tidyverse)

#just grabbing the mode and putting it into a submission
train <- read_csv(here::here("Disaster Tweets/Data/train.csv"))
test <- read_csv(here::here("Disaster Tweets/Data/test.csv"))

train %>%
  count(target, sort = TRUE) %>%
  pluck(1, 1)
#mode is 0

sub <- tibble(
  id = test$id,
  target = 0
)

write_csv(sub, here::here("Disaster Tweets/Submissions/baseline.csv"))
