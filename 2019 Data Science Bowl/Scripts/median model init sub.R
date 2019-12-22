library(tidyverse)
library(here)

train_labels <- data.table::fread("../input/data-science-bowl-2019/train_labels.csv", stringsAsFactors = F)
test <- data.table::fread("../input/data-science-bowl-2019/test.csv")

last_assessment <- test %>%
  filter(type == "Assessment") %>% 
  arrange(desc(timestamp)) %>% 
  distinct(installation_id, .keep_all = T) %>% 
  select(installation_id, title)  

median_table <- train_labels %>%
  group_by(title) %>% 
  summarise(accuracy_group = median(accuracy_group, na.rm = T)) %>% 
  ungroup()

submission <- last_assessment %>%
  left_join(median_table, by = "title") %>% 
  select(-title)

write.csv(submission, "submission.csv", row.names = F)
