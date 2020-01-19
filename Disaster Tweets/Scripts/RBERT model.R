#BERT Learning

library(tidyverse)
library(RBERT)
library(glmnet)

set.seed(0408)

train <- read_csv(here::here("Disaster Tweets/Data/train.csv"))
test <- read_csv(here::here("Disaster Tweets/Data/test.csv"))

bert_pretrained_dir <- RBERT::download_BERT_checkpoint(model = "bert_base_uncased")

#cleaning up some of the wonkiness in the text
all_df <- bind_rows(train, test)

rm(test, train)

all_df <- all_df %>%
  mutate( text = str_remove_all(text, "http[^[:space:]]*"),
          text = str_remove_all(text, "@[^[:space:]]*"),
          sequence_index = row_number())

bert_feats <- extract_features(
  examples = all_df$text,
  model = "bert_base_uncased",
  layer_indexes = 12,
  features = "output"
)

save.image(file = "~/Data/Large Files/bert_tweets.RData")

bert_out <- bert_feats$output %>%
  filter(token_index == 1) %>%
  select(-matches("seg|token|layer")) %>%
  left_join(all_df %>%
              select(id, sequence_index, target),
            by = "sequence_index") %>%
  select(id, target, everything(), -sequence_index)
#so, I think sequence_index will just be the row number because I didn't supply an identifier to the rbert model function --
#will want to test this later, though

rm(all_df)

train_targ <- all_df %>%
  filter(!is.na(target)) %>%
  select(target) %>%
  as_vector()

train_use <- bert_out %>%
  filter(!is.na(target)) %>%
  select(-c("id", "target"))

test_use <- bert_out %>%
  filter(is.na(target)) %>%
  select(-c("id", "target"))

ridge_mod <- cv.glmnet(x = as.matrix(train_use),
                       y = train_targ,
                       family = "binomial",
                       nfolds = 25,
                       alpha = 0)

test_preds <- predict(ridge_mod, as.matrix(test_use), s = "lambda.1se", type = "class")

submission <- tibble(
  id = bert_out %>% filter(is.na(target)) %>% pull(id),
  target = test_preds
)

write_csv(submission, here::here("Disaster Tweets/Submissions/bert_ridge_25fold.csv"))

#next step would be to get more layers, I think. And to try to predict out probabilities and ensemble them
  #also another step is to create some other features from location, keyword, and mining out some other text features (like in 
  #the initial BoW model I submitted). Will want to use those with the ridge model I think?

####below here is from the RBERT intro vignette

#bert_pretrained_dir <- RBERT::download_BERT_checkpoint(model = "bert_base_uncased")

#text_to_process <- c("Impulse is equal to the change in momentum.",
#                     "Changing momentum requires an impulse.",
#                     "An impulse is like a push.",
#                     "Impulse is force times time.")

#bert_feats <- extract_features(
#  examples = text_to_process,
#  ckpt_dir = bert_pretrained_dir,
#  layer_indexes = 1:12
#)

#output_vector1 <- bert_feats$output %>%
#  dplyr::filter(
#    sequence_index == 1, 
#    token == "[CLS]", 
#    layer_index == 12
#  ) %>% 
#  dplyr::select(dplyr::starts_with("V")) %>% 
#  unlist()
#output_vector1


#output_vectors <- bert_feats$output %>% 
#  dplyr::filter(token_index == 1, layer_index == 12)
#output_vectors
