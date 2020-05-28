
library(tidyverse)
library(tidytext)
library(keras)
library(tm)


train <- read_csv(here::here("Disaster Tweets/Data/train.csv"))
test <- read_csv(here::here("Disaster Tweets/Data/test.csv"))

train <- na.omit(train)

all_df <- bind_rows(train, test)

all_df <- all_df %>%
  mutate(text = str_remove_all(text, "http[^[:space:]]*"),
         text = str_remove_all(text, "@[^[:space:]]*"),
         text = str_remove_all(text, "[^[:alpha:][:space:]]*"),
         text = stripWhitespace(text),
         text = str_to_lower(text))

View(head(all_df))

train <- all_df %>%
  filter(!is.na(target))

test <- all_df %>%
  filter(is.na(target))


# Setting Up Data for Use -------------------------------------------------

trn_text <- train$text
trn_tar <- train$target

trn_new <- tibble(
  text = trn_text,
  tar = trn_tar
)

set.seed(0408)
samp_size <- floor(.75*length(trn_text))

train_ind <- sample(seq_len(length(trn_text)), size = samp_size)

train <- trn_new[train_ind, ]
val <- trn_new[-train_ind, ]



# Setting Up Keras Model --------------------------------------------------

max_features <- 15000
tokenizer <- text_tokenizer(num_words = max_features)


tokenizer %>%  fit_text_tokenizer(train$text)

tokenizer$word_index %>%head()

#will convert text into a sequence of integers
text_seqs <- texts_to_sequences(tokenizer, train$text)

#setting parameters for keras model
maxlen <- 120
batch_size <- 64
embedding_dims <- 16
#hidden_dims <- 16
epochs <- 10



x_train <- text_seqs %>% pad_sequences(maxlen = maxlen, padding = "post")
dim(x_train)


y_train <- train$tar


#setting up the keras model
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = embedding_dims) %>%
  layer_gru(units = embedding_dims, dropout = .2, recurrent_dropout = .2) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

history <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = .2
  )
#this doesn't seem to give me a very good model

##trying with dense layers 


model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.3) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

history <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = .2
  )
#this model is actually pretty good.