
library(tidyverse)
library(lubridate)
library(vroom)
library(tidymodels)

##To do
#1. add natural spline to day and month
#2. allow interactions between all vars except nom
#3. select best1se model
##later -- try xgboost, mars, rf
##merge branch with master and work through xgboost, rf, etc in separate scripts

miss_plot <- function(data, color1 = "steelblue1", color2 = "steelblue4", bound = 0) {
  miss_tab <<- tibble(
    column = names(data),
    perc_miss = map_dbl(data, function(x) sum(is.na(x))/length(x))
  ) %>%
    filter(perc_miss > bound)
  
  ggplot(miss_tab, aes(x = column, y = perc_miss)) +
    geom_bar(stat = "identity", aes(fill = ..y..)) +
    scale_y_continuous(labels = scales::percent) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) + 
    scale_fill_gradient(low = color1, high = color2, name = "Percent Missing") +
    labs(
      title = "Missingness by Variable",
      y = "Percent Missing",
      x = "Variables"
    )
}

mode_func <- function(df, var) {
  df %>%
    count({{var}}, sort = TRUE) %>%
    top_n(n = 1) %>%
    pull(1)
}

train <- vroom(here::here("Categorical Feature Challenge 2/Data/train.csv"))

ord_vars <- str_subset(names(train), "ord")

ord_vals <- map(ord_vars,
                function(x) {
                  train %>%
                    distinct(.data[[x]]) %>%
                    pull()
                })

ord_tbl <- tibble(
  vars = ord_vars,
  vals = ord_vals
)

mode_ord3 <- mode_func(train, ord_3)
mode_ord4 <- mode_func(train, ord_4)


train <- train %>%
  mutate(ord_1 = str_replace_all(ord_1, c("Novice" = "0", "Contributor" = "1", "Expert" = "2",
                                          "Master" = "3", "Grandmaster" = "4")),
         ord_2 = str_replace_all(ord_2, c("Freezing" = "0", "Cold" = "1", "Warm" = "2", "^Hot$" = "3", 
                                          "^Boiling Hot$" = "4", "^Lava Hot$" = "5")),
         ord_3 = replace_na(ord_3, mode_ord3),
         ord_4 = replace_na(ord_4, mode_ord4),
         ord_5 = replace_na(ord_5, "Zx")) %>%
  mutate(ord_3 = map_dbl(ord_3, ~strtoi(charToRaw(.), 16L)),
         ord_4 = map_dbl(ord_4, ~strtoi(charToRaw(.), 16L)),
         ord_5 = map_dbl(ord_5, ~strtoi(charToRaw(.), 16L) %>%
                           reduce(`+`))) %>%
  mutate_at(vars(matches("ord|day|month")), as.numeric) %>%
  mutate_at(vars(-matches("ord|day|month")), as.factor) %>%
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))

#setting our recipe
preproc_recipe <- recipe(target ~ .,
         data = train %>%
                  select(-id)) %>%
  step_meanimpute(matches("ord|day|month")) %>%
  step_other(all_nominal(), threshold = .05) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(matches("ord|day|month")) %>%
  step_scale(matches("ord|day|month")) %>%
  step_ns(starts_with("day"), deg_free = 3) %>%
  step_ns(starts_with("month"), deg_free = 3) %>%
  step_interact(terms = ~matches("bin|ord|day|month"):matches("ord|bin|day|month"))



#bootstrapping cv
train_cv <- train %>%
  vfold_cv(v = 5, strata = "target")


glmnet_mod <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

#training the model
glmnet_tuned_results <- tune_grid(
  preproc_recipe,
  model = glmnet_mod,
  resamples = train_cv,
  grid = 10,
  metrics = metric_set(roc_auc)
) 

#extracting the best model parameters
##can go back and rerun with 1se
best_glmnet_params <- glmnet_tuned_results %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))

#finalizing recipe
final_rec <- finalize_recipe(preproc_recipe, best_glmnet_params)

#getting final design matrix
trn <- final_rec %>%
  prep() %>%
  juice()

#finalizing the model & fitting
glmnet_final_mod <- glmnet_mod %>%
  finalize_model(parameters = best_glmnet_params) %>%
  fit(target ~ ., data = trn)

#cleaning up env a bit
rm(train, train_cv)
gc()

#reading in and manipulating test data
test <- vroom(here::here("Categorical Feature Challenge 2/Data/test.csv")) %>%
  mutate(ord_1 = str_replace_all(ord_1, c("Novice" = "0", "Contributor" = "1", "Expert" = "2",
                                          "Master" = "3", "Grandmaster" = "4")),
         ord_2 = str_replace_all(ord_2, c("Freezing" = "0", "Cold" = "1", "Warm" = "2", "^Hot$" = "3", 
                                          "^Boiling Hot$" = "4", "^Lava Hot$" = "5")),
         ord_3 = replace_na(ord_3, mode_ord3),
         ord_4 = replace_na(ord_4, mode_ord4),
         ord_5 = replace_na(ord_5, "Zx")) %>%
  mutate(ord_3 = map_dbl(ord_3, ~strtoi(charToRaw(.), 16L)),
         ord_4 = map_dbl(ord_4, ~strtoi(charToRaw(.), 16L)),
         ord_5 = map_dbl(ord_5, ~strtoi(charToRaw(.), 16L) %>%
                           reduce(`+`))) %>%
  mutate_at(vars(matches("ord|day|month")), as.numeric) %>%
  mutate_at(vars(-matches("ord|day|month")), as.factor) %>%
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))

test_prep <- final_rec %>%
  prep() %>%
  bake(test)

#predicting
glmnet_prob <- predict(glmnet_final_mod, new_data = test_prep, type = "prob")

#
glmnet_preds <- tibble(
  id = seq(600000, 999999),
  target = glmnet_prob$.pred_1
)

write_csv(glmnet_preds, here::here("Categorical Feature Challenge 2/Submissions/glmnet_update_3_24.csv"))