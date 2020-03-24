
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
  mutate_at(vars(matches("ord")), as.numeric) %>%
  mutate_at(vars(-matches("ord")), as.factor) %>%
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))


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
  mutate_at(vars(matches("ord")), as.numeric) %>%
  mutate_at(vars(-matches("ord")), as.factor) %>%
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))

#setting our recipe
preproc_recipe <- train %>%
  select(-id) %>%
  recipe(target ~ .) %>%
  step_meanimpute(all_numeric()) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_other(all_nominal(), threshold = .03) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_interact(terms = ~starts_with("bin"):starts_with("ord")) %>%
  prep()

train_prep <- juice(preproc_recipe)

test_prep <- preproc_recipe %>%
  bake(test)

#bootstrapping cv
train_cv <- train_prep %>%
  bootstraps(times = 10, strata = "target")

  
glmnet_mod <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

#establishing a tuning grid for the glmnet hyperparameters
glmnet_hypers <- parameters(penalty(), mixture()) %>%
  grid_max_entropy(size = 20)

#training the model
glmnet_tuned_results <- tune_grid(
  formula = target ~ .,
  model = glmnet_mod,
  resamples = train_cv,
  grid = glmnet_hypers,
  metrics = metric_set(roc_auc),
  control = control_grid()
)

#extracting the best model parameters
best_glmnet_params <- glmnet_tuned_results %>%
  select_best()

#finalizing the model
glmnet_final_mod <- glmnet_mod %>%
  finalize_model(parameters = best_glmnet_params) %>%
  fit(target ~ ., data = train_prep)

#predicting
glmnet_prob <- predict(glmnet_final_mod, new_data = test_prep, type = "prob")

#
glmnet_preds <- tibble(
  id = seq(600000, 999999),
  target = glmnet_prob$.pred_1
)

write_csv(glmnet_preds, here::here("Categorical Feature Challenge 2/Submissions/glmnet_update_3_21.csv"))

###to do
#1. Encode ordinal data differently
#2. Mean impute ordinaldata
#3. Give explicit NA for binary and nominal data
#4. Look for interactions?


##To consider for later
#1. How to handle missing values other than just assigning them a separate factor level.
#2. Look into other potentially useful recipe steps
#3. Using vfold_cv right now, but can try bootstrapping later
#4. Look into bayesian approach for hyperparameters?