
#Running initial glmnet model -- will refine later

library(tidyverse)
library(lubridate)
library(vroom)
library(tidymodels)

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

train <- vroom(here::here("Categorical Feature Challenge 2/Data/train.csv")) %>%
  mutate_all(as.factor) %>%
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))
  
test <- vroom(here::here("Categorical Feature Challenge 2/Data/test.csv")) %>%
  mutate_all(as.factor) %>%
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))

#setting our recipe
preproc_recipe <- train %>%
  select(-id) %>%
  recipe(target ~ .) %>%
  step_other(all_nominal(), threshold = .03) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  prep()

train_prep <- juice(preproc_recipe)

test_prep <- preproc_recipe %>%
  bake(test)

#setting up 10 fold cv stratified by target
train_cv <- train_prep %>%
  vfold_cv(v = 10, strata = "target")

  
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
  metrics = metric_set(roc_auc), #we'll just use mae here, although there are other metrics we can choose for regression
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

write_csv(glmnet_preds, here::here("Categorical Feature Challenge 2/Submissions/glmnet_initial.csv"))
##To consider for later
#1. How to handle missing values other than just assigning them a separate factor level.
#2. Look into other potentially useful recipe steps
#3. Using vfold_cv right now, but can try bootstrapping later
#4. Look into bayesian approach for hyperparameters?