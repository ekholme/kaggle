# code for a relatively straightforward linear model
library(tidyverse)
library(tidymodels)

# read in data
trn <- read_csv(here::here("may_22_tabular/data/train.csv")) |>
    mutate(target = as.factor(target))
tst <- read_csv(here::here("may_22_tabular/data/test.csv"))

# set up resample
set.seed(0408)
splits <- initial_split(trn, prop = .8)
trn <- training(splits)
val <- testing(splits) #this will be more useful later as I experiment, but not really going to do anything with it here

# Create Recipe ---------------------------

rec <- recipe(target ~ ., data = trn) |>
    update_role(id, new_role = "id") |>
    step_mutate(f_27 = substring(f_27, 1, 3)) |>
    step_other(f_27, threshold = 20000) |>
    step_log(f_07:f_18, offset = 1) |>
    step_normalize(all_numeric_predictors())

# Create Model Spec ----------------------

logreg_spec <- logistic_reg() |>
    set_engine("glm")

# Create Workflow ------------------------

logreg_wf <- workflow() |>
    add_model(logreg_spec) |>
    add_recipe(rec)

# Fit ------------------------------------

logreg_fit <- logreg_wf |>
    fit(data = trn)

# Predict --------------------------------

preds <- predict(logreg_fit, new_data = tst, type = "prob")

# Write Submission -----------------------

sub <- tibble(
    id = tst$id,
    target = preds$.pred_1
)

write_csv(sub, here::here("may_22_tabular/submissions/logreg_base.csv"))
