library(tidyverse)
library(tidymodels)
library(splines)

trn <- read_csv(here::here("aug_22_tabular/data/train.csv")) |>
    mutate(failure = as.character(failure))

tst <- read_csv(here::here("aug_22_tabular/data/test.csv"))

# Set Up Initial Split --------------------

splits <- initial_split(trn, prop = .8)

trn1 <- training(splits)
val <- testing(splits)

# Set Up Initial Recipe -------------------

rec <- recipe(failure ~ ., data = trn) |>
    update_role(id, new_role = "id") |>
    step_impute_mode(all_nominal_predictors()) |>
    step_impute_median(all_numeric_predictors()) |>
    step_YeoJohnson(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors())

# Set Up Initial Model ---------------------

#will tune this later
#right now running a pure lasso model
spec <- logistic_reg(penalty = .01) |>
    set_engine("glmnet")


# Create Workflow ----------------------

wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

# Fit ---------------------------------

wf_fit <- fit(wf, data = trn)


# Examine Features --------------------

feat_pull <- wf_fit |>
    tidy()

# ok so loading seems to be the only thing that matters
# let's try a spline model

only_loading <- select(trn, failure, loading)

new_rec <- recipe(failure ~ ., data = only_loading) |>
    step_impute_median(loading) |>
    step_YeoJohnson(loading) |>
    step_ns(loading, deg_free = tune())

spline_wf <- workflow() |>
    add_recipe(new_rec) |>
    add_model(spec)

df_grid <- grid_regular(deg_free(c(2, 6)), levels = 5)

cvs <- vfold_cv(only_loading, v = 5)

# Tune DF -----------------------------

spline_res <- spline_wf |>
    tune_grid(
        resamples = cvs,
        grid = df_grid
    )

# Examine Fits -----------------------

spline_res |>
    show_best("accuracy")


best_fits <- spline_res |>
    select_best("accuracy")

# finalize wf
final_wf <- spline_wf |>
    finalize_workflow(best_fits)

final_fit <- final_wf |>
    fit(data = only_loading)

# predict
preds <- predict(final_fit, new_data = data.frame(loading = tst$loading), type = "prob")

#make submission
sub <- tibble(
    id = tst$id,
    failure = preds$.pred_1
)

write_csv(sub, here::here("aug_22_tabular/submissions/lasso_spline.csv"))
