library(tidyverse)
library(tidymodels)
library(hms)
library(lubridate)
library(mgcv)
library(xgboost)
library(finetune)

trn <- read_csv(here::here("mar_22_tabular/data/train.csv"))
tst <- read_csv(here::here("mar_22_tabular/data/test.csv"))

#function to extract datetime preds
extract_dttm <- function(x) {
    x %>%
        mutate(
            tm = as.numeric(as_hms(time)),
            wkday = as.character(wday(time, label = TRUE)),
            mo = as.character(month(time, label = TRUE)),
            wkday = if_else(
                wkday %in% c("Mon", "Tue", "Wed", "Thu", "Fri"),
                "Weekday", 
                wkday
            )
        )
}

# function to convert the x/y coordinates into discrete data, because I don't think they're actually continuous
convert_xy <- function(.x) {
    .x %>%
        mutate(x = letters[.x$x + 1], y = LETTERS[.x$y + 1])
}

remove_feats <- function(x, vars) {
    x %>%
        select(-vars)
}

#wrapper for the previous two functions
transform_feats <- function(.x, rmv) {
    .x %>%
        extract_dttm() %>%
        convert_xy() %>%
        remove_feats(vars = rmv)
}

rmv <- c("time", "row_id")

trn <- trn %>%
    transform_feats(rmv = rmv)

row_ids <- tst$row_id

tst <- tst %>%
    transform_feats(rmv = rmv)

trn_folds <- vfold_cv(trn, v = 5)

# Recipe ----------------------------------------------

rec <- recipe(congestion ~ ., data = trn) %>%
    step_dummy(x, y, direction, wkday, mo) %>%
    step_ns(tm, deg_free = 10) %>%
    step_interact(terms = ~starts_with("tm"):starts_with("wkday"))

# pr <- rec %>%
#     prep() %>%
#     bake(new_data = NULL)

# Model Spec ------------------------------------------

xgb_spec <- boost_tree(
    trees = 2000,
    tree_depth = 2,
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    mtry = tune(),
    learn_rate = tune()
) %>%
    set_engine("xgboost") %>%
        set_mode("regression")
    
# Parameter grid -------------------------------------

xgb_grid <- grid_latin_hypercube(
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), trn),
    learn_rate(),
    size = 10
)

params <- parameters(list(
    min_n = min_n(),
    loss_reduction = loss_reduction(),
    sample_size = sample_prop(),
    mtry = finalize(mtry(), trn),
    learn_rate = learn_rate()#,
    #trees = 2000,
    #tree_depth = 2
))

# Workflow --------------------------------------------

wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(xgb_spec)

# Fit -------------------------------------------------

xgb_res <- tune_sim_anneal(
    wf,
    resamples = trn_folds,
    iter = 10,
    param_info = params
)

# looking at best versions
best_params <- select_best(xgb_res, "rmse")

#finalize fit
final_xgb <- finalize_workflow(
    wf,
    best_params
)

final_xgb_fit <- fit(final_xgb, data = trn)

# Predict ---------------------------------------------

preds <- simplify(predict(final_xgb_fit, tst))

# Write Sub -------------------------------------------

sub <- tibble(
    row_id = row_ids,
    congestion = preds
)

write_csv(sub, file = here::here("mar_22_tabular/submissions/xgb_preds.csv"))
