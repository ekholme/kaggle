library(tidyverse)
library(tidymodels)
library(hms)
library(lubridate)
library(mgcv)

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

# GAM -------------------------------------------------

mod <- lm(congestion ~ 1 + splines::ns(tm, df = 10) + x + y + direction + wkday + mo, data = trn)

# mod <- gam(congestion ~ s(as.numeric(tm), k = 10) + x + y + direction + wkday + mo, data = trn, na.action = "na.omit")

preds <- as.numeric(predict(mod, newdata = tst))

# Recipe ----------------------------------------------

rec <- recipe(congestion ~ ., data = trn) %>%
    step_dummy(x, y, direction, wkday, mo) %>%
    step_ns(tm, deg_free = 10) %>%
    step_interact(terms = ~starts_with("tm"):starts_with("wkday"))

pr <- rec %>%
    prep() %>%
    bake(new_data = NULL)

# Model Spec ------------------------------------------

glm_spec <- linear_reg(penalty = .01, mixture = 1) %>% # will want to tune these later
    set_engine("glmnet")

# Workflow --------------------------------------------

wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(glm_spec)

# Fit -------------------------------------------------

wf_fit <- fit(wf, data = trn)

# Predict ---------------------------------------------

preds <- simplify(predict(wf_fit, tst))

# Write Sub -------------------------------------------

sub <- tibble(
    row_id = row_ids,
    congestion = preds
)

write_csv(sub, file = here::here("mar_22_tabular/submissions/spline_interact_sub.csv"))
