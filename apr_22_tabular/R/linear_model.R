# Setup -------------------------------------
library(tidyverse)
library(tidymodels)

 trn <- read_csv(here::here("apr_22_tabular/data/train.csv"))
 trn_labs <- read_csv(here::here("apr_22_tabular/data/train_labels.csv"))
 tst <- read_csv(here::here("apr_22_tabular/data/test.csv"))

#functions to compute rci
std_err <- function(x) {
    sd(x) / sqrt(length(x))
}

# sort of a bastardized reliable change, but w/e
rci <- function(x1, x2) {
    se <- std_err(x1)

    sdiff <- sqrt(2 * se^2)

    v <- (x1 - x2) / sdiff

    abs(v) > 1.96
}

#wrapper to just pass in a df
rci_wrapper <- function(df) {

    df %>%
        pivot_longer(cols = starts_with("sensor"), names_to = "sensor", values_to = "val") %>%
        group_by(sequence, sensor) %>%
        arrange(sequence, sensor, step) %>%
        mutate(lag_val = lag(val), rci = rci(val, lag_val)) %>%
        summarize(prop_rc = sum(rci, na.rm = TRUE) / 59) %>%
        ungroup() %>%
        pivot_wider(names_from = sensor, values_from = prop_rc, names_glue = "{sensor}_prop_rc")
}

# Preprocess -------------------------------

trn_rc <- rci_wrapper(trn)
tst_rc <- rci_wrapper(tst)