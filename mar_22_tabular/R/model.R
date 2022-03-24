library(tidyverse)
library(tidymodels)

trn <- read_csv(here::here("mar_22_tabular/data/train.csv"))
tst <- read_csv(here::here("mar_22_tabular/data/test.csv"))

#function to extract datetime preds
extract_dttm <- function(x) {
    x %>%
        mutate(
            tm = as_hms(time),
            wkday = as.character(wday(time, label = TRUE)),
            mo = month(time, label = TRUE),
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

#wrapper for the previous two functions
transform_feats <- function(.x) {
    .x %>%
        extract_dttm() %>%
        convert_xy()
}

trn <- trn %>%
    transform_feats()

tst <- tst %>%
    transform_feats()

# Basic Linear Model ----------------------------------

mod <- lm(congestion ~ ., data = trn)

preds <- predict(mod, newdata = tst)

# Write Sub -------------------------------------------

sub <- tibble(
    row_id = tst$row_id,
    congestion = preds
)

write_csv(sub, file = here::here("submissions/lm_sub.csv"))