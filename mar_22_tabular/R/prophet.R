library(tidyverse)
library(prophet)

trn <- read_csv(here::here("mar_22_tabular/data/train.csv"))
tst <- read_csv(here::here("mar_22_tabular/data/test.csv"))

make_roads <- function(x) {
    x %>%
        mutate(road = paste0(x, "_", y, "_", direction)) %>%
        select(-c("x", "y", "direction"))
}

trn <- trn %>%
    make_roads()

tst <- tst %>%
    make_roads()

#iterating
trn_nest <- trn %>%
    select(ds = time, y = congestion, road) %>%
    group_by(road) %>%
    nest(trn = c(ds, y))

tst_nest <- tst %>%
    select(ds = time, row_id, road) %>%
    group_by(road) %>%
    nest(tst = c(row_id, ds))

df <- trn_nest %>%
    left_join(tst_nest, by = "road") %>%
    mutate(
        ids = map(.data$tst, ~ pull(.x, var = "row_id")),
        mods = map(.data$trn, ~ prophet(.x))
    )

preds <- map2(df$tst, df$mods, ~predict(.y, .x %>% select(ds)))

p_df <- tibble(
    ids = df$ids,
    p = preds
)

aa <- unnest(p_df)

sub <- aa %>%
    select(row_id = ids, congestion = yhat) %>%
    arrange(row_id)

write_csv(sub, file = here::here("mar_22_tabular/submissions/prophet_baseline.csv"))
