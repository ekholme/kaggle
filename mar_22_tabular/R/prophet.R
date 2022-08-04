library(tidyverse)
library(prophet)

trn <- read_csv(here::here("mar_22_tabular/data/train.csv"))
tst <- read_csv(here::here("mar_22_tabular/data/test.csv"))

make_roads <- function(x) {
    x %>%
        mutate(road = paste0(x, "_", y, "_", direction)) %>%
        select(-c("x", "y", "direction"))
}

cf <- function(v) {
    n <- length(v)
    sigma2 <- var(v)
    med <- median(v)

    ss <- sum((v - med)^2)

    ret <- 1 - ((n - 3) * sigma2 / ss)

    ret
}

js_shrink <- function(x, grp, var) {
    lookup <- x %>%
        select({{ grp }}, {{ var }}) %>%
        group_by({{ grp }}) %>%
        nest() %>%
        mutate(cf = map_dbl(.data$data, ~ cf(.x[[1]]))) %>%
        select({{ grp }}, cf)
    
    x %>%
        left_join(lookup) %>%
        group_by({{ grp }}) %>%
        mutate(med = median({{ var }}), {{ var }} := med + cf * ({{ var }} - med)) %>%
        select(-c("cf", "med"))
}

## RESUME HERE -- although I think the shrinkage is too much

trn <- trn %>%
    make_roads()

# testing out JS stuff
rd1 <- trn %>%
    filter(road == unique(trn$road)[1])

n <- length(rd1$congestion)
sigma2 <- var(rd1$congestion)
med <- median(rd1$congestion)

ss <- sum((rd1$congestion - med)^2)

cf <- 1 - ((n - 3) * sigma2 / ss)

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
