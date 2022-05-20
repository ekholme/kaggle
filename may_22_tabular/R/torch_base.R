library(tidyverse)
library(tidymodels)
library(torch)
library(luz)

# read in data
trn <- read_csv(here::here("may_22_tabular/data/train.csv"))
tst <- read_csv(here::here("may_22_tabular/data/test.csv"))

# set up resample
set.seed(0408)
splits <- initial_split(trn, prop = .8)
trn <- training(splits)
val <- testing(splits) # this will be more useful later as I experiment, but not really going to do anything with it here

# set up recipe
rec <- recipe(target ~ ., data = trn) |>
    update_role(id, new_role = "id") |>
    step_mutate(f_27 = substring(f_27, 1, 3)) |>
    step_other(f_27, threshold = 20000) |>
    step_dummy(f_27) |>
    step_log(f_07:f_18, offset = 1) |>
    step_normalize(all_numeric_predictors())

# apply recipe to data
trn <- rec |>
    prep() |>
    bake(new_data = NULL)

# and let's do the same for our validation data
val <- rec |>
    prep() |>
    bake(new_data = val)

# create a function to assist with loading data
may_tabular_dataset <- dataset(
    name = "may_tabular_dataset",

    initialize = function(df) {
        self$x <- df |>
            select(-c("id", "target")) |>
            as.matrix() |>
            torch_tensor()
        
        self$y <- torch_tensor(df$target)
    },

    .getitem = function(i) {
        x <- self$x[i, ]
        y <- self$y[i]

        list(x, y)
    },

    .length = function() {
        self$y$size()[[1]]
    }
)

# create actual datasets
trn_tensor <- may_tabular_dataset(trn)
val_tensor <- may_tabular_dataset(val)

# and create our dataloaders
trn_dl <- dataloader(trn_tensor, batch_size = 256, shuffle = TRUE)
val_dl <- dataloader(val_tensor, batch_size = 256, shuffle = TRUE)

# specify the main module
net <- nn_module(
    "m_tab_net",

    initialize = function() {

        self$fc1 <- nn_linear(47, 512)
            self$fc2 <- nn_linear(512, 256)
            self$fc3 <- nn_linear(256, 128)
            self$fc4 <- nn_linear(128, 64)
            self$output <- nn_linear(64, 1)
    },

    forward = function(x) {
        x <- self$fc1(x)
        x <- nnf_relu(x)
        x <- self$fc2(x)
        x <- nnf_relu(x)
        x <- self$fc3(x)
        x <- nnf_relu(x)
        x <- self$fc4(x)
        x <- self$output(x)
        x <- nnf_sigmoid()
        x
    }
)

# training the model
# see here for more: https://cran.r-project.org/web/packages/luz/vignettes/get-started.html

#ok so this is starting at least -- try running at home
nn_fitted <- net |>
    setup(
        loss = nn_cross_entropy_loss(),
        optimizer = optim_adam
    ) |>
    set_opt_hparams(lr = .003) |>
    fit(trn_dl, epochs = 3, valid_data = val_dl)
