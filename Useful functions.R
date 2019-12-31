#to plot missingness in data
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