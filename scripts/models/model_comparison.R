library(tidyverse)
library(magrittr)
library(kableExtra)
library(fs)
library(lubridate)

setwd("~/VegetationMapPaper")

translate_classes <- function(names) {
  res <- c()
  for (i in 1:length(names)) {
    res[i] <-
      switch (names[i],
              "ハイマツ" = "Dwarf Pine",
              "ササ" = "Dwarf Bamboo",
              "ナナカマド" = "Rowans",
              "ダケカンバ" = "Birch",
              "ミヤマハンノキ" = "Montane Alder",
              "その他植生" = "Other vegetation",
              "無植生" = "Non Vegetation",
              "macro avg" = "Macro Average",
              "weighted avg" = "Weighted Average"
      )
  }
  return(res)
}

vegetation_levels <- c(
  "Macro Average",
  "Weighted Average",
  "Dwarf Pine",
  "Dwarf Bamboo",
  "Rowans",
  "Birch",
  "Montane Alder",
  "Other vegetation",
  "Non Vegetation"
)

summarise_cv <- function(path) {
    out <- str_replace(path, ".csv", "_sum.csv")
    df <- read_csv(path,
               col_select = c(metrics, vegetation, value, fold))
    df %>%
        filter(metrics != "support") %>%
        mutate(vegetation = translate_classes(vegetation)) %>%
        mutate(vegetation = factor(vegetation, levels = vegetation_levels)) %>%
        group_by(., vegetation, metrics) %>%
        summarise(
          mean = sprintf("%1.3f", mean(value)),
          sd = sprintf("(%1.3f)", sd(value))
        ) %>%
        pivot_longer(cols = c(mean, sd), names_to = "var") %>%
        pivot_wider(names_from = metrics, values_from = value) %>%
        mutate(path = path) %T>%
        write_csv(out)
}

read_cv <- function(path) {
    read_csv(path, col_select = c(metrics, vegetation, value, fold)) %>%
        filter(metrics != "support") %>%
        mutate(vegetation = translate_classes(vegetation)) %>%
        mutate(vegetation = factor(vegetation, levels = vegetation_levels)) %>%
        mutate(path = path)
}

paths <- list.files(path = "runs/cv", pattern = ".*stratified_cv.csv",
    recursive = TRUE, full.names = TRUE)

df <- paths %>%
    map(read_cv) %>%
    reduce(bind_rows)

df <- df %>%
    mutate(path = str_remove_all(path, "runs/cv/|stratified_cv.csv")) %>%
    mutate(kernel_size = str_extract(path, ".x.")) %>%
    mutate(date = str_extract(path, "\\d{8}")) %>%
    mutate(date = ymd(date)) %>%
    filter(metrics == "f1-score") %>%
    mutate(multidays = is.na(date))

p1 <- df %>%
    filter(vegetation != "Macro Average") %>%
    filter(vegetation != "Weighted Average") %>%
    mutate(date = as.character(date)) %>%
    mutate(date = ifelse(is.na(date), "Multidays", date)) %>%
    mutate(date = ifelse(str_detect(path, "rnn"), "Multidays RNN", date)) %>%
    ggplot(aes(x = date, y = value)) +
    geom_boxplot() +
    ylab("F1 score") +
    xlab("Date") +
    facet_wrap(~ vegetation, nrow = 2) +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 18),
        text = element_text(size = 20),
        axis.text.x = element_text(angle = 60, hjust = 1),
        plot.background = element_rect(fill = "white"),
        panel.spacing = grid::unit(2, "lines")
    )

ggsave(file = "results/cv.png", plot = p1, width = 12, height = 6)

df %>% 
  filter(metrics == "f1-score") %>%
  filter(vegetation == "Macro Average") %>%
  group_by(path) %>%
  summarise(
    F1_mean = mean(value),
    F1_var = var(value)
    )
