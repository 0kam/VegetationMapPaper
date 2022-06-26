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
              "ダケカンバ" = "Golden Birch",
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
  "Golden Birch",
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
        write_csv(out) #%>%
        # select(-var) %>%
        # kable(booktabs = TRUE,
        #     caption = "Cross validation result \\label{tab:vegetation_cv}") %>%
        # kable_styling() %>%
        # collapse_rows(columns = 1) %>%
        # footnote(general = "Mean (SD) of 5-Fold Cross-Varidation")
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
    mutate(date = ifelse(is.na(date), str_c("Multidays", kernel_size), date)) %>%
    mutate(date = ifelse(str_detect(path, "rnn"), "Multidays RNN 1x1", date)) %>%
    ggplot(aes(x = date, y = value)) +
    geom_boxplot() +
    ylab("F1 score") +
    facet_wrap(~ vegetation) +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 18),
        text = element_text(size = 20),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.background = element_rect(fill = "white")
    )

ggsave(file = "results/cv.png", plot = p1, width = 18, height = 9)

p2 <- df %>%
    filter(vegetation != "Macro Average") %>%
    filter(vegetation != "Weighted Average") %>%
    filter(multidays == FALSE) %>%
    filter(kernel_size == "1x1") %>%
    ggplot(aes(x = as.factor(date), y = value)) +
    geom_boxplot() +
    ylab("F1 score") +
    xlab("Date") +
    facet_wrap(~ vegetation) +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, hjust = 1),
        text = element_text(size = 20),
        plot.background = element_rect(fill = "white")
    )

ggsave(file = "results/cv_singleday.png", plot = p2)

p3 <- df %>%
    filter(vegetation != "Macro Average") %>%
    filter(vegetation != "Weighted Average") %>%
    filter(multidays == FALSE) %>%
    filter(date == "2015-10-10") %>%
    ggplot(aes(x = kernel_size, y = value)) +
    geom_boxplot() +
    ylab("F1 score") +
    xlab("Date") +
    facet_wrap(~ vegetation) +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, hjust = 1),
        text = element_text(size = 20),
        plot.background = element_rect(fill = "white")
    )

ggsave(file = "results/cv_singleday_10_10.png", plot = p3)

df2 <- df %>%
    filter(vegetation != "Macro Average") %>%
    filter(vegetation != "Weighted Average") %>%
    filter(multidays == FALSE) %>%
    filter(date == "2015-10-10")

df3 <- df %>%
    filter(vegetation != "Macro Average") %>%
    filter(vegetation != "Weighted Average") %>%
    filter(multidays == TRUE) %>%
    filter(kernel_size %in% c("1x1", "5x5")) %>%
    bind_rows(df2) %>%
    mutate(multidays = ifelse(multidays, "Multidays", "Singleday"))


p4 <- df3 %>%
    ggplot(aes(x = multidays, y = value)) +
    geom_boxplot() +
    ylab("F1 score") +
    xlab("Multidays") +
    facet_grid(kernel_size ~ vegetation) +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, hjust = 1),
        text = element_text(size = 20),
        plot.background = element_rect(fill = "white")
    )

ggsave(file = "results/cv_singleday_vs_multidays.png", plot = p4, width = 12, height = 8)

p5 <- df3 %>%
    ggplot(aes(x = kernel_size, y = value)) +
    geom_boxplot() +
    ylab("F1 score") +
    xlab("kernel_size") +
    facet_grid(multidays ~ vegetation) +
    theme_minimal() +
    theme(
        axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, hjust = 1),
        text = element_text(size = 20),
        plot.background = element_rect(fill = "white")
    )

ggsave(file = "results/cv_kernel_size.png", plot = p5, width = 12, height = 8)
