library(tidyverse)
library(reticulate)
library(lubridate)
setwd("~/Projects/VegetationMapPaper/")

reticulate::source_python("scripts/plot_phenology.py")

df <- sample_timeseries(n = 10L)

change_names <- function(nums) {
  case_when(
    nums == 0 ~ "Dwarf Bamboo",
    nums == 1 ~ "Other Vegetation",
    nums == 2 ~ "No Vegetation",
    nums == 3 ~ "Rowans",
    nums == 4 ~ "Maple",
    nums == 5 ~ "Alder",
    nums == 6 ~ "Dwarf Pine"
  )
}

df %>%
  as_tibble() %>%
  mutate(
    vegetation = change_names(vegetation),
    date = ymd(str_extract(path, "\\d{8}"))
  ) %>%
  select(-path) %>%
  pivot_longer(cols = c("R", "G", "B"), names_to = "band", values_to = "value") %>% 
  write_csv("results/phenology.csv")

df <- read_csv("results/phenology.csv") %>%
  mutate(vegetation = ifelse(vegetation == "Golden Birch", "Maple", vegetation)) %>%
  mutate(date = str_c(str_pad(month(date), 2, pad="0"), "/", day(date)))

p <- df %>%
  mutate(band = factor(band, levels = c("R", "G", "B"))) %>%
  mutate(id = str_c(index, band)) %>%
  ggplot() +
  geom_line(aes(x = date, y = value, colour = band, group = id)) +
  facet_wrap(~ vegetation, nrow = 3) +
  labs(
    x = "Date",
    y = "Pixel Value",
    colour = "Band"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 18),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 60, hjust = 1),
    plot.background = element_rect(fill = "white"),
    panel.spacing = grid::unit(2, "lines")
  )

p

ggsave("results/phenology.png", plot = p, width = 10, height = 8)
