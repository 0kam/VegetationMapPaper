library(tidyverse)
library(stars)
library(sf)

change_names <- function(nums) {
    case_when(
        nums == 1 ~ "Dwarf Bamboo",
        nums == 2 ~ "Other Vegetation",
        nums == 3 ~ "No Vegetation",
        nums == 4 ~ "Rowans",
        nums == 5 ~ "Golden Birch",
        nums == 6 ~ "Montane Alder",
        nums == 7 ~ "Dwarf Pine"
    )
}

vege_levels <- c(
    "Dwarf Pine",
    "Dwarf Bamboo",
    "Rowans",
    "Golden Birch",
    "Montane Alder",
    "Other Vegetation",
    "No Vegetation"
)

rnn <- read_stars("results/rnn.tiff")
dem <- read_stars("data/mrd_dem_1m.tiff") %>%
    st_warp(rnn)

# Get raster resolutions
res <- st_dimensions(rnn) %>%
    as_tibble() %>%
    pull(delta) %>%
    as.numeric() %>%
    abs()

df <- c(rnn, dem) %>%
    as_tibble() %>%
    rename(
        vegetation = rnn.tiff,
        elevation = mrd_dem_1m.tiff
        ) %>%
    filter(is.na(vegetation) == FALSE) %>%
    mutate(vegetation = change_names(vegetation)) %>%
    mutate(vegetation = factor(vegetation, levels = vege_levels))

# Vegetation histogram along elevation
breaks <- seq(2350, 3000, 50)

p1 <- df %>%
    group_by(vegetation) %>%
    ggplot(aes(x = elevation)) +
    geom_histogram(
        aes(y = after_stat(count * res[1] * res[2])),
        breaks = breaks,
        position = "identity"
        ) +
    facet_wrap(~vegetation, scales = "free") +
    ylab(expression(paste ("Area (", {mm^2}, ")", sep = ""))) +
    xlab("Elevation") +
    theme_bw() +
    scale_fill_hue(direction = -1)

ggsave("results/vegetation_area_rnn.png", p1, width = 12, height = 8)
