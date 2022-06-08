library(tidyverse)
library(stars)
library(sf)

change_names <- function(nums) {
    case_when(
        nums == 1 ~ "Dwarf Bamboo",
        nums == 2 ~ "No Vegetation",
        nums == 3 ~ "Other Vegetation",
        nums == 4 ~ "Sky",
        nums == 5 ~ "Dwarf Pine"
    )
}

vege2012 <- read_stars("data/res2012_cnn_9x9.tiff")
vege2020 <- read_stars("data/res2020_cnn_9x9.tiff")
dem <- read_stars("data/tateyamadem_small.tiff") %>%
    st_warp(vege2012)

# Get raster resolutions
res <- st_dimensions(vege2012) %>%
    as_tibble() %>%
    pull(delta) %>%
    as.numeric() %>%
    abs()

df <- c(vege2012, vege2020, dem) %>%
    as_tibble() %>%
    rename(
        v2012 = res2012_cnn_9x9.tiff,
        v2020 = res2020_cnn_9x9.tiff,
        elevation = tateyamadem_small.tiff
        ) %>%
    pivot_longer(
        cols = c("v2012", "v2020"),
        names_to = "year",
        values_to = "vegetation",
        names_prefix = "v"
        ) %>%
    filter(is.na(vegetation) == FALSE) %>%
    mutate(vegetation = change_names(vegetation)) %>%
    filter(vegetation != "Sky")

# Vegetation histogram along elevation
breaks <- seq(2350, 3000, 50)

df %>%
    group_by(vegetation, year) %>%
    ggplot(aes(x = elevation, fill = year)) +
    geom_histogram(
        aes(y = after_stat(count * res[1] * res[2])),
        breaks = breaks,
        alpha = 0.4,
        position = "identity"
        ) +
    facet_wrap(~vegetation) +
    ylab(expression(paste ("Area (", {mm^2}, ")", sep = ""))) +
    xlab("Elevation") +
    theme_minimal()

# Elevation-axis movement of the vegetation
df %>%
    group_by(vegetation, year) %>%
    summarise(
        mean_elevation = mean(elevation)
    )

df %>%
    group_by(vegetation, year) %>%
    summarise(
        elevation_2012 = mean(elevation[year == "2012"]),
        elevation_2020 = mean(elevation[year == "2020"])
    ) %>%
    select(-year) %>%
    fill(elevation_2012, .direction = "down") %>%
    fill(elevation_2020, .direction = "up") %>%
    distinct() %>%
    mutate(movement = elevation_2020 - elevation_2012)

# Increase, decrease in area
df %>%
    group_by(vegetation, year) %>%
    tally() %>%
    mutate(area = n * res[1] * res[2]) %>%
    mutate(
        area_2012 = area[year == "2012"],
        area_2020 = area[year == "2020"]
        ) %>%
    select(-c(year, n, area)) %>%
    distinct() %>%
    mutate(diff = area_2020 - area_2012)
# 