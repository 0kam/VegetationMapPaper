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

vege2012 <- read_stars("results/cnn_lstm5x5_cv_5_ep_200/res2012_cnn_5x5.tiff")
vege2021 <- read_stars("results/cnn_lstm5x5_cv_5_ep_200/res2021_cnn_5x5.tiff")
dem <- read_stars("data/mrd_dem_1m.tiff") %>%
    st_warp(vege2012)

# Get raster resolutions
res <- st_dimensions(vege2012) %>%
    as_tibble() %>%
    pull(delta) %>%
    as.numeric() %>%
    abs()

df <- c(vege2012, vege2021, dem) %>%
    as_tibble() %>%
    rename(
        v2012 = res2012_cnn_5x5.tiff,
        v2021 = res2021_cnn_5x5.tiff,
        elevation = mrd_dem_1m.tiff
        ) %>%
    pivot_longer(
        cols = c("v2012", "v2021"),
        names_to = "year",
        values_to = "vegetation",
        names_prefix = "v"
        ) %>%
    filter(is.na(vegetation) == FALSE) %>%
    mutate(vegetation = change_names(vegetation)) %>%
    filter(vegetation != "Sky")

# Vegetation histogram along elevation
breaks <- seq(2350, 3000, 50)

p <- df %>%
    group_by(vegetation, year) %>%
    ggplot(aes(x = elevation, fill = year)) +
    geom_histogram(
        aes(y = after_stat(count * res[1] * res[2])),
        breaks = breaks,
        alpha = 0.3,
        position = "identity"
        ) +
    facet_wrap(~vegetation) +
    ylab(expression(paste ("Area (", {mm^2}, ")", sep = ""))) +
    xlab("Elevation") +
    theme_minimal() +
    scale_fill_hue(direction = -1)

pg <- ggplot_build(p)

head(pg$data[[1]])


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
        elevation_2021 = mean(elevation[year == "2021"])
    ) %>%
    select(-year) %>%
    fill(elevation_2012, .direction = "down") %>%
    fill(elevation_2021, .direction = "up") %>%
    distinct() %>%
    mutate(movement = elevation_2021 - elevation_2012)

# Increase, decrease in area
df %>%
    group_by(vegetation, year) %>%
    tally() %>%
    mutate(area = n * res[1] * res[2]) %>%
    mutate(
        area_2012 = area[year == "2012"],
        area_2021 = area[year == "2021"]
        ) %>%
    select(-c(year, n, area)) %>%
    distinct() %>%
    mutate(diff = area_2021 - area_2012)