library(stars)
library(tidyverse)
library(ggspatial)

setwd("~/Projects/VegetationMapPaper/")

cmap = c(
  "#2aa198", # Dwarf pine
  "#859900", # Dwarf bamboo
  "#dc322f", # Rowans
  "#b58900", # Maple
  "#6c71c4", # Montane Alder
  "#eee8d5", # Other vegetation
  "#c0c0c0" # No vegetation
)

vegetation_levels <- c(
  "Dwarf Pine",
  "Dwarf Bamboo",
  "Rowans",
  "Maple",
  "Montane Alder",
  "Other Vegetation",
  "No Vegetation"
)


ras <- read_stars("results/rnn.tiff") %>%
  rename(vegetation = rnn.tiff) %>%
  mutate(vegetation = as.integer(vegetation)) %>%
  mutate(
    vegetation = case_when(
      vegetation == 1 ~ "Dwarf Bamboo",
      vegetation == 2 ~ "Other Vegetation",
      vegetation == 3 ~ "No Vegetation",
      vegetation == 4 ~ "Rowans",
      vegetation == 5 ~ "Maple",
      vegetation == 6 ~ "Montane Alder",
      vegetation == 7 ~ "Dwarf Pine"
    )
  ) %>%
  mutate(vegetation = factor(vegetation, levels = vegetation_levels))

dem <- read_stars("data/mrd_dem_1m.tiff") %>%
  st_crop(ras) %>%
  rename(Elevation = mrd_dem_1m.tiff)

write_stars(dem, "data/dem_small.tiff")
dem <- read_stars("data/dem_small.tiff")

cont <- st_contour(dem, contour_lines = T)

p1 <- ggplot() +
  geom_sf(data = cont, size = 0.1) +
  geom_stars(
    mapping = aes(x = x, y = y, fill = vegetation), 
    data = ras
  ) +
  scale_fill_manual(values = cmap, na.value = "transparent") +
  labs(x = "Longitude", y = "Latitude", fill = "Vegetation") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 18),
    text = element_text(size = 20),
    plot.background = element_rect(fill = "white")
  )

ggsave("results/vegemap.png", p1, width = 12, height = 8)

library(magick)
rnn <- magick::image_read("results/rnn.png")
vegemap <- magick::image_read("results/vegemap.png")
img <- c(rnn, vegemap)
imgs <- image_append(image_scale(img, "x2000"))

image_write(imgs, "paper/paper_files/figures/vege.png")
