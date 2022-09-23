library(sf)
library(tidyverse)
library(stars)
library(terra)
library(stringr)
setwd("~/projects/VegetationMapPaper/")

interpolate <- function(in_path, out_path, res, max_dist, fun=terra::modal) {
  print(str_c("Reading ", in_path, " ......"))
  points <- read_csv(
    in_path
  ) %>%
    st_as_sf(coords = c("x", "y")) %>%
    st_set_crs(6690) %>%
    mutate(z = as.integer(z)) %>%
    select(-c(u, v, z))
  
  print("Rasterizing point data ......")
  ras <- st_rasterize(points, dx = res, dy = res)
  rm(points)
  gc()
  
  ras <- ras %>% 
    as("Raster") %>%
    terra::rast()
  
  times <- ceiling(max_dist / res) - 1
  for (i in 1:times) {
    print(str_c("Interpolating ......", i, " of ", times, " iterations"))
    ras <- ras %>%
      terra::focal(3, fun, na.policy="only", na.rm=TRUE)
  }
  print(str_c("Saving file to ", out_path, " ......"))
  terra::writeRaster(ras, out_path, overwrite=TRUE)
  rm(ras)
  gc()
  print("Finished !")
}
  
files <- c("results/svm.csv", "results/rnn.csv", "results/teacher.csv")

for (file in files) {
  out_path <- stringr::str_replace(file, "csv", "tiff")
  interpolate(file, out_path, 0.5, 2.0)
}


file <- "results/georectified.csv"
out_path <- stringr::str_replace(file, "csv", "tiff")
interpolate(file, out_path, 0.5, 2.0, fun = mean)