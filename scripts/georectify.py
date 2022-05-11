# Loading requirements
from alproj.surface import create_db, crop
from alproj.project import sim_image, reverse_proj
from alproj.gcp import akaze_match, set_gcp
from alproj.optimize import CMAOptimizer
import sqlite3
import numpy as np
import rasterio

res = 1.0 # Resolution in m
aerial = rasterio.open("airborne.tif")
dsm = rasterio.open("dsm.tif")
out_path = "pointcloud.db" # Output database

create_db(aerial, dsm, out_path) # This takes some minutes