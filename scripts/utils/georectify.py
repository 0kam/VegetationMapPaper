# Loading requirements
from alproj.surface import create_db, crop
from alproj.project import sim_image, reverse_proj
from alproj.gcp import akaze_match, set_gcp
from alproj.optimize import CMAOptimizer
import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import rasterio
import cv2

res = 1.0 # Resolution in m
aerial = rasterio.open("data/tateyama2.tiff")
dsm = rasterio.open("data/mrd_dem_1m.tiff")
out_path = "data/pointcloud.db" # Output database

# create_db(aerial, dsm, out_path) # This takes some minutes

params = {"x":732731,"y":4051171, "z":2458, "fov":75, "pan":95, "tilt":0, "roll":0,\
          "a1":1, "a2":1, "k1":0, "k2":0, "k3":0, "k4":0, "k5":0, "k6":0, \
          "p1":0, "p2":0, "s1":0, "s2":0, "s3":0, "s4":0, \
          "w":5616, "h":3744, "cx":5616/2, "cy":3744/2}

conn = sqlite3.connect("data/pointcloud.db")

distance = 3000 # The radius of the fan shape
chunksize = 1000000

vert, col, ind = crop(conn, params, distance, chunksize) # This takes some minutes.

sim = sim_image(vert, col, ind, params)
cv2.imwrite("data/initial.png", sim)

df = reverse_proj(sim, vert, ind, params)

path_org = "data_source/aligned/2012/mrd_085_eos_vis_20121021_1200.png"
path_sim = "data/initial.png"

match, plot = akaze_match(path_org, path_sim, ransac_th=100, plot_result=True)
cv2.imwrite("data/matched.png", plot)
gcps = set_gcp(match, df)
gcps.to_csv("data/gcp.csv")

obj_points = gcps[["x","y","z"]] # Object points in a geographic coordinate system
img_points = gcps[["u","v"]] # Image points in an image coordinate system 
params_init = params # Initial parameters
target_params = ["fov", "pan", "tilt", "roll", "a1", "a2", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"] # Parameters to be optimized

cma_optimizer = CMAOptimizer(obj_points, img_points, params_init) # Create an optimizer instance.
cma_optimizer.set_target(target_params)

params_optim, error = cma_optimizer.optimize(generation = 300, sigma = 1.0, population_size = 30) # Executing optimization
print("Alignment mean square error: {} pix".format(error))

#vert, col, ind = crop(conn, params_optim, 3000, 1000000)
sim2 = sim_image(vert, col, ind, params_optim)
cv2.imwrite("data/optimized.png", sim2)

params_optim["error"] = error
import json
with open('data/params_optim.json', 'w') as f:
    json.dump(params_optim, f, indent=4)

# Apply georectification
## Mask images
# m1 = cv2.imread("data_source/sky_mask_2015.png")[:,:,0].astype(float)
# m2 = cv2.imread("data_source/ground_mask.png")[:,:,0].astype(float)
# m3 = np.load("data_source/mask.npy")

# mask = m1 * m2 * m3
# mask[mask > 0] = 1
# np.save("data_source/mask_all.npy", mask)

# mask = np.load("data_source/mask_all.npy")
# img = cv2.imread("data_source/aligned/multi_days/mrd_085_eos_vis_20150926_1205.png")
# img = img * mask[:,:,None].astype(np.uint8)
# cv2.imwrite("results/image.png", img)

# for f in ["results/rnn.png", "results/svm.png"]:
#     img = cv2.imread(f)
#     img = img * mask[:,:,None].astype(np.uint8)
#     cv2.imwrite(f, img)

# for f in ["results/rnn.npy", "results/svm.npy"]:
#     img = np.load(f)
#     # img = img + 1 # only run once
#     img = img * mask
#     np.save(f, img.astype(int))

# Reverse projection
import json
with open('data/params_optim.json', 'r') as f:
    params_optim = json.load(f)

conn = sqlite3.connect("data/pointcloud.db")
vert, col, ind = crop(conn, params_optim, 3000, 1000000)

original = cv2.imread("results/image.png")
georectified = reverse_proj(original, vert, ind, params_optim, chnames=["B", "G", "R"])
georectified = georectified[georectified["R"] > 0]
georectified = georectified[georectified["G"] > 0]
georectified = georectified[georectified["B"] > 0]
georectified.to_csv("results/georectified.csv", index=False)

targets = ["results/rnn.npy", "results/svm.png"]

for target in targets:
    t = np.load(target)
    t = t[:, :, np.newaxis]
    georectified = reverse_proj(t, vert, ind, params_optim, chnames=["vegetation"])
    georectified["vegetation"] = georectified["vegetation"].astype(int)
    georectified = georectified[georectified["vegetation"]!=0]
    out = target.replace("npy", "csv")
    georectified.to_csv(out, index=False)