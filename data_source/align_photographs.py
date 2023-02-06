import cv2
import numpy as np
from glob import glob
import os
import pandas as pd
from tqdm import tqdm

def homography(src_img, dst_img, flann = True, ransac_th=5):
    im1 = src_img
    im2 = dst_img
    # Akaze descripter
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(im1, None)
    kp2, des2 = akaze.detectAndCompute(im2, None)
    if flann:
        # Flann matcher
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                          table_number = 6,  
                          key_size = 12,     
                          multi_probe_level = 1) 
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else: 
        # Brute Force matcher
        matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k = 2)
    ratio = 0.75
    good = []
    # Filter matching points
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    pts1 = np.float32([ kp1[match.queryIdx].pt for match in good ])
    pts2 = np.float32([ kp2[match.trainIdx].pt for match in good ])
    pts1 = pts1.reshape(-1,1,2)
    pts2 = pts2.reshape(-1,1,2)

    # Find homography matrix
    hmat, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_th)
    pts1 = pts1[mask.astype('bool')]
    pts2 = pts2[mask.astype('bool')]
    
    # Culculate Root Mean Square Error
    pts1 = pts1.reshape(-1,2)
    pts2 = pts2.reshape(-1,2)
    pts1 = np.insert(pts1, 2, 1, axis = 1)
    pts1 = np.dot(hmat, pts1.T).T
    pts1[:,0] = pts1[:,0] / pts1[:,2]
    pts1[:,1] = pts1[:,1] / pts1[:,2]
    pts1 = pts1[:,0:2]
    rmse = np.mean(((pts1[:,0] - pts2[:,0])**2 + (pts1[:,1] - pts2[:,1])**2)**0.5) 
    
    # Homography Transformation 
    im1_h = cv2.warpPerspective(im1, hmat, (im2.shape[1], im2.shape[0]))

    # Return the transformed source image, homography matrix, rmse, and number of matched points
    return im1_h, hmat, rmse, len(pts1)

# Creating output directories if not found
for d in ["data_source/aligned/2015"]:
    if os.path.exists(d) == False:
        os.makedirs(d)

# Images to align
destination = "data_source/sky_mask_2015.png"
sources = sorted(glob("data_source/source/2015/*"))

# Applying alignment
results = []
for s in sources:
    src_img = cv2.imread(s)
    dst_img = cv2.imread(destination)
    src_h, hmat, rmse, n_matches = \
        homography(src_img, dst_img, ransac_th=5)
    res = dict(
        source = s,
        destination = destination,
        rmse = rmse,
        n_matches = n_matches
    )
    res = pd.DataFrame.from_dict(res, orient="index").T
    results.append(res)
    print("-----")
    print(res)
    print("-----")
    cv2.imwrite(s.replace("/source/", "/aligned/").replace(".JPG", ".png"), src_h)

# Alignment report
results = pd.concat(results)
results.to_csv("data_source/alignment_report.csv", index=None)

# Adjust Occulusion
aligned = sorted(glob("data_source/aligned/2015/*"))
aligned.append(destination)
mask = np.ones(cv2.imread(aligned[0])[:,:,0].shape)
for s in aligned:
    src = cv2.imread(s)
    for c in range(3):
        mask[src[:,:,c]==0] = 0
np.save("data_source/mask", mask)

for s in aligned:
    src = cv2.imread(s)
    src[mask==0] = 0
    cv2.imwrite(s, src)

# Since our label data is prepeared with the picture of 2015, homography matrix for 2015 to 2020 translation will be needed.
src_img = cv2.imread("data_source/mrd_085_eos_vis_20151010_1205.jpg")
src_h, hmat, rmse, n_matches = \
        homography(src_img, dst_img, ransac_th=5)
cv2.imwrite("data_source/mrd_085_eos_vis_20151010_1205_aligned.jpg", src_h)
np.savetxt("data_source/hmat_label.txt", hmat)