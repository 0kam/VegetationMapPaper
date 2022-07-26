
from scripts.models.svm import SVClassifier
from matplotlib.colors import ListedColormap
import numpy as np
from glob import glob
from pathlib import Path

cmap = ListedColormap(
    np.array(
        [[154,205,50], # 1: Dwarf bamboo
        [70, 130, 180], # 2: Other vegetation
        [192,192,192], # 3: No vegetation
        [220,20,60], # 4: ナナカマド
        [255,215,0], # 5: ダケカンバ
        [139,69,19],
        [0,100,0]] # 4: Dwarf pine
    ) / 255
)

# Cross Validation setting
k = 5
cv_log_dir = "runs/cv"
shuffle = False

svm = SVClassifier("data/multidays_1x1", "data_source/labels", transforms=None, cmap=cmap)
svm.draw_teacher("results/teacher.png", (5616, 3744))

def kfold(data_dir, image_dir, kernel_size, out_dir):
    svm = SVClassifier(data_dir, "data_source/labels", transforms=None, cmap = cmap)
    svm.kfold(out_dir, k, shuffle=shuffle)
    svm.train()
    svm.save(out_dir + "/model.txt")
    svm.load(out_dir + "/model.txt")
    pred = svm.draw(image_dir, out_dir + "/pred.png", kernel_size, 5000)
    np.save(out_dir + "/pred.npy", pred)

# multidays-------------------------------------------------------------------------------------------
data_dirs = ["data/multidays_1x1", "data/multidays_3x3", "data/multidays_5x5"]
image_dirs = ["data_source/aligned/multi_days/" for _ in range(3)]
kernel_sizes = [(1,1), (3,3), (5,5)]
out_dirs = [d.replace("data", cv_log_dir) for d in data_dirs]

for data_dir, image_dir, kernel_size, out_dir in zip(data_dirs, image_dirs, kernel_sizes, out_dirs):
    print(data_dir, image_dir, out_dir)
    kfold(data_dir, image_dir, kernel_size, out_dir)

# 1x1 single day--------------------------------------------------------------------------------------
data_dirs = [d + "/1x1/" for d in sorted(glob("data/single_day/*"))]
image_dirs = sorted(glob("data_source/aligned/single_day/*"))
kernel_sizes = [(1,1) for i in range(len(glob("data/single_day/*")))]
out_dirs = [d.replace("data/single_day", cv_log_dir) for d in data_dirs]

for data_dir, image_dir, kernel_size, out_dir in zip(data_dirs, image_dirs, kernel_sizes, out_dirs):
    print(data_dir, image_dir, out_dir)
    kfold(data_dir, image_dir, kernel_size, out_dir)

# 3x3 single day--------------------------------------------------------------------------------------
data_dirs = [d + "/3x3/" for d in sorted(glob("data/single_day/*"))]
image_dirs = sorted(glob("data_source/aligned/single_day/*"))
data_dirs.reverse()
image_dirs.reverse()
kernel_sizes = [(3,3) for i in range(len(glob("data/single_day/*")))]
out_dirs = [d.replace("data/single_day", cv_log_dir) for d in data_dirs]

for data_dir, image_dir, kernel_size, out_dir in zip(data_dirs, image_dirs, kernel_sizes, out_dirs):
    print(data_dir, image_dir, out_dir)
    kfold(data_dir, image_dir, kernel_size, out_dir)

# 5x5 single day--------------------------------------------------------------------------------------
data_dirs = [d + "/5x5/" for d in sorted(glob("data/single_day/*"))]
image_dirs = sorted(glob("data_source/aligned/single_day/*"))
data_dirs.reverse()
image_dirs.reverse()
kernel_sizes = [(5,5) for i in range(len(glob("data/single_day/*")))]
out_dirs = [d.replace("data/single_day", cv_log_dir) for d in data_dirs]

for data_dir, image_dir, kernel_size, out_dir in zip(data_dirs, image_dirs, kernel_sizes, out_dirs):
    print(data_dir, image_dir, out_dir)
    kfold(data_dir, image_dir, kernel_size, out_dir)