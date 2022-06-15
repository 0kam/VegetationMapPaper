from PIL import Image
from torch import log_
from scripts.utils import set_patches, find_best_epoch
from scripts.cnn_lstm import CNNLSTM
import numpy as np
import tensorboardX
hmat = np.loadtxt("data_source/hmat_label.txt")
mask = np.load("data_source/mask.npy")
shrink = 3
log_dir = "cnn_5x5"

# set_patches("data_source/labels", "data_source/aligned/2012", "data/train/patches5x5/", (5,5), batch_size=50, unlabelled=False, shrink=shrink, hmat=hmat, mask=mask)
# set_patches("data_source/labels", "data_source/aligned/2021", "data/train/patches5x5/", (5,5), batch_size=50, unlabelled=False, shrink=shrink, hmat=hmat, mask=mask)

lstm = CNNLSTM("data/train/patches5x5", "data_source/labels", (5,5), 200, device="cuda", test_size = 0.2)
lstm.draw_legend("results/legend.png")
lstm.draw_teacher("results/teacher.png", (5616,3744), shrink=shrink, hmat=hmat, mask=mask)

k = 5
cv_log_dir = "cnn_lstm5x5_cv_5_ep_200"
lstm.kfold(100, cv_log_dir, k)

best_epochs = []
res2012 = []
res2021 = []
for i in range(k):
    path = "runs/" +  cv_log_dir + "/fold_{}".format(i) + "/best.pth"
    lstm = CNNLSTM("data/train/patches5x5", "data_source/labels", (5,5), 1000, device="cuda", test_size = 0)
    lstm.load(path)
    r2012, _ = lstm.draw("data_source/aligned/2012", path.replace("best.pth", "2012.png"), (5,5), 50000)
    r2021, _ = lstm.draw("data_source/aligned/2021", path.replace("best.pth", "2021.png"), (5,5), 50000)
    np.save(path.replace("best.pth", "2012.npy"), r2012)
    np.save(path.replace("best.pth", "2021.npy"), r2021)
    res2012.append(r2012)
    res2021.append(r2021)
    best_epochs.append(find_best_epoch(path.replace("best.pth", "all_scalars.json")))

best_epoch = int(np.mean(np.array(best_epochs)))
lstm = CNNLSTM("data/train/patches5x5", "data_source/labels", (5,5), 200, device="cuda", test_size = 0.2)
lstm.train(best_epoch, log_dir)

lstm.load("runs/" + log_dir + "/best.pth")

res2021, _ = lstm.draw("data_source/aligned/2021", "results/cnn_5x5_2021.png", (5,5), 50000)
res2012, _ = lstm.draw("data_source/aligned/2012", "results/cnn_5x5_2012.png", (5,5), 50000)

# Plotting results
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from matplotlib.colors import ListedColormap

## Color maps

cmap_veg = ListedColormap(
    np.array(
        [[0,0,0], # 0: No data
        [154,205,50], # 1: Dwarf bamboo
        [192,192,192], # 2: No vegetation
        [139,69,19], # 3: Other vegetation
        [255,255,255], # 4: Sky
        [0,100,0]] # 5: Dwarf pine
    ) / 255
)

cmap_diff = ListedColormap(
    np.array(
        [[0,0,0,0], # 0: No data
        [192,192,192, 127], # 1: No Difference
        [255,0,0, 127], # 2: Increase
        [0,0,255, 127]] # 3: Decrease
    ) / 255
)

res2012 = (res2012 +1) * mask
res2021 = (res2021  +1) * mask
np.save("results/res2012_cnn_5x5.npy", res2012)
np.save("results/res2021_cnn_5x5.npy", res2021)
plt.imsave("results/cnn_5x5_2012.png", res2012, cmap=cmap_veg)
plt.imsave("results/cnn_5x5_2021.png", res2021, cmap=cmap_veg)

lstm.labels
lstm.class_to_idx

res2012 = np.load("results/res2012_cnn_5x5.npy")
res2021 = np.load("results/res2021_cnn_5x5.npy")

diff = np.ones(res2012.shape)
diff[np.logical_and(res2012!=5, res2021==5)] = 2 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2012==5, res2021!=5)] = 3 # ハイマツ -> ハイマツ以外
diff = diff * mask
plt.imsave("results/diff_haimatsu_cnn_5x5.png", diff, cmap = cmap_diff)
np.save("results/diff_haimatsu.npy", diff)

diff_labels = [
    "出現",
    "消滅"
]
pl.cla()
for i, l in zip(range(2), diff_labels):
    color = cmap_diff(i+1)
    pl.plot(0, 0, "-", c = color, label = l, linewidth = 10)
pl.legend(loc = "center", prop = {"family": "MigMix 1P"})
pl.savefig("results/diff_legend.png")

diff = np.ones(res2012.shape)
diff[np.logical_and(res2012!=1, res2021==1)] = 2 # ササ以外 -> ササ
diff[np.logical_and(res2012==1, res2021!=1)] = 3 # ササ -> ササ以外 
diff = diff * mask
np.save("results/diff_sasa.npy", diff)
plt.imsave("results/diff_sasa_cnn_5x5.png", diff, cmap = cmap_diff)

src = Image.open("data_source/aligned/2012/IMG_9514.png")
src.putalpha(alpha=255)
diff_sasa = Image.open("results/diff_sasa_cnn_5x5.png")
diff_haimatsu = Image.open("results/diff_haimatsu_cnn_5x5.png")

diff_sasa = Image.alpha_composite(src, diff_sasa)
diff_haimatsu = Image.alpha_composite(src, diff_haimatsu)

diff_sasa.save("results/diff_sasa_cnn_5x5.png")
diff_haimatsu.save("results/diff_haimatsu_cnn_5x5.png")
