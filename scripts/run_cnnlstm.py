from PIL import Image
from scripts.utils import set_patches
from scripts.cnn_lstm import CNNLSTM
import numpy as np
import tensorboardX

hmat = np.loadtxt("data_source/hmat_label.txt")
mask = np.load("data_source/mask.npy")
shrink = 10
log_dir = "cnn_lstm9x9"

#set_patches("data_source/labels", "data_source/aligned/2012", "data/train/patches9x9/", (9,9), batch_size=50, unlabelled=False, shrink=shrink, hmat=hmat, mask=mask)
#set_patches("data_source/labels", "data_source/aligned/2020", "data/train/patches9x9/", (9,9), batch_size=50, unlabelled=False, shrink=shrink, hmat=hmat, mask=mask)

lstm = CNNLSTM("data/train/patches9x9", "data_source/labels", (9,9), 200, device="cuda")
lstm.draw_legend("results/legend.png")
lstm.draw_teacher("results/teacher.png", (5616,3744), shrink=shrink, hmat=hmat, mask=mask)

lstm.kfold(10, "cnn_lstm9x9_cv_5_ep_200", 5)

lstm.train(200, log_dir)
lstm.load("./runs/"+log_dir+"/best.pth")

res2012, _ = lstm.draw("data_source/aligned/2012", "results/cnn_9x9_2012.png", (9,9), 50000)
res2020, _ = lstm.draw("data_source/aligned/2020", "results/cnn_9x9_2020.png", (9,9), 50000)


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

#res2012 = (res2012 +1) * mask
#res2020 = (res2020  +1) * mask
np.save("results/res2012_cnn_9x9.npy", res2012)
np.save("results/res2020_cnn_9x9.npy", res2020)
plt.imsave("results/cnn_9x9_2012.png", res2012, cmap=cmap_veg)
plt.imsave("results/cnn_9x9_2020.png", res2020, cmap=cmap_veg)

lstm.labels
lstm.class_to_idx

res2012 = np.load("results/res2012_cnn_9x9.npy")
res2020 = np.load("results/res2020_cnn_9x9.npy")

diff = np.ones(res2012.shape)
diff[np.logical_and(res2012!=5, res2020==5)] = 2 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2012==5, res2020!=5)] = 3 # ハイマツ -> ハイマツ以外
diff = diff * mask
plt.imsave("results/diff_haimatsu_cnn_9x9.png", diff, cmap = cmap_diff)
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
diff[np.logical_and(res2012!=1, res2020==1)] = 2 # ササ以外 -> ササ
diff[np.logical_and(res2012==1, res2020!=1)] = 3 # ササ -> ササ以外 
diff = diff * mask
np.save("results/diff_sasa.npy", diff)
plt.imsave("results/diff_sasa_cnn_9x9.png", diff, cmap = cmap_diff)

src = Image.open("data_source/aligned/2012/mrd_085_eos_vis_20121014_1200.png")
src.putalpha(alpha=255)
diff_sasa = Image.open("results/diff_sasa_cnn_9x9.png")
diff_haimatsu = Image.open("results/diff_haimatsu_cnn_9x9.png")

diff_sasa = Image.alpha_composite(src, diff_sasa)
diff_haimatsu = Image.alpha_composite(src, diff_haimatsu)

diff_sasa.save("results/diff_sasa_cnn_9x9.png")
diff_haimatsu.save("results/diff_haimatsu_cnn_9x9.png")