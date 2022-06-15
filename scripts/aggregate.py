import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np
from scipy.stats import mode
import torch

mask = np.asarray(Image.open("results/cnn_lstm5x5_cv_5_ep_200/mask.png"))[:,:,0]
mask[mask>0] = 1

d = "results/cnn_lstm5x5_cv_5_ep_200/"
res2012 = []
res2021 = []

for i in range(5):
    r2012 = np.load(d + "fold_{}/2012.npy".format(i))
    r2021 = np.load(d + "fold_{}/2021.npy".format(i))
    res2012.append(r2012)
    res2021.append(r2021)

res2012 = torch.Tensor(np.stack(res2012))
res2021 = torch.Tensor(np.stack(res2021))

res2012 = torch.mode(res2012, dim = 0)
res2021 = torch.mode(res2021, dim = 0)

res2012 = res2012[0].numpy()
res2021 = res2021[0].numpy()

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
np.save("results/cnn_lstm5x5_cv_5_ep_200/res2012_cnn_5x5.npy", res2012)
np.save("results/cnn_lstm5x5_cv_5_ep_200/res2021_cnn_5x5.npy", res2021)
plt.imsave("results/cnn_lstm5x5_cv_5_ep_200/cnn_5x5_2012.png", res2012, cmap=cmap_veg)
plt.imsave("results/cnn_lstm5x5_cv_5_ep_200/cnn_5x5_2021.png", res2021, cmap=cmap_veg)

res2012 = np.load("results/cnn_lstm5x5_cv_5_ep_200/res2012_cnn_5x5.npy")
res2021 = np.load("results/cnn_lstm5x5_cv_5_ep_200/res2021_cnn_5x5.npy")

diff = np.ones(res2012.shape)
diff[np.logical_and(res2012!=5, res2021==5)] = 2 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2012==5, res2021!=5)] = 3 # ハイマツ -> ハイマツ以外
diff = diff * mask
plt.imsave("results/cnn_lstm5x5_cv_5_ep_200/diff_haimatsu_cnn_5x5.png", diff, cmap = cmap_diff)
np.save("results/cnn_lstm5x5_cv_5_ep_200/diff_haimatsu.npy", diff)

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
np.save("results/cnn_lstm5x5_cv_5_ep_200/diff_sasa.npy", diff)
plt.imsave("results/cnn_lstm5x5_cv_5_ep_200/diff_sasa_cnn_5x5.png", diff, cmap = cmap_diff)

src = Image.open("results/cnn_lstm5x5_cv_5_ep_200/mask.png")
src.putalpha(alpha=255)
diff_sasa = Image.open("results/cnn_lstm5x5_cv_5_ep_200/diff_sasa_cnn_5x5.png")
diff_haimatsu = Image.open("results/cnn_lstm5x5_cv_5_ep_200/diff_haimatsu_cnn_5x5.png")

diff_sasa = Image.alpha_composite(src, diff_sasa)
diff_haimatsu = Image.alpha_composite(src, diff_haimatsu)

diff_sasa.save("results/cnn_lstm5x5_cv_5_ep_200/diff_sasa_cnn_5x5.png")
diff_haimatsu.save("results/cnn_lstm5x5_cv_5_ep_200/diff_haimatsu_cnn_5x5.png")
