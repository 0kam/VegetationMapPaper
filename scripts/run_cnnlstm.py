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

import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
import cv2
np.save("results/res2012_cnn_9x9.npy", res2012)
np.save("results/res2020_cnn_9x9.npy", res2020)

lstm.labels
lstm.class_to_idx

diff = np.zeros(res2012.shape)
diff[np.logical_and(res2012!=4, res2020==4)] = 1 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2012==4, res2020!=4)] = 2 # ハイマツ -> ハイマツ以外
cmap = plt.get_cmap("tab20", 3)
plt.imsave("results/diff_haimatsu_cnn_9x9.png", diff, cmap = cmap)

diff_labels = [
    "出現",
    "消滅"
]
pl.cla()
for i, l in zip(range(2), diff_labels):
    color = cmap(i+1)
    pl.plot(0, 0, "-", c = color, label = l, linewidth = 10)
pl.legend(loc = "center", prop = {"family": "MigMix 1P"})
pl.savefig("results/diff_legend.png")

diff = np.zeros(res2012.shape)
diff[np.logical_and(res2012!=0, res2020==0)] = 1 # ササ以外 -> ササ
diff[np.logical_and(res2012==0, res2020!=0)] = 2 # ササ -> ササ以外 
diff = diff * mask
plt.imsave("results/diff_sasa_cnn_9x9_R2.png", diff, cmap = cmap)