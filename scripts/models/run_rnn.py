from scripts.models.rnn import RNNClassifier
import tensorboardX
from matplotlib.colors import ListedColormap
import numpy as np

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

rnn = RNNClassifier("data/multidays_1x1", "data_source/labels", 500, test_size=0.2, cmap=cmap)
rnn.kfold(200, "cv/rnn1x1/", k=5, shuffle=False)
res = rnn.draw("data_source/aligned/multi_days", "results/rnn1x1.png", (1,1), 5000)
np.save("results/rnn1x1.npy", res[0])

from matplotlib import pyplot as plt

plt.imsave("results/rnn1x1.png", res[0], cmap = cmap)

from glob import glob
rnn = RNNClassifier("data/multidays_1x1", "data_source/labels", 500, test_size=0.2, cmap=cmap)

for d in glob("runs/cv/rnn1x1/*"):
    print(d)
    rnn.load(d + "/best.pth")
    res = rnn.draw("data_source/aligned/multi_days", "results/rnn1x1.png", (1,1), 5000)
    np.save(d + "/pred.npy", res[0])
    plt.imsave(d + "/pred.png", res[0], cmap = cmap)

preds = []
import os
import torch
for d in glob("runs/cv/rnn1x1/*"):
    if os.path.isdir(d):
        preds.append(np.load(d + "/pred.npy"))

preds = torch.Tensor(np.stack(preds))
pred, _ = torch.mode(preds, dim = 0)
np.save("results/rnn.npy", pred)
plt.imsave("results/rnn.png", pred, cmap=cmap)