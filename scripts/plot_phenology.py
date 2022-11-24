import numpy as np
from PIL import Image
from glob import glob
import random
from pathlib import Path
import pandas as pd

def sample_timeseries(n = 10):
  n = int(n)
  files = sorted(glob("data_source/aligned/multi_days/*"))
  
  t = np.load("results/teacher.npy")
  t2 = np.asarray(Image.open("results/teacher.png"))
  t2 = np.max(t2, 2)
  t[t2 == 0] = 99
  imgs = np.stack([Image.open(f) for f in files])
  
  vege = []
  path = []
  num = []
  R = []
  G = []
  B = []
  
  paths = [Path(f).stem for f in files]
  
  for c in range(7):
    pixes = imgs[:, t == c, :]
    random.seed(12)
    idx = random.sample(range(pixes.shape[1]), n)
    pix = pixes[:, idx, :]
    for i in range(pix.shape[1]):
      p = pix[:, i, :]
      for j in range(pix.shape[0]):
        vege.append(c)
        path.append(paths[j])
        num.append(i)
        R.append(int(p[j, 0]))
        G.append(int(p[j, 1]))
        B.append(int(p[j, 2]))
  df = pd.DataFrame({
    "vegetation": vege,
    "index": num,
    "path": path,
    "R": R,
    "G": G,
    "B": B
  })
  
  return df
