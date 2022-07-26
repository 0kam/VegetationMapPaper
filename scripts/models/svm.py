from torch.utils.data import DataLoader
from scripts.utils.utils import read_sses, DrawDS, TrainDS, draw_legend, draw_teacher
from tqdm import tqdm
from pathlib import Path
from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.svm import SVC
from thundersvm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
import os

class SVClassifier():
    def __init__(self, data_dir, labels_dir, label="all", kernel="rbf", C=1.0, degree=3, gamma="scaled", transforms = None, cmap = "jet"):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.cmap = cmap

        self.classes = [Path(n).name for n in glob(data_dir + "/*")]
        _, labels = read_sses(labels_dir, (9999,9999), label=label)
        self.label = label
        self.labels = labels
        self.labels_dir = labels_dir
        self.ds = TrainDS(data_dir)
        self.transforms = transforms
        x, _ = self.get_data(self.ds)
        if self.gamma == "scaled":
            self.gamma = 1 / (x.shape[1] * x.var())
        self.class_to_idx = self.ds.dataset.class_to_idx
        self.model = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma= self.gamma)
    
    def train(self, train_data=None, val_data=None):
        """
        Training the SVM model.
        -----------
        train_data: dict of np.array
            The training data as a dict with "x" and "y" as keys. 
        val_data: dict of np.array
            The validation data as a dict with "x" and "y" as keys. 
        """
        # Model fitting
        if train_data is None:
            x, y = self.get_data(self.ds)
            train_data = {"x": x, "y": y}
        if val_data is None:
            x, y = self.get_data(self.ds)
            val_data = {"x": x, "y": y}
        self.model.fit(train_data["x"], train_data["y"])
        # Model Testing
        pred_ys = self.model.predict(val_data["x"])
        r = classification_report(val_data["y"].astype(int), pred_ys.astype(int), \
            output_dict=True, zero_division = 0)
        res = {}
        for c, i in self.class_to_idx.items():
            c = self.labels.query("classIndex == {}".format(c))["label"].item()
            res[c] = r[str(i)]
        res["macro avg"] = r["macro avg"]
        res["weighted avg"] = r["weighted avg"]
        # print("F1 macro:", res["macro avg"]["f1-score"])
        return res
    
    def kfold(self, log_dir, k=5, shuffle = True):
        log_dir = log_dir
        if os.path.exists(log_dir) == False:
            os.makedirs(log_dir)
        kf = StratifiedKFold(n_splits=k, shuffle=shuffle)
        results = []
        x, y = self.get_data(self.ds)
        for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(x, y)), total=k):
            # Resetting the data and model
            train_data = {"x": x[train_idx], "y": y[train_idx]}
            val_data = {"x": x[val_idx], "y": y[val_idx]}
            self.model = self.get_instance()
            # Training model
            res = self.train(train_data, val_data)
            # Saving result
            df = pd.DataFrame(res)
            df["metrics"] = df.index
            df = pd.melt(df, id_vars="metrics", var_name="vegetation", value_name="value")
            df["fold"] = fold
            results.append(df)
        pd.concat(results).to_csv(log_dir + "/stratified_cv.csv")
        
    def draw(self, image_dir, out_path, kernel_size, batch_size):
        with Image.open(glob(image_dir+"/*")[0]) as img:
            w, h = img.size
        dataset = DrawDS(image_dir, kernel_size)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        for x in tqdm(loader):
            x = np.array(x.reshape(x.shape[0], -1))
            y = self.model.predict(x)
            pred_ys.append(y)
        
        seg_image = np.concatenate(pred_ys).reshape([h,w])
        plt.imsave(out_path, seg_image, cmap = self.cmap)
        return seg_image
    
    def get_data(self, ds):
        x = []
        y = []
        # Train Dataset returns data for each label
        for i in range(len(self.classes)):
            xi, yi = ds[i]
            x.append(xi)
            y.append(yi)
        x = np.concatenate(x)
        x = self.apply_transforms(x)
        y = np.concatenate(y)
        return x, y
    
    def save(self, path):
        """
        Save a model as a .txt file.
        """
        d = Path(path).parent
        if os.path.exists(d) == False:
            os.makedirs(d)
        self.model.save_to_file(path)
    
    def load(self, path):
        if os.path.exists(path) == False:
            raise FileNotFoundError("{The model file {} not found".format(path))
        self.model.load_from_file(path)

    def get_instance(self):
        return SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma= self.gamma)
    
    def apply_transforms(self, x):
        if self.transforms is None:
            return x
        else:
            return self.transforms(x)
    
    def draw_teacher(self, out_path, image_size):
        draw_teacher(out_path, self.labels_dir, self.class_to_idx, image_size, \
            self.label, cmap = self.cmap)
    
    def draw_legend(self, out_path):
        draw_legend(out_path, self.labels_dir, self.class_to_idx, self.label, cmap = self.cmap)