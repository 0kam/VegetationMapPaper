import torch
from torch import nn
from torch.nn import functional as F
import torch_optimizer as optim 
from scripts.utils import read_sses, DrawDS, LabelledDS, cf_labelled, draw_legend, draw_teacher
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from tqdm import tqdm
from pathlib import Path
from glob import glob
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

class AsImage(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    def __call__(self, x):
        return x.view((-1,) + self.kernel_size)

class CRNNClassifier(nn.Module):
    def __init__(self, x_shape, y_dim):
        super(CRNNClassifier, self).__init__()
        self.x_shape = x_shape
        self.conv1 = nn.Conv2d(3, 8, (3,3), 1)
        self.bn2d_1 = nn.BatchNorm2d(8)
        self.prelu1 = nn.PReLU()
        self.maxpool1 = nn.MaxPool2d((3,3), 1)       
        conv_out_shape = (x_shape[2]-4) * (x_shape[3]-4) * 8
        self.h_dim = int((conv_out_shape + y_dim) / 2)
        self.lstm = nn.LSTM(conv_out_shape, self.h_dim, batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.h_dim)
        self.do1 = nn.Dropout(0.0)
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)
        self.prelu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(self.h_dim)
        self.do2 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(self.h_dim, y_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.prelu1(h)
        h = self.bn2d_1(h)
        h = self.maxpool1(h)
        h = h.view((-1, self.x_shape[0], h.shape[1]*h.shape[2]*h.shape[3]))
        _, h = self.lstm(h)
        h = h[0].view(-1, self.h_dim)
        h = self.bn1(h)
        h = self.do1(h)
        h = self.prelu2(self.fc1(h))
        h = self.bn2(h)
        h = self.do2(h)
        return F.softmax(self.fc2(h), dim=1)

class CNNLSTM():
    def __init__(self, data_dir, labels_dir, kernel_size, batch_size, device="cuda", num_workers=20, label="all"):
        self.classes = [Path(n).name for n in glob(data_dir + "/labelled/*")]
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        _, labels = read_sses(labels_dir, (9999,9999), label=label)
        self.label = label
        self.labels = labels
        self.labels_dir = labels_dir
        ## labelled data loader
        self.ds = LabelledDS(data_dir + "/labelled/")
        self.idx = list(range(len(self.ds.dataset.targets)))
        train_indices, val_indices = train_test_split(self.idx, test_size=0.2, stratify=self.ds.dataset.targets)
        train_dataset = torch.utils.data.Subset(self.ds, train_indices)
        val_dataset = torch.utils.data.Subset(self.ds, val_indices)
        x, y = train_dataset[0]

        self.x_shape = (x.shape[1],) + (3,) + kernel_size
        self.y_dim = len(self.classes)
        self.tf_train = transforms.Compose([
            AsImage(self.x_shape[1:4])#,
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        ])
        self.tf_valid = transforms.Compose([
            AsImage(self.x_shape[1:4])
        ])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=cf_labelled)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cf_labelled)
        self.class_to_idx = self.train_loader.dataset.dataset.dataset.class_to_idx

        self.model = CRNNClassifier(self.x_shape, self.y_dim).to(self.device)
        self.optimizer = optim.RAdam(self.model.parameters(), lr=1e-3)
        self.loss_cls = nn.CrossEntropyLoss()
        self.best_test_loss = 9999
    
    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for x, y in tqdm(self.train_loader):
            x = self.tf_train(x)
            x = x.to(self.device)
            y = torch.eye(self.y_dim)[y].to(self.device)
            self.model.zero_grad()
            y2 = self.model(x)
            loss = self.loss_cls(y2, y.argmax(1))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss * self.train_loader.batch_size / len(self.train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss
    
    def _val(self, epoch):
        self.model.eval()
        test_loss = 0
        # total = [0 for _ in range(len(self.classes))]
        # tp = [0 for _ in range(len(self.classes))]
        # fp = [0 for _ in range(len(self.classes))]
        ys = []
        pred_ys = []
        for x, _y in self.val_loader:
            x = self.tf_valid(x)
            x = x.to(self.device)
            y = torch.eye(self.y_dim)[_y].to(self.device)
            with torch.no_grad():
                y2 = self.model(x)
            loss = self.loss_cls(y2, y.argmax(1))
            test_loss += loss
            pred_ys.append(y2.argmax(1))
            ys.append(_y)
            # for c in range(len(self.classes)):
            #     pred_yc = y2[_y==c]
            #     _yc = _y[y2==c]
            #     total[c] += len(_y[_y==c])
            #     tp[c] += len(pred_yc[pred_yc==c])
            #     fp[c] += len(_yc[_yc!=c])
        ys = torch.cat(ys).detach().cpu()
        pred_ys = torch.cat(pred_ys).detach().cpu()

        test_loss = test_loss * self.val_loader.batch_size / len(self.val_loader.dataset)
        r = classification_report(ys, pred_ys, output_dict=True)
        res = {}
        for c, i in self.class_to_idx.items():
            c = self.labels.query("classIndex == {}".format(c))["label"].item()
            res[c] = r[str(i)]
        res["macro avg"] = r["macro avg"]
        res["weighted avg"] = r["weighted avg"]
        print("Test Loss:", str(test_loss), "F1 macro:", res["macro avg"]["f1-score"])
        return test_loss, res
    
    def train(self, epochs, log_dir):
        writer = SummaryWriter("./runs/" + log_dir)
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            val_loss, res = self._val(epoch)
            if val_loss < self.best_test_loss:
                self.best_model = CRNNClassifier(self.x_shape, self.y_dim).to(self.device)
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_metrics = res
                self.best_test_loss = val_loss
                torch.save(self.best_model.state_dict(), "./runs/"+log_dir+"/best.pth")
            writer.add_scalar("test_loss", val_loss, epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            for c, r in res.items():
                r = {k: v for k, v in r.items() if k.lower() != "support"}
                writer.add_scalars("val_" + c, r, epoch)
        writer.export_scalars_to_json("./runs/" + log_dir + "all_scalars.json")
    
    def kfold(self, epochs, log_dir, k=5):
        kf = StratifiedKFold(n_splits=k)
        results = []
        for fold, (train_indices, val_indices) in enumerate(kf.split(self.idx, self.ds.dataset.targets)):
            # Reset the dataloaders and the model
            train_dataset = torch.utils.data.Subset(self.ds, train_indices)
            val_dataset = torch.utils.data.Subset(self.ds, val_indices)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=cf_labelled)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=cf_labelled)
            self.model = CRNNClassifier(self.x_shape, self.y_dim).to(self.device)
            self.optimizer = optim.RAdam(self.model.parameters(), lr=1e-3)
            # train
            self.train(epochs, log_dir+"/fold_"+str(fold))
            df = pd.DataFrame(self.best_metrics)
            df["metrics"] = df.index
            df = pd.melt(df, id_vars="metrics", var_name="vegetation", value_name="value")
            df["fold"] = fold
            results.append(df)
        pd.concat(results).to_csv("./runs/" + log_dir + "/staritfied_cv.csv")
    
    def draw(self, image_dir, out_path, kernel_size, batch_size):
        with Image.open(glob(image_dir+"/*")[0]) as img:
            w, h = img.size
        dataset = DrawDS(image_dir, kernel_size)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        confs = []
        self.best_model.eval()
        with torch.no_grad():
            for x in tqdm(loader):
                x = self.tf_valid(x)
                x = x.to(self.device)
                y = self.best_model(x)
                pred_y = y.argmax(1).detach().cpu()
                conf = y.max(1)[0].detach().cpu()
                pred_ys.append(pred_y)
                confs.append(conf)
        
        seg_image = torch.cat(pred_ys).reshape([h,w]).numpy()
        confs = torch.cat(confs).reshape([h,w]).numpy()
        cmap = plt.get_cmap("tab20", len(self.classes))
        plt.imsave(out_path, seg_image, cmap = cmap)
        return seg_image, confs
    
    def draw_teacher(self, out_path, image_size, shrink=0, hmat=None, mask=None):
        draw_teacher(out_path, self.labels_dir, self.class_to_idx, image_size, self.label, shrink, hmat, mask)
    
    def draw_legend(self, out_path):
        draw_legend(out_path, self.labels_dir, self.class_to_idx, self.label)
    
    def load(self, path):
        self.best_model = CRNNClassifier(self.x_shape, self.y_dim).to(self.device)
        self.best_model.load_state_dict(torch.load(path))