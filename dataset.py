import glob

import torch
import math
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torch.nn as nn
import PIL
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self):
        super(MyDataSet,self).__init__()
        self.root=glob.glob("D:\\BaiduNetdiskDownload\\anime-faces\\*.png")
        self.trans=torchvision.transforms.Compose([
            transform.ToTensor(),
            transform.Resize(64),
        ])
    def __len__(self):
        return len(self.root)

    def __getitem__(self, item):
        rt=self.root[item]
        pct=PIL.Image.open(rt).convert("RGB")
        pct=self.trans(pct)*2-1
        return pct