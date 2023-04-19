import torch
import math
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter
class MyDataLoader():
    def __init__(self,dataset,batchsize):
        self.dataset=dataset
        self.length=dataset.__len__()
        self.k=0
        self.batchsize=batchsize
    def next(self):
        if self.k+self.batchsize>self.length:
            self.k=0
        ap=[]
        for i in range(self.k,self.k+self.batchsize):
            ap.append(self.dataset.__getitem__(i))
        self.k+=self.batchsize
        return torch.stack(ap,0)

class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
class Block(nn.Module):
    def __init__(self,input_kernal,output_kernal):
        super(Block,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(input_kernal,output_kernal,3,1,1,bias=False),
            nn.BatchNorm2d(output_kernal),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.main(x)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        #print(self.VGG19)
        #nn.ConvTranspose2d
        """
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(512,256,3,1,1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2,True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256,128,3,1,1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2,True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128,64,3,1,1),
                    nn.InstanceNorm2d(64),
                    nn.LeakyReLU(0,True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64,32,3,1,1),
                    nn.InstanceNorm2d(32),
                    nn.LeakyReLU(0.2,True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(32,16,3,1,1),
                    nn.InstanceNorm2d(16),
                    nn.LeakyReLU(0.2,True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(16,3,3,1,1),
                    nn.Tanh()"""
        self.up_sample=nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(100,512),
            nn.Upsample(scale_factor=2),
            Block(512,256),
            nn.Upsample(scale_factor=2),
            Block(256,128),
            nn.Upsample(scale_factor=2),
            Block(128,64),
            nn.Upsample(scale_factor=2),
            Block(64,32),
            nn.Upsample(scale_factor=2),
            Block(32,16),
            Block(16,8),
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Tanh()
            )
    def forward(self,x):
        return self.up_sample(x)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv=nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([64,16,16]),
            #nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([128,8,8]),
            #nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32 * 4, 32 * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([256,4,4]),
            #nn.BatchNorm2d(32 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(32 * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
            MinibatchStdDev()
            )
    def forward(self,x):
        return self.conv(x).view(-1,1).squeeze(1)
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}