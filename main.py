import copy
import random

import higher
import torch
import math
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter
import model
import dataset
def load_para():
    st=torch.load('model2.pth')
    modG.load_state_dict(st['modG'])
    modD.load_state_dict(st['modD'])
    optD.load_state_dict(st['optD'])
    optG.load_state_dict(st['optG'])

def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def cal_gp(D, real_imgs, fake_imgs):  # 定义函数，计算梯度惩罚项gp
    r = torch.rand(size=(real_imgs.shape[0], 1, 1, 1))  # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布
    x = (r * real_imgs + (1 - r) * fake_imgs).requires_grad_(True)  # 输入样本x，由真假样本按照比例产生，需要计算梯度
    d = D(x)  # 判别网络D对输入样本x的判别结果D(x)
    fake = torch.ones_like(d)  # 定义与d形状相同的张量，代表梯度计算时每一个元素的权重
    g = torch.autograd.grad(  # 进行梯度计算
        outputs=d,  # 计算梯度的函数d，即D(x)
        inputs=x,  # 计算梯度的变量x
        grad_outputs=fake,  # 梯度计算权重
        create_graph=True,  # 创建计算图
        retain_graph=True  # 保留计算图
    )[0]  # 返回元组的第一个元素为梯度计算结果
    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()  # (||grad(D(x))||2-1)^2 的均值
    return gp  # 返回梯度惩罚项gp

def train_d(D,G,opD):
    pct=train_loader.next()
    opD.zero_grad()
    pout=G(torch.randn(pct.size(0),100,1,1))
    gd=cal_gp(D,pct,pout.detach())
    dc=torch.mean(D(pout.detach()))-torch.mean(D(pct))+gd*10
    dc.backward()
    opD.step()
    #emaD.update()
    return (dc).item()
def train_d_uprolled_higher(D,G,opD):
    pct=train_loader.next()
    pout = G(torch.randn(pct.size(0), 100, 1, 1))
    gd=cal_gp(D,pct,pout.detach())
    dc = torch.mean(D(pout.detach())) - torch.mean(D(pct))+gd*10
    opD.zero_grad()
    dc.backward()
    opD.step()
    #for dt in D.parameters():
    #     dt.data.clamp_(-0.01,0.01)
    return dc.item()

def g_loop(D,G,opD,opG):
    opD.zero_grad()
    geninput=torch.randn(train_loader.batchsize,100,1,1)
    backup=D.state_dict().copy()
    backup2=opD.state_dict().copy()
    g_error=0
    kk=(int)(standard)
    for i in range(kk):
        train_d_uprolled_higher(D,G,opD)
    g_error=-torch.mean(D(G(torch.randn(train_loader.batchsize,100,1,1))))
    opG.zero_grad()
    g_error.backward()
    opG.step()
    #emaG.update()
    D.load_state_dict(backup)
    opD.load_state_dict(backup2)
    del backup,backup2
    return g_error.item(),G(geninput)

train_data=dataset.MyDataSet()
train_loader=model.MyDataLoader(train_data,batchsize=64)
modG=model.Generator()
modD=model.Discriminator()
optG=torch.optim.Adam(modG.parameters(),lr=1e-3,betas=(0.5,0.9))
optD=torch.optim.Adam(modD.parameters(),lr=1e-3,betas=(0.5,0.9))
modG.apply(weights_init)
modD.apply(weights_init)
scG=torch.optim.lr_scheduler.StepLR(optG,10,0.99)
scD=torch.optim.lr_scheduler.StepLR(optD,10,0.99)
modD=add_sn(modD)
modG=add_sn(modG)
real_label=1
fake_label=0
writer=SummaryWriter()
emaD=model.EMA(modD,0.999)
emaG=model.EMA(modG,0.999)
emaG.register()
emaD.register()
k=0
before_pct=[[]]*4
#load_para()
standard=6
for epoch in range(100):
    for tms in range(1000):
        standard*=0.99
        k+=1
        gz=0
        dz=0
        pout=0
        for i in range(5):
            dz+=train_d(modD,modG,optD)

        for i in range(1):
            addg,pout=g_loop(modD,modG,optD,optG)
            gz+=addg
        if k <= 2000:
            scD.step()
            scG.step()
        writer.add_scalar("LossG/train", gz/3, k)
        writer.add_scalar("LossD/train",dz/6,k)


        if k%10==0:
            st_D={
                'modD':modD.state_dict(),
                'modG':modG.state_dict(),
                'optD':optD.state_dict(),
                'optG':optG.state_dict()
            }
            torch.save(st_D,"model2.pth")
            print("Saved")
            writer.add_images("Generate",pout/2+0.5,k)

            writer.add_images("Real", train_loader.next() / 2 + 0.5, k)
