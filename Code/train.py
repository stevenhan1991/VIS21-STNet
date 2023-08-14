import torch.nn as nn
import torch.optim as optim
import time
import argparse
import DataPre
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.transform import resize
import torch.nn.functional as F
from model import *
import torch.nn.functional as F
from bicubic import *


def trainNet(model,D,args,dataset):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr_G,betas=(0.9,0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr_G,betas=(0.9,0.999))
    criterion = nn.MSELoss()
    for itera in range(1,args.epochs):
        
        print('======='+str(itera)+'========')
        loss_mse = 0
        x = time.time()
        if itera<=200:
            train_loader = dataset.GetTrainingDataS()
            for batch_idx, (ls,le,li) in enumerate(train_loader):
                if args.cuda:
                    ls = ls.cuda()
                    le = le.cuda()
                    li = li.cuda()
                high = model(ls,le)
                optimizer_G.zero_grad()
                error = (args.interval+2)*criterion(F.interpolate(high.permute(1,0,2,3,4),size=[dataset.crop_size[0],dataset.crop_size[1],dataset.crop_size[2]],mode='trilinear'),li)
                error.backward()
                loss_mse += error.mean().item()
                optimizer_G.step()

        elif itera>200 and itera<=400:
            train_loader = dataset.GetTrainingDataST()
            for batch_idx, (ls,le,hi) in enumerate(train_loader):
                if args.cuda:
                    ls = ls.cuda()
                    le = le.cuda()
                    hi = hi.cuda()
                high = model(ls,le)
                optimizer_G.zero_grad()
                error = (args.interval+2)*criterion(high.permute(1,0,2,3,4),hi)
                error.backward()
                loss_mse += error.mean().item()
                optimizer_G.step()

        elif itera>400:
            train_loader = dataset.GetTrainingDataST()
            for batch_idx, (ls,le,hi) in enumerate(train_loader):
                if args.cuda:
                    ls = ls.cuda()
                    le = le.cuda()
                    hi = hi.cuda()

                for p in model.parameters():
                    p.requires_grad = False

                optimizer_D.zero_grad()
                output_real = D(hi.permute(1,0,2,3,4))
                label_real = torch.ones(output_real.size()).cuda()
                real_loss = criterion(output_real,label_real)
                fake_data = model(ls,le)
                label_fake = torch.zeros(output_real.size()).cuda()
                output_fake = D(fake_data)
                fake_loss = criterion(output_fake,label_fake)
                loss = 0.5*(real_loss+fake_loss)    
                loss.backward()
                optimizer_D.step()

                for p in model.parameters():
                    p.requires_grad = True
                for p in D.parameters():
                    p.requires_grad = False

                high = model(ls,le)
                output_real = D(high)
                optimizer_G.zero_grad()
                label_real = torch.ones(output_real.size()).cuda()
                real_loss = criterion(output_real,label_real)
                error = (args.interval+2)*criterion(high.permute(1,0,2,3,4),hi)+1e-3*real_loss
                error.backward()
                loss_mse += error.mean().item()
                optimizer_G.step()

                for p in D.parameters():
                    p.requires_grad = True


        y = time.time()
        print("Time = "+str(y-x))
        print("MSE Loss = "+str(loss_mse))
        if itera%50==0 or itera==1:
            torch.save(model.state_dict(),args.model_path+args.dataset+'/'+args.var+'-S-'+str(args.scale)+'-T-'+str(args.interval)+'-E-'+str(itera)+'.pth')
