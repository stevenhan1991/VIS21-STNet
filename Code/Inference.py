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
import time


def inference(model,Scalar,args):
    t = 0
    low, high = Scalar.GetInferenceData()
    for i in range(0,len(low),args.interval+1):
        if i+1+args.interval<len(low):
            ls = torch.FloatTensor(low[i].reshape(1,1,Scalar.dim[0]//args.scale,Scalar.dim[1]//args.scale,Scalar.dim[2]//args.scale))
            le = torch.FloatTensor(low[args.interval+i+1].reshape(1,1,Scalar.dim[0]//args.scale,Scalar.dim[1]//args.scale,Scalar.dim[2]//args.scale))
            h = torch.FloatTensor(high[i].reshape(1,1,Scalar.dim[0],Scalar.dim[1],Scalar.dim[2]))
            if args.cuda:
                ls = ls.cuda()
                le = le.cuda()
                h = h.cuda()
            with torch.no_grad():
                    x = time.time()
                    s = model(ls,le)
                    y = time.time()
                    t += y-x
                    s = s.detach().cpu().numpy()
            for j in range(0,args.interval+2):
                print(i+1+j)
                data = s[j][0]
                data = np.asarray(data,dtype='<f')
                data = data.flatten('F')
                data.tofile(args.result_path+args.dataset+'/'+args.mode+'-T-'+str(args.interval)+'-S-'+str(args.scale)+'-'+args.dataset+'-'+args.var+'-'+"{:04d}".format(i+1+j)+'.dat',format='<f')

