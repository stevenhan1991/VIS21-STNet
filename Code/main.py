from model import *
from train import *
import math
import os
import argparse
from DataPre import ScalarDataSet
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "TSR-VFD"')
parser.add_argument('--lr_G', type=float, default=1e-4, metavar='LR',
                    help='learning rate of generative model')
parser.add_argument('--lr_D', type=float, default=4e-4, metavar='LR',
                    help='learning rate of D')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=650, metavar='N',
                    help='number of epochs to train (default: 650)')
parser.add_argument('--dataset', type=str, default='combustion', metavar='N',
                    help='the data set we used for training')
parser.add_argument('--var', type=str, default='HR', metavar='N',
                    help='the variable we used for training ')
parser.add_argument('--t', type=float, default=0.7, metavar='N',
                    help='the temperature ')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='the number of crops for a pair of data')
parser.add_argument('--interval', type=int, default=3, metavar='N',
                    help='the interval for interpolating')
parser.add_argument('--scale', type=int, default=4, metavar='N',
                    help='the downsample scale')
parser.add_argument('--model_path', type=str, default='../Exp/', metavar='N',
                    help='the path where we stored the saved model')
parser.add_argument('--result_path', type=str, default='../Result/', metavar='N',
                    help='the path where we stored the synthesized data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():

    if not os.path.exists(args.model_path+args.dataset):
        os.mkdir(args.model_path+args.dataset)
    ScalarData = ScalarDataSet(args)
    ScalarData.ReadData()
    SModel = Net(args)
    Dis = Discriminator()
    SModel.cuda()
    Dis.cuda()
    SModel.apply(weights_init_kaiming)
    Dis.apply(weights_init_kaiming)
    trainNet(SModel,Dis,args,ScalarData)

if __name__== "__main__":
    main()