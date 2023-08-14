import numpy as np
import torch
import skimage 
from skimage.transform import resize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

class ScalarDataSet():
	def __init__(self,args):
		self.dataset = args.dataset
		self.var = args.var
		self.interval = args.interval
		self.scale = args.scale
		self.result_path = args.result_path
		self.mode = args.ssr
		self.train = args.train
		if self.dataset == 'Tangaroa':
			self.dim = [300,180,120]
			self.total_samples = 150
			self.vars = ['M']
			self.data_path = {'M':'../Data/tangaroa'}
			self.crop_size = [16,16,16]
		if self.dataset == 'ionization':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.vars = ['H','H+','He','PD','T']
			self.data_path = {'H':'../Data/H',
			                 'H+':'../Data/H+',
			                 'PD':'../Data/PD',
			                 "T":'../Data/T',
			                 'He':'../Data/He'}
			self.crop_size = [16,16,16]
		elif self.dataset == 'Vortex':
			self.dim = [128,128,128]
			self.vars = ['vort']
			self.crop_size = [16,16,16]
			self.total_samples = 90
			self.data_path = {'vort':'../Data/vorts'}
		elif self.dataset == 'Jet':
			self.dim = [128,128,128]
			self.vars = ['I']
			self.crop_size = [16,16,16]
			self.total_samples = 100
			self.data_path = {'I':'../Data/Jet'}
		elif self.dataset == 'Cylinder':
			self.dim = [640,240,80]
			self.crop_size = [16,16,16]
			self.total_samples = 100
			self.vars = ['320','160','640','6400']
			self.data_path = {'320':'../Data/half-cylinder-320-',
			                  '640':'../Data/half-cylinder-640-',
			                  '160':'../Data/half-cylinder-160-',
			                  '6400':'../Data/half-cylinder-6400-'}
		elif self.dataset == 'Supercurrent':
			self.dim = [256,128,32]
			self.crop_size = [16,16,8]
			self.total_samples = 200
			self.vars = ['rho','M']
			self.data_path = {'rho':'../Data/rho-',
			                  'M':'../Data/'}
		self.croptimes = args.croptimes
		if self.mode == 'ST':
			self.train_samples_v1 = [i for i in range(1,self.total_samples*2//10+1)]
			self.train_samples_v2 = [i for i in range(1,self.total_samples+1,self.interval+1)]
		else:
			self.train_samples_v1 = [i for i in range(1,self.total_samples+1)]
			self.train_samples_v2 = []
		if (self.dim[0]//self.scale == self.crop_size[0]) and (self.dim[1]//self.scale == self.crop_size[1]) and (self.dim[2]//self.scale == self.crop_size[2]):
			self.croptimes = 1

		if not os.path.exists(args.result_path+args.dataset):
			os.mkdir(args.result_path+args.dataset)

	def ReadData(self):
		self.h1 = []
		self.l1 = []
		self.h2 = []
		self.l2 = []

		for i in self.train_samples_v1:
			print(i)
			l = np.zeros((self.dim[0]//self.scale,self.dim[1]//self.scale,self.dim[2]//self.scale))
			h = np.zeros((self.dim[0],self.dim[1],self.dim[2]))
			v = np.fromfile(self.data_path[self.var]+'{:03d}'.format(i)+'.dat',dtype='<f')
			v = v.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
			

			v = 2*(v-np.min(v))/(np.max(v)-np.min(v))-1

			h = v
			self.h1.append(h)

			v = resize(v,(self.dim[0]//self.scale,self.dim[1]//self.scale,self.dim[2]//self.scale),order=3)
			v = 2*(v-np.min(v))/(np.max(v)-np.min(v))-1
			l = v
			self.l1.append(l)

		for i in self.train_samples_v2:
			print(i)
			l = np.zeros((self.dim[0]//self.scale,self.dim[1]//self.scale,self.dim[2]//self.scale))
			h = np.zeros((self.dim[0],self.dim[1],self.dim[2]))
			v = np.fromfile(self.data_path[self.var]+'{:03d}'.format(i)+'.dat',dtype='<f')
			
			v = v.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()

			v = 2*(v-np.min(v))/(np.max(v)-np.min(v))-1

			h = v
			self.h2.append(h)

			v = resize(v,(self.dim[0]//self.scale,self.dim[1]//self.scale,self.dim[2]//self.scale),order=3)
			v = 2*(v-np.min(v))/(np.max(v)-np.min(v))-1
			l = v
			self.l2.append(l)

		self.h1 = np.asarray(self.h1)
		self.l1 = np.asarray(self.l1)
		self.h2 = np.asarray(self.h2)
		self.l2 = np.asarray(self.l2)


	def GetTrainingDataST(self):
		num = len(self.h1)-self.interval-1
		ls_train = np.zeros((self.croptimes*num,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]))
		le_train = np.zeros((self.croptimes*num,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]))
		hi_train = np.zeros((self.croptimes*num,self.interval+2,self.crop_size[0]*self.scale,self.crop_size[1]*self.scale,self.crop_size[2]*self.scale))
		idx = 0
		for t in range(0,num):
			ls_,le_,hi_ = self.CropDataST(self.l1[t:t+self.interval+2],self.h1[t:t+self.interval+2])
			for j in range(0,self.croptimes):
				ls_train[idx] = ls_[j]
				le_train[idx] = le_[j]
				hi_train[idx] = hi_[j]
				idx += 1
		ls_train = torch.FloatTensor(ls_train)
		le_train = torch.FloatTensor(le_train)
		hi_train = torch.FloatTensor(hi_train)
		data = torch.utils.data.TensorDataset(ls_train,le_train,hi_train)
		train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
		return train_loader

	def GetTrainingDataS(self):
		num = len(self.h2)-self.interval-1
		ls_train = np.zeros((self.croptimes*num,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]))
		le_train = np.zeros((self.croptimes*num,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]))
		li_train = np.zeros((self.croptimes*num,self.interval+2,self.crop_size[0],self.crop_size[1],self.crop_size[2]))
		idx = 0
		for t in range(0,num):
			ls_,le_,li_ = self.CropDataS(self.l2[t:t+self.interval+2])
			for j in range(0,self.croptimes):
				ls_train[idx] = ls_[j]
				le_train[idx] = le_[j]
				li_train[idx] = li_[j]
				idx += 1
		ls_train = torch.FloatTensor(ls_train)
		le_train = torch.FloatTensor(le_train)
		li_train = torch.FloatTensor(li_train)
		print(li_train.size())
		data = torch.utils.data.TensorDataset(ls_train,le_train,li_train)
		train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
		return train_loader

	def GetInferenceData(self):
		return self.l1, self.h1

	def CropDataST(self,l,h):
		hi = []
		ls = []
		le = []
		n = 0
		while n<self.croptimes:
			if self.crop_size[0]==self.dim[0]//self.scale:
				x = 0
			else:
				x = np.random.randint(0,self.dim[0]//self.scale-self.crop_size[0])

			if self.crop_size[1] == self.dim[1]//self.scale:
				y = 0
			else:
				y = np.random.randint(0,self.dim[1]//self.scale-self.crop_size[1])
			if self.crop_size[2] == self.dim[2]//self.scale:
				z = 0
			else:
				z = np.random.randint(0,self.dim[2]//self.scale-self.crop_size[2])

			ls_ = l[0:1,x:x+self.crop_size[0],y:y+self.crop_size[1],z:z+self.crop_size[2]]
			le_ = l[self.interval+1:self.interval+2,x:x+self.crop_size[0],y:y+self.crop_size[1],z:z+self.crop_size[2]]
			hi_ = h[:,x*self.scale:(x+self.crop_size[0])*self.scale,self.scale*y:(y+self.crop_size[1])*self.scale,self.scale*z:(z+self.crop_size[2])*self.scale]
			ls.append(ls_)
			hi.append(hi_)
			le.append(le_)
			n = n+1
		return ls,le,hi

	def CropDataS(self,l):
		ls = []
		le = []
		li = []
		n = 0
		while n<self.croptimes:
			if self.crop_size[0]==self.dim[0]//self.scale:
				x = 0
			else:
				x = np.random.randint(0,self.dim[0]//self.scale-self.crop_size[0])

			if self.crop_size[1] == self.dim[1]//self.scale:
				y = 0
			else:
				y = np.random.randint(0,self.dim[1]//self.scale-self.crop_size[1])
			if self.crop_size[2] == self.dim[2]//self.scale:
				z = 0
			else:
				z = np.random.randint(0,self.dim[2]//self.scale-self.crop_size[2])

			ls_ = l[0:1,x:x+self.crop_size[0],y:y+self.crop_size[1],z:z+self.crop_size[2]]
			le_ = l[self.interval+1:self.interval+2,x:x+self.crop_size[0],y:y+self.crop_size[1],z:z+self.crop_size[2]]
			li_ = l[:,x:x+self.crop_size[0],y:y+self.crop_size[1],z:z+self.crop_size[2]]
			ls.append(ls_)
			le.append(le_)
			li.append(li_)
			n = n+1
		return ls,le,li