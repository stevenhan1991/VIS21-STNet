import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math
import torch.optim as optim
import copy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm")!=-1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def BuildResidualBlock(channels,dropout,kernel,depth,bias):
  layers = []
  for i in range(int(depth)):
    layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
               #nn.BatchNorm3d(channels),
               nn.ReLU(True)]
    if dropout:
      layers += [nn.Dropout(0.5)]
  layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
             #nn.BatchNorm3d(channels),
           ]
  return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
  def __init__(self,channels,dropout,kernel,depth,bias):
    super(ResidualBlock,self).__init__()
    self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

  def forward(self,x):
    out = x+self.block(x)
    return out


class Encoder(nn.Module):
	def __init__(self,inc,init_channels):
		super(Encoder,self).__init__()
		self.conv1 = nn.Conv3d(inc,init_channels,4,2,1) 
		self.rb1 = ResidualBlock(init_channels,dropout=False,kernel=3,depth=2,bias=False)
		self.conv2 = nn.Conv3d(init_channels,2*init_channels,4,2,1) 
		self.rb2 = ResidualBlock(2*init_channels,dropout=False,kernel=3,depth=2,bias=False)
		self.conv3 = nn.Conv3d(2*init_channels,4*init_channels,4,2,1)
		self.rb3 = ResidualBlock(4*init_channels,dropout=False,kernel=3,depth=2,bias=False)
		self.conv4 = nn.Conv3d(4*init_channels,8*init_channels,4,2,1) 
		self.rb4 = ResidualBlock(8*init_channels,dropout=False,kernel=3,depth=2,bias=False)

	def forward(self,x):
		x1 = F.relu(self.conv1(x))
		x1 = self.rb1(x1)
		x2 = F.relu(self.conv2(x1))
		x2 = self.rb2(x2)
		x3 = F.relu(self.conv3(x2))
		x3 = self.rb3(x3)
		x4 = F.relu(self.conv4(x3))
		x4 = self.rb4(x4)
		return [x1,x2,x3,x4]

def voxel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class VoxelShuffle(nn.Module):
	def __init__(self,inchannels,outchannels,upscale_factor):
		super(VoxelShuffle,self).__init__()
		self.upscale_factor = upscale_factor
		self.conv = nn.Conv3d(inchannels,outchannels*(upscale_factor**3),3,1,1)

	def forward(self,x):
		x = voxel_shuffle(self.conv(x),self.upscale_factor)
		return x


class D(nn.Module):
	def __init__(self):
		super(D,self).__init__()
		self.conv1 = nn.Conv3d(1,64,4,2,1)
		self.conv2 = nn.Conv3d(64,128,4,2,1)
		self.lstm = LSTMCell(128,128,3)
		self.conv3 = nn.Conv3d(128,1,4,2,1)


	def forward(self,x):
		num = x.size()[0]
		h = None
		c = None
		comps = []
		for i in range(num):
			f = F.relu(self.conv1(x[i:i+1,:,:,:,:,]))
			f = F.relu(self.conv2(f))
			h,c = self.lstm(f,h,c)
			f = self.conv3(f)
			f = F.avg_pool3d(f,f.size()[2:]).view(-1)
			comps.append(f)
		comps = torch.stack(comps)
		comps = torch.squeeze(comps)
		return comps


class Decoder(nn.Module):
	def __init__(self,outc,init_channels):
		super(Decoder,self).__init__()
		self.deconv41 = nn.ConvTranspose3d(init_channels,init_channels//2,4,2,1) 
		self.conv_u41 = nn.Conv3d(init_channels,init_channels//2,3,1,1)
		self.deconv31 = nn.ConvTranspose3d(init_channels//2,init_channels//4,4,2,1) 
		self.conv_u31 = nn.Conv3d(init_channels//2,init_channels//4,3,1,1)
		self.deconv21 = nn.ConvTranspose3d(init_channels//4,init_channels//8,4,2,1)
		self.conv_u21 = nn.Conv3d(init_channels//4,init_channels//8,3,1,1)
		self.deconv11 = nn.ConvTranspose3d(init_channels//8,init_channels//16,4,2,1)
		self.conv_u11 = nn.Conv3d(init_channels//16,outc,3,1,1)

	def forward(self,features):
		u11 = F.relu(self.deconv41(features[-1]))
		u11 = F.relu(self.conv_u41(torch.cat((features[-2],u11),dim=1)))
		u21 = F.relu(self.deconv31(u11))
		u21 = F.relu(self.conv_u31(torch.cat((features[-3],u21),dim=1)))
		u31 = F.relu(self.deconv21(u21)) 
		u31 = F.relu(self.conv_u21(torch.cat((features[-4],u31),dim=1)))
		u41 = F.relu(self.deconv11(u31))
		out = self.conv_u11(u41)
		out = torch.tanh(out)
		return out


class UNet(nn.Module):
	def __init__(self,inc,outc,init_channels):
		super(UNet,self).__init__()
		self.encoder = Encoder(inc,init_channels)
		self.decoder = Decoder(outc,init_channels*8)

	def forward(self,x):
		return self.decoder(self.encoder(x))

	def Encode(self,x):
		return self.encoder(x)

	def Decode(self,x):
		return self.decoder(x)


class RDB(nn.Module):
	def __init__(self,init_channels,outchannels,active='relu'):
		super(RDB,self).__init__()
		self.conv1 = nn.Conv3d(init_channels,2*init_channels,3,1,1)
		self.conv2 = nn.Conv3d(3*init_channels,4*init_channels,3,1,1)
		self.conv3 = nn.Conv3d(4*init_channels+2*init_channels+init_channels,outchannels,3,1,1)
		if active=='relu':
			self.ac = nn.ReLU(inplace=True)
		elif active == 'tanh':
			self.ac = nn.Tanh()
		elif active == 'siren':
			self.ac = Siren()
		elif active == 'switch':
			self.ac = Switch()

	def forward(self,x):
		x1 = self.ac(self.conv1(x))
		x2 = self.ac(self.conv2(torch.cat((x,x1),dim=1)))
		x3 = self.ac(self.conv3(torch.cat((x,x1,x2),dim=1)))
		return x3


class Upscale(nn.Module):
	def __init__(self,inc,outc):
		super(Upscale,self).__init__()
		self.deconv = VoxelShuffle(inc,outc,2)
		self.up = VoxelShuffle(inc,outc,2)
		self.conv = nn.Conv3d(outc,outc,3,1,1)
		self.conv1 = nn.Conv3d(outc,outc,3,1,1)
		self.conv2 = nn.Conv3d(2*outc,outc,3,1,1)


	def forward(self,x):
		x1 = F.relu(self.deconv(x))
		x1 = torch.sigmoid(self.conv(self.up(x)))*x1
		x2 = F.relu(self.conv1(x1))
		x3 = self.conv2(torch.cat((x1,x2),dim=1))
		return x3


class FeatureExtractor(nn.Module):
	def __init__(self,inc):
		super(FeatureExtractor,self).__init__()
		
		self.s = nn.Sequential(*[RDB(inc,16),
			                      RDB(16,32),
			                      RDB(32,64),
			                      RDB(64,64)])
	def forward(self,x):
		return self.s(x)



### upscale for 4 times along each dimension
class Net(nn.Module):
	def __init__(self,args):
		super(Netv2,self).__init__()
		self.rb1 = RDB(2,32)
		self.rb2 = RDB(32,64)
		self.rb3 = RDB(64,128)
		self.rb4 = RDB(128,256)

		self.upscaler = nn.Sequential(*[Upscale(256,128),
			                            nn.ReLU(True),
			                            Upscale(128,64),
			                            nn.ReLU(True),
			                            nn.Conv3d(64,2+args.interval,3,1,1)
			                            ])

	def forward(self,s,e):
		x = self.rb1(torch.cat((s,e),dim=1))
		x = self.rb2(x)
		x = self.rb3(x)
		x = self.rb4(x)
		x = self.upscaler(x)
		return x.permute(1,0,2,3,4)



class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.c1 = Block(inchannels=1,outchannels=16,dropout=False,kernel=3,bias=False,depth=2,mode='down',factor=2)
		self.c2 = Block(inchannels=16,outchannels=32,dropout=False,kernel=3,bias=False,depth=2,mode='down',factor=2)
		self.c3 = Block(inchannels=32,outchannels=64,dropout=False,kernel=3,bias=False,depth=2,mode='down',factor=2)
		self.c4 = Block(inchannels=64,outchannels=128,dropout=False,kernel=3,bias=False,depth=2,mode='down',factor=2)
		self.c5 = Block(inchannels=128,outchannels=1,dropout=False,kernel=3,bias=False,depth=2,mode='down',factor=2)
		self.p1 = nn.AdaptiveAvgPool3d(1)

	def forward(self,x):
		features = []
		x = self.c1(x)
		features.append(x)
		x = self.c2(x)
		features.append(x)
		x = self.c3(x)
		features.append(x)
		x = self.c4(x)
		features.append(x)
		x = self.c5(x)
		features.append(x)
		score = self.p1(x)
		return features, score.view(-1)



