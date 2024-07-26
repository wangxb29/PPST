import sys
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import util
from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ResBlock, ConvLayer, ToRGB, EqualLinear, Blur, Upsample, make_kernel
from models.networks.stylegan2_op import upfirdn2d
from torch.nn import init

class StyleGAN2ResnetEncodercol(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netE2_scale_capacity", default=1.0, type=float)
        parser.add_argument("--netE2_num_downsampling_gl1", default=3, type=int)
        parser.add_argument("--netE2_num_downsampling_gl2", default=0, type=int)
        parser.add_argument("--netE2_nc_steepness", default=2.0, type=float)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if self.opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))

        self.DownToGlobalCode1 = nn.Sequential()

        for i in range(self.opt.netE2_num_downsampling_gl1):
            self.DownToGlobalCode1.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )
        nchannels = self.nc(self.opt.netE2_num_downsampling_gl1)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.gmp = nn.AdaptiveMaxPool2d((1,1))   
        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(nchannels, self.opt.global_code_ch)
            )
        )
        self.conv1x1_9 = nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=True)
        self.conv1x1_0 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.conv1x1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True)
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)

        self.projector9 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(32, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector0 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(64, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector1 = nn.Sequential(
            #nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(128, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector2 = nn.Sequential(
            #nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(256,1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        init_net(self.projector9, 'normal', 0.02, [])
        init_net(self.projector0, 'normal', 0.02, [])
        init_net(self.projector1, 'normal', 0.02, [])
        init_net(self.projector2, 'normal', 0.02, [])
    def nc(self, idx):
        nc = self.opt.netE2_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE2_scale_capacity
        nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)

    def warp(self, fea, corr=None, resize=True, scale_factor=0):
        b,c,h,w = fea.size()
        l = h * w
        unfold = False
        if corr.dim()==3:
            B,H,W = corr.size()
        else: H,W = corr.size()
        if resize:
            s = scale_factor
            if unfold:
                sh = int(s * h / w) 
                feas = F.unfold(fea, s, stride=s)
                feas = feas.permute(0,2,1).contiguous()
                warp_fea = torch.matmul(corr, feas)
                warp_fea = warp_fea.permute(0,2,1).contiguous()    
                warp_fea = F.fold(warp_fea, (h,w) ,s, stride=s)            
            else:
                # print(fea.size())
                if h>w:
                    feas = F.adaptive_avg_pool2d(fea,(int(64*h/w),64))
                else:
                    feas = F.adaptive_avg_pool2d(fea,(64,int(64*w/h)))
                feas = feas.view(b,c,-1).permute(0,2,1)
                # print(corr.size())
                # print(feas.size())
                warp_fea = torch.matmul(corr, feas)
                if h>w:
                    warp_fea = warp_fea.permute(0,2,1).view(b,c,-1,64)
                else:
                    warp_fea = warp_fea.permute(0,2,1).view(b,c,64,-1)
                warp_fea = F.interpolate(warp_fea,scale_factor=s,mode='bilinear')
            return warp_fea
        fea = fea.view(b,c,-1).permute(0,2,1).contiguous()
        warp_feat_f = torch.matmul(corr, fea)
        if h>w:
            warp_feat = warp_feat_f.permute(0, 2, 1).view(b,c,-1,64).contiguous()
        else:
            warp_feat = warp_feat_f.permute(0, 2, 1).view(b,c,64,-1).contiguous()
        return warp_feat  
    
    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        #return x
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)  
     
    def forward(self, x=None, extract_features=False, mask=None, corrmatrix=None):
        vectors = []
        vectors_w = []
        B, c, h, w = x.size()
        if mask != None:
            projections_m = []
            projections_mw = []
            B, c, h, w = mask.size()
        x = self.FromRGB(x)  
        gap = self.gap(x)
        gmp = self.gmp(x)
        x1 = torch.cat([gap, gmp], 1) 
        conv1x1 = getattr(self, 'conv1x1_9')
        x1 = conv1x1(x1)
        projector = getattr(self, 'projector9') 
        vectors.append(F.normalize(projector(x1.view(x1.size(0),-1))))
        if corrmatrix != None:
            xx = self.warp(x,corrmatrix,resize=True,scale_factor=8)
            gap = self.gap(xx)
            gmp = self.gmp(xx)
            x1 = torch.cat([gap, gmp], 1) 
            x1 = conv1x1(x1)
            vectors_w.append(F.normalize(projector(x1.view(x1.size(0),-1))))
        if mask != None:
            gl_proo = torch.zeros((B,c,h,w)).cuda()
            for i in range(3):
                gl_proo = x * mask[:,i:i+1]
                gap = self.gap(gl_proo)
                gmp = self.gmp(gl_proo)
                x1 = torch.cat([gap, gmp], 1) 
                x1 = conv1x1(x1)             
                pro = projector(x1.view(x1.size(0),-1))
                projections_m.append(F.normalize(pro))   
                if corrmatrix != None:
                    gl_proo = xx * self.swap(mask)[:,i:i+1]
                    gap = self.gap(gl_proo)
                    gmp = self.gmp(gl_proo)
                    x1 = torch.cat([gap, gmp], 1) 
                    x1 = conv1x1(x1)             
                    pro = projector(x1.view(x1.size(0),-1))
                    projections_mw.append(F.normalize(pro))   

        for layer_id, layer in enumerate(self.DownToGlobalCode1):
            x = layer(x)
            B, c, h, w = x.size()
            gap = self.gap(x)
            gmp = self.gmp(x)
            x1 = torch.cat([gap, gmp], 1) 
            conv1x1 = getattr(self, 'conv1x1_{:d}'.format(layer_id))
            x1 = conv1x1(x1)
            projector = getattr(self, 'projector{:d}'.format(layer_id)) 
            vectors.append(F.normalize(projector(x1.view(x1.size(0),-1))))
            if corrmatrix != None:
                if layer_id <=2:
                    if layer_id <=1:
                        resize = True
                    else: resize = False
                    xx = self.warp(x,corrmatrix.detach(),resize=resize,scale_factor=2**(2-layer_id))
                else:
                    xx = layer(xx)
                gap = self.gap(xx)
                gmp = self.gmp(xx)
                x1 = torch.cat([gap, gmp], 1) 
                conv1x1 = getattr(self, 'conv1x1_{:d}'.format(layer_id))
                x1 = conv1x1(x1)
                projector = getattr(self, 'projector{:d}'.format(layer_id)) 
                vectors_w.append(F.normalize(projector(x1.view(x1.size(0),-1))))
            if mask != None: #and layer_id<=1:
                mask = self.pool(mask)
                # if layer_id==0:
                #     B, c, h, w = mask.size()
                #     mask_reshape = torch.zeros(B,h,w)
                #     mask_reshape[mask[:,1]>0.5] = 1
                #     #mask_reshape[mask[:,2]>0.5] = 2
                #     mask_reshape = mask_reshape.flatten(1, 2)
                conv1x1 = getattr(self, 'conv1x1_{:d}'.format(layer_id))
                projector = getattr(self, 'projector{:d}'.format(layer_id)) 
                gl_proo = torch.zeros((B,c,h,w)).cuda()
                for i in range(3):
                    gl_proo = x * mask[:,i:i+1]
                    gap = self.gap(gl_proo)
                    #gap = (torch.sum(gl_proo,dim=(2,3)) / (torch.sum(mask[:,i:i+1],dim=(2,3))+1e-5)).reshape(B,-1,1,1)
                    gmp = self.gmp(gl_proo)
                    x1 = torch.cat([gap, gmp], 1) 
                    x1 = conv1x1(x1)      
                    pro = projector(x1.view(x1.size(0),-1))     
                    projections_m.append(F.normalize(pro))
                    if corrmatrix != None:
                        gl_proo = xx * self.swap(mask)[:,i:i+1]
                        gap = self.gap(gl_proo)
                        #gap = (torch.sum(gl_proo,dim=(2,3)) / (torch.sum(mask[:,i:i+1],dim=(2,3))+1e-5)).reshape(B,-1,1,1)
                        gmp = self.gmp(gl_proo)
                        x1 = torch.cat([gap, gmp], 1) 
                        x1 = conv1x1(x1)      
                        pro = projector(x1.view(x1.size(0),-1))     
                        projections_mw.append(F.normalize(pro))
        if mask != None:
            if extract_features:
                return vectors, projections_m, vectors_w, projections_mw
            return vectors, projections_m, vectors_w, projections_mw
        else:            
            return vectors, vectors_w

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
    
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>