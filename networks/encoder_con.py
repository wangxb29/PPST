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

class StyleGAN2ResnetEncodercon(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
        parser.add_argument("--netE_num_downsampling_sp", default=3, type=int)
        parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
        parser.add_argument("--netE_nc_steepness", default=2.0, type=float)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.l2norm = Normalize(2)
        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if self.opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))
        self.mlp_01 = nn.Sequential(*[nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 256)]).cuda()
        init_net(self.mlp_01, self.init_type, self.init_gain, [])

        self.DownToSpatialCode = nn.Sequential()

        for i in range(self.opt.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True, norm="in")
            )
                
        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(self.opt.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True, norm="in"),
                ConvLayer(nchannels, self.opt.spatial_code_ch, kernel_size=1,
                          activate=False, bias=True, norm="in")
            )
        )

        self.gap = nn.AdaptiveAvgPool2d((64,64))

    def nc(self, idx):
        nc = self.opt.netE_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE_scale_capacity
        nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)
    
    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
        
        concat = torch.cat((x, xx_channel, yy_channel), dim=1)
        return concat
    
    def forward(self, x, extract_features=False,  patch_ids=None):

        feas = []
        newfeas= []
        x = self.FromRGB(x)
        newfeas.append(self.gap(x))
        for layer in self.DownToSpatialCode:
            x = layer(x)
            feas.append(x)
        sp = self.ToSpatialCode(x)
        return sp

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
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
            if debug:
                print(classname)
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

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
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
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CoordWarpNet(torch.nn.Module):
    def __init__(self, in_ch=514, out_ch=2, lat_dim=512):
        super().__init__()

        self.conv1_1 = nn.Conv1d(in_ch, lat_dim, 1)
        self.conv1_2 = nn.Conv1d(lat_dim, lat_dim, 1)
        self.conv1_3 = nn.Conv1d(lat_dim, out_ch, 1)
        self.relu = nn.ReLU()

    def forward(self, coords, lat):
        B,C,H,W = coords.shape
        coords = coords.view(B,C,-1)
        lat = lat.unsqueeze(-1).repeat(1,1,H*W)

        x = torch.cat((coords, lat), dim=1)
        x = self.relu(self.conv1_1(x))  # x = batch,512,45^2
        x = self.relu(self.conv1_2(x))
        x = self.conv1_3(x)
        x = torch.tanh(x)

        out = x.view(B,2,H,W)

        return out