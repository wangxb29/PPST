"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numbers
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
from sklearn.decomposition import PCA as PCA
from torch.linalg import det

def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

def slerp(a, b, r):
    d = torch.sum(a * b, dim=-1, keepdim=True)
    p = r * torch.acos(d * (1 - 1e-4))
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)
    return normalize(d)


def lerp(a, b, r):
    if type(a) == list or type(a) == tuple:
        return [lerp(aa, bb, r) for aa, bb in zip(a, b)]
    return a * (1 - r) + b * r


def madd(a, b, r):
    if type(a) == list or type(a) == tuple:
        return [madd(aa, bb, r) for aa, bb in zip(a, b)]
    return a + b * r

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=2):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if len(image_tensor.shape) == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.shape[0]):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, normalize=normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile is not False:
            tile = max(min(images_np.shape[0] // 2, 4), 1) if tile is True else tile
            images_tiled = tile_images(images_np, picturesPerRow=tile)
            return images_tiled
        else:
            return images_np

    if len(image_tensor.shape) == 2:
        assert False
        #imagce_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().numpy() if type(image_tensor) is not np.ndarray else image_tensor
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, axis=2)
    return image_numpy.astype(imtype)


def toPILImage(images, tile=None):
    if isinstance(images, list):
        if all(['tensor' in str(type(image)).lower() for image in images]):
            return toPILImage(torch.cat([im.cpu() for im in images], dim=0), tile)
        return [toPILImage(image, tile=tile) for image in images]

    if 'ndarray' in str(type(images)).lower():
        return toPILImage(torch.from_numpy(images))

    assert 'tensor' in str(type(images)).lower(), "input of type %s cannot be handled." % str(type(images))

    if tile is None:
        max_width = 2560
        tile = min(images.size(0), int(max_width / images.size(3)))

    return Image.fromarray(tensor2im(images, tile=tile))


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_spatial_code(sp):
    device = sp.device
    #sp = (sp - sp.min()) / (sp.max() - sp.min() + 1e-7)
    if sp.size(1) <= 2:
        sp = sp.repeat([1, 3, 1, 1])[:, :3, :, :]
    if sp.size(1) == 3:
        pass
    else:
        sp = sp.detach().cpu().numpy()
        X = np.transpose(sp, (0, 2, 3, 1))
        B, H, W = X.shape[0], X.shape[1], X.shape[2]
        X = np.reshape(X, (-1, X.shape[3]))
        X = X - X.mean(axis=0, keepdims=True)
        try:
            Z = PCA(3).fit_transform(X)
        except ValueError:
            print("Running PCA on the structure code has failed.")
            print("This is likely a bug of scikit-learn in version 0.18.1.")
            print("https://stackoverflow.com/a/42764378")
            print("The visualization of the structure code on visdom won't work.")
            return torch.zeros(B, 3, H, W, device=device)
        sp = np.transpose(np.reshape(Z, (B, H, W, -1)), (0, 3, 1, 2))
        sp = (sp - sp.min()) / (sp.max() - sp.min()) * 2 - 1
        sp = torch.from_numpy(sp).to(device)
    return sp


def blank_tensor(w, h):
    return torch.ones(1, 3, h, w)



class RandomSpatialTransformer:
    def __init__(self, opt, bs):
        self.opt = opt
        #self.resample_transformation(bs)


    def create_affine_transformation(self, ref, rot, sx, sy, tx, ty):
        return torch.stack([-ref * sx * torch.cos(rot), -sy * torch.sin(rot), tx,
                            -ref * sx * torch.sin(rot), sy * torch.cos(rot), ty], axis=1)

    def resample_transformation(self, bs, device, reflection=None, rotation=None, scale=None, translation=None):
        dev = device
        zero = torch.zeros((bs), device=dev)
        if reflection is None:
            #if "ref" in self.opt.random_transformation_mode:
            ref = torch.round(torch.rand((bs), device=dev)) * 2 - 1
            #else:
            #    ref = 1.0
        else:
            ref = reflection

        if rotation is None:
            #if "rot" in self.opt.random_transformation_mode:
            max_rotation = 30 * math.pi / 180
            rot = torch.rand((bs), device=dev) * (2 * max_rotation) - max_rotation
            #else:
            #    rot = 0.0
        else:
            rot = rotation

        if scale is None:
            #if "scale" in self.opt.random_transformation_mode:
            min_scale = 1.0
            max_scale = 1.0
            sx = torch.rand((bs), device=dev) * (max_scale - min_scale) + min_scale
            sy = torch.rand((bs), device=dev) * (max_scale - min_scale) + min_scale
            #else:
            #    sx, sy = 1.0, 1.0
        else:
            sx, sy = scale

        tx, ty = zero, zero

        A = torch.stack([ref * sx * torch.cos(rot), -sy * torch.sin(rot), tx,
                         ref * sx * torch.sin(rot), sy * torch.cos(rot), ty], axis=1)
        return A.view(bs, 2, 3)



    def forward_transform(self, x, size):
        if type(x) == list:
            return [self.forward_transform(xx) for xx in x]

        affine_param = self.resample_transformation(x.size(0), x.device)
        affine_grid = F.affine_grid(affine_param, (x.size(0), x.size(1), size[0], size[1]), align_corners=False)
        x = F.grid_sample(x, affine_grid, padding_mode='reflection', align_corners=False)

        return x


def apply_random_crop(x, mask, target_size, scale_range, num_crops=1, return_rect=False):
    # build grid
    B = x.size(0) * num_crops
    flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
    unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis].repeat(B, target_size, 1, 1)
    unit_grid_y = unit_grid_x.transpose(1, 2)
    unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)

    #crops = []
    x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
    mask = mask.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
    #for i in range(num_crops):
    scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
    sampling_grid = unit_grid * scale + offset
    crop = F.grid_sample(x, sampling_grid, align_corners=False)
    crop_m = F.grid_sample(mask, sampling_grid, align_corners=False)
    #crops.append(crop)
    #crop = torch.stack(crops, dim=1)
    crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))
    crop_m = crop_m.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))
    crop_m = torch.sum(crop_m,dim=(3,4))
    label = torch.zeros(B // num_crops,num_crops)
    _, label = torch.max(crop_m,dim=2)
    # for i in range(B // num_crops):
    #     for j in range(num_crops):
    #         _, label[i][j] = max(crop_m[i][j])
    return crop, label




def five_crop_noresize(A):
    Y, X = A.size(2) // 3, A.size(3) // 3
    H, W = Y * 2, X * 2
    return torch.stack([A[:, :, 0:0+H, 0:0+W],
                        A[:, :, Y:Y+H, 0:0+W],
                        A[:, :, Y:Y+H, X:X+W],
                        A[:, :, 0:0+H, X:X+W],
                        A[:, :, Y//2:Y//2+H, X//2:X//2+W]],
                       dim=1)  # return 5-dim tensor


def random_crop_noresize(A, crop_size):
    offset_y = np.random.randint(A.size(2) - crop_size[0])
    offset_x = np.random.randint(A.size(3) - crop_size[1])
    return A[:, :, offset_y:offset_y + crop_size[0], offset_x:offset_x + crop_size[1]], (offset_y, offset_x)


def random_crop_with_resize(A, crop_size):
    #size_y = np.random.randint(crop_size[0], A.size(2) + 1)
    #size_x = np.random.randint(crop_size[1], A.size(3) + 1)
    #size_y, size_x = crop_size
    size_y = max(crop_size[0], np.random.randint(A.size(2) // 3, A.size(2) + 1))
    size_x = max(crop_size[1], np.random.randint(A.size(3) // 3, A.size(3) + 1))
    offset_y = np.random.randint(A.size(2) - size_y + 1)
    offset_x = np.random.randint(A.size(3) - size_x + 1)
    crop_rect = (offset_y, offset_x, size_y, size_x)
    resized = crop_with_resize(A, crop_rect, crop_size)
    #print('resized %s to %s' % (A.size(), resized.size()))
    return resized, crop_rect


def crop_with_resize(A, crop_rect, return_size):
    offset_y, offset_x, size_y, size_x = crop_rect
    crop = A[:, :, offset_y:offset_y + size_y, offset_x:offset_x + size_x]
    resized = F.interpolate(crop, size=return_size, mode='bilinear', align_corners=False)
    #print('resized %s to %s' % (A.size(), resized.size()))
    return resized


def compute_similarity_logit(x, y, p=1, compute_interdistances=True):

    def compute_dist(x, y, p):
        if p == 2:
            return ((x - y) ** 2).sum(dim=-1).sqrt()
        else:
            return (x - y).abs().sum(dim=-1)
    C = x.shape[-1]

    if len(x.shape) == 2:
        if compute_interdistances:
            dist = torch.cdist(x[None, :, :], y[None, :, :], p)[0]
        else:
            dist = compute_dist(x, y, p)
    if len(x.shape) == 3:
        if compute_interdistances:
            dist = torch.cdist(x, y, p)
        else:
            dist = compute_dist(x, y, p)

    if p == 1:
        dist = 1 - dist / math.sqrt(C)
    elif p == 2:
        dist = 1 - 0.5 * (dist ** 2)

    return dist / 0.07


def set_diag_(x, value):
    assert x.size(-2) == x.size(-1)
    L = x.size(-2)
    identity = torch.eye(L, dtype=torch.bool, device=x.device)
    identity = identity.view([1] * (len(x.shape) - 2) + [L, L])
    x.masked_fill_(identity, value)


def to_numpy(metric_dict):
    new_dict = {}
    for k, v in metric_dict.items():
        if "numpy" not in str(type(v)):
            v = v.detach().cpu().mean().numpy()
        new_dict[k] = v
    return new_dict


def is_custom_kernel_supported():
    version_str = str(torch.version.cuda).split(".")
    major = version_str[0]
    minor = version_str[1]
    return int(major) >= 10 and int(minor) >= 1


def shuffle_batch(x):
    B = x.size(0)
    perm = torch.randperm(B, dtype=torch.long, device=x.device)
    return x[perm]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def quantize_color(x, num=64):
    return (x * num / 2).round() * (2 / num)


def resize2d_tensor(x, size_or_tensor_of_size):
    if torch.is_tensor(size_or_tensor_of_size):
        size = size_or_tensor_of_size.size()
    elif isinstance(size_or_tensor_of_size, np.ndarray):
        size = size_or_tensor_of_size.shape
    else:
        size = size_or_tensor_of_size

    if isinstance(size, tuple) or isinstance(size, list):
        return F.interpolate(x, size[-2:],
                             mode='bilinear', align_corners=False)
    else:
        raise ValueError("%s is unrecognized" % str(type(size)))


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i+1]
        one_image = Image.fromarray(tensor2im(one_t, tile=1)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)




class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            self.pad_size = kernel_size // 2
            kernel_size = [kernel_size] * dim
        else:
            raise NotImplementedError()

        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / (torch.sum(kernel))


        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )


    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        x = F.pad(input, [self.pad_size] * 4, mode="reflect")
        return self.conv(x, weight=self.weight, groups=self.groups)


# Color conversion code
def rgb2xyz(img):
    """
    RGB from 0 to 255
    :param img:
    :return:
    """
    r, g, b = torch.split(img, 1, dim=1)
    rr = r.clone().detach()
    r1 = torch.zeros_like(r)
    g1 = torch.zeros_like(g)
    b1 = torch.zeros_like(b)
    r1[rr > 0.04045] = torch.pow((r[rr > 0.04045]+0.055) / 1.055, 2.4)
    r1[rr <= 0.04045] = r[rr <= 0.04045] / 12.92
    gg = g.clone().detach()
    g1[gg > 0.04045] = torch.pow((g[gg > 0.04045]+0.055) / 1.055, 2.4)
    g1[gg <= 0.04045] = g[gg <= 0.04045] / 12.92
    bb = b.clone().detach()
    b1[bb > 0.04045] = torch.pow((b[bb > 0.04045]+0.055) / 1.055, 2.4)
    b1[bb <= 0.04045] = b[bb <= 0.04045] / 12.92
    # r = torch.where(r > 0.04045, torch.pow((r+0.055) / 1.055, 2.4), r / 12.92)
    # g = torch.where(g > 0.04045, torch.pow((g+0.055) / 1.055, 2.4), g / 12.92)
    # b = torch.where(b > 0.04045, torch.pow((b+0.055) / 1.055, 2.4), b / 12.92)
    r1 = r1 * 100
    g1 = g1 * 100
    b1 = b1 * 100
    x = r1 * 0.4124 + g1 * 0.3576 + b1* 0.1805
    y = r1 * 0.2126 + g1 * 0.7152 + b1 * 0.0722
    z = r1 * 0.0193 + g1 * 0.1192 + b1 * 0.9505
    # x = ste_round(r1 * 0.412453 + g1 * 0.357580 + b1 * 0.180423)
    # y = ste_round(r1 * 0.212671 + g1 * 0.715160 + b1 * 0.072169)
    # z = ste_round(r1 * 0.019334 + g1 * 0.119193 + b1 * 0.950227)
    return torch.cat([x,y,z], dim=1)


def xyz2rgb(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)

    x = x / 100.
    y = y / 100.
    z = z / 100.

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    r = torch.where(r > 0.0031308, 1.055 * torch.pow( r , ( 1 / 2.4 ) ) - 0.055,  12.92 * r)
    g = torch.where(g > 0.0031308, 1.055 * torch.pow( g , ( 1 / 2.4 ) ) - 0.055,  12.92 * g)
    b = torch.where(b > 0.0031308, 1.055 * torch.pow( b , ( 1 / 2.4 ) ) - 0.055,  12.92 * b)

    r = torch.round(r * 255.)
    g = torch.round(g * 255.)
    b = torch.round(b * 255.)

    return torch.cat([r,g,b], dim=1)

def xyz2lab(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    # ref_x, ref_y, ref_z = 0.95047, 1., 1.08883
    x = x / ref_x
    y = y / ref_y
    z = z / ref_z
    xx = x.clone().detach()
    x[xx > 0.008856] = torch.pow( x[xx > 0.008856], 1/3 )
    x[xx <= 0.008856] = (7.787 * x[xx <= 0.008856]) + (16 / 116.)
    yy = y.clone().detach()
    y[yy > 0.008856] = torch.pow( y[yy > 0.008856], 1/3 )
    y[yy <= 0.008856] = (7.787 * y[yy <= 0.008856]) + (16 / 116.)
    zz = z.clone().detach()
    z[zz > 0.008856] = torch.pow( z[zz > 0.008856], 1/3 )
    z[zz <= 0.008856] = (7.787 * z[zz <= 0.008856]) + (16 / 116.)
    # x = torch.where(x > 0.008856, torch.pow( x , 1/3 ), (7.787 * x) + (16 / 116.))
    # y = torch.where(y > 0.008856, torch.pow( y , 1/3 ), (7.787 * y) + (16 / 116.))
    # z = torch.where(z > 0.008856, torch.pow( z , 1/3 ), (7.787 * z) + (16 / 116.))
    l = (116. * y) - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    return torch.cat([l,a,b], dim=1)

def lab2xyz(lab):
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    l, a, b = torch.split(lab, 1, dim=1)
    y = (l + 16) / 116.
    x = a / 500. + y
    z = y - b / 200.

    y = torch.where(torch.pow( y , 3 ) > 0.008856, torch.pow( y , 3 ), ( y - 16 / 116. ) / 7.787)
    x = torch.where(torch.pow( x , 3 ) > 0.008856, torch.pow( x , 3 ), ( x - 16 / 116. ) / 7.787)
    z = torch.where(torch.pow( z , 3 ) > 0.008856, torch.pow( z , 3 ), ( z - 16 / 116. ) / 7.787)

    x = ref_x * x
    y = ref_y * y
    z = ref_z * z
    return torch.cat([x,y,z],dim=1)

def rgb2lab(rgb, opt=None):
    lab = xyz2lab(rgb2xyz(rgb))
    #l_rs = (lab[:,0:1,:,:]-opt.l_cent)/opt.l_cent
    l_rs = (lab[:,0:1,:,:]-50)/50
    l_rs = ste_round(l_rs,decimals=4)
    #ab_rs = lab[:,1:,:,:]/opt.ab_norm
    ab_rs = lab[:,1:,:,:]/110
    ab_rs = ste_round(ab_rs,decimals=4)
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def lab2rgb(lab_rs, opt=None):
    #l = lab_rs[:,0:1:,:]*opt.l_norm + opt.l_cent
    l = lab_rs[:,0:1:,:]*50 + 50
    #ab = lab_rs[:,1:,:,:]*opt.ab_norm
    ab = lab_rs[:,1:,:,:]*110
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out

def get_colorization_data(data_raw, opt):

    data_lab = rgb2lab(data_raw["real_A"], opt)

    return data_lab

def tensor2im2(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[1] == 1:
        image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (0, 2, 3, 1)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)

def ste_round(x,decimals=0):
    return torch.round(x,decimals=decimals) - x.detach() + x


def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))
 
def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)
 
def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1]),device=M.device)
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result
 
def invmat(M):
    return 1.0/det(M).to(M.device)#*adj(M)
