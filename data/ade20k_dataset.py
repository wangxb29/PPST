# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder2 import make_dataset
from PIL import Image
import numpy as np
import torch
from util.MattingLaplacian import compute_laplacian

class ade20kDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        root = opt.dataroot
        cache = True
        all_images = sorted(make_dataset(root, recursive=True, read_cache=cache, write_cache=False))

        self.transform = get_transform(self.opt, grayscale=False, convert=False)
        self.transform2 = get_transform(self.opt, grayscale=False, convert=False)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.image_paths = []
        self.label_paths = []
        phase = 'train'
        for p in all_images:
            if '_%s_' % phase not in p:
                continue
            if p.endswith('.jpg'):
                self.image_paths.append(p)
                self.label_paths.append(p.replace('.jpg','_seg.png'))
        self.B_size = len(self.image_paths)
        #self.use_lap = False
    def __getitem__(self, index):
        B_path = self.label_paths[index % self.B_size]
        #A_path = B_path.replace('home/xinbo/datasets', 'mnt/Datasets')
        A_path = self.image_paths[index % self.B_size]
        return self.getitem_by_path(A_path, B_path)

    def getitem_by_path(self, A_path, B_path):
        try:
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
        except OSError as err:
            print(err)
            return self.__getitem__(random.randint(0, len(self) - 1))
        A = self.transform(A_img)
        B = self.transform2(B_img)   
        if self.opt.use_lap:
            laplacian_m = compute_laplacian(A)
        else:
            laplacian_m = None    
        A = self.to_tensor(A)
        A = (A-0.5) * 2
        #B = self.to_tensor(B)
        #mask_tensor = (B-0.5) * 2
        mask_np = np.array(B)
        labels = self._mask_labels(mask_np)
        mask_tensor = torch.tensor(labels, dtype=torch.float)
        #mask_tensor = (mask_tensor - 0.5) / 0.5
        if self.opt.use_lap:
            return {'real_A': A, 'mask_A': mask_tensor, 'path_A': A_path, 'laplacian_m': laplacian_m} 
        else:
            return {'real_A': A, 'mask_A': mask_tensor, 'path_A': A_path}

    def __len__(self):
        return self.B_size
    
    def _mask_labels(self, mask_np):
        label_size = 3
        labels = np.ones((label_size, mask_np.shape[0], mask_np.shape[1]))
        masknpR = mask_np[:,:,0] 
        masknpG = mask_np[:,:,1]
        #labels[0][np.all(masknpR==90,masknpG==116,masknpB==20)] = 1.0
        labels[0][masknpR!=90] = 0.0
        labels[0][masknpG!=116] = 0.0
        labels[1] = self.greenthing(masknpR,masknpG)
        labels[2] = 1-labels[0]-labels[1]
        return labels

    def greenthing(self, masknpR, masknpG):
        labels = torch.zeros((masknpR.shape[0],masknpG.shape[1]))
        color1 = np.all((masknpR == 40 , masknpG==101),axis=0)
        labels[color1] = 1
        color1 = np.all((masknpR == 110 , masknpG==39),axis=0)
        labels[color1] = 1
        color1 = np.all((masknpR == 90 , masknpG==64),axis=0)
        labels[color1] = 1
        color1 = np.all((masknpR == 60 , masknpG==74),axis=0)
        labels[color1] = 1
        color1 = np.all((masknpR == 30 , masknpG==70),axis=0)
        labels[color1] = 1
        color1 = np.all((masknpR == 30 , masknpG==139),axis=0)
        labels[color1] = 1
        color1 = np.all((masknpR == 30 , masknpG==87),axis=0)
        labels[color1] = 1
        return labels

    def to_mytensor(self, pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

        See ``ToTensor`` for more details.

        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic_arr = np.array(pic)
        if pic_arr.ndim == 2:
            pic_arr = pic_arr[..., np.newaxis]
        img = torch.from_numpy(pic_arr.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            return img.float()  # no normalize .div(255)
        else:
            return img

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.

        See ``Normalize`` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # if not _is_tensor_image(tensor):
        #     raise TypeError("tensor is not a torch image.")
        # TODO: make efficient
        if tensor.size(0) == 1:
            tensor.sub_(mean).div_(std)
        else:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return tensor