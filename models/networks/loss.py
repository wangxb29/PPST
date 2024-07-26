import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import util
from .stylegan2_layers import Downsample


def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        #return F.softplus(-pred).view(bs, -1).mean(dim=1)
        return torch.mean((pred - 1)**2)
    else:
        #return F.softplus(pred).view(bs, -1).mean(dim=1)
        return torch.mean(pred**2)
        
class CE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, predict, target):
        n, c, h, w = target.data.shape
        predict = predict.permute(0,2,3,1).contiguous().view(n*h*w, -1)
        target = target.permute(0,2,3,1).contiguous().view(n*h*w, -1)
        #[262144, 313]
        return self.loss(predict, torch.max(target, 1)[1])

class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode

        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss = loss + new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class SoftmaxLoss(torch.nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, true):
        logits = pred / self.tau
        l = self.ce_loss(logits, true)
        
        return l

class SoftBinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        # for numerical stable reason
        self.bce_logit = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, true):
        logits = pred / self.tau
        l = self.bce_logit(logits, true)

        return l

def feature_matching_loss(xs, ys, equal_weights=False, num_layers=6):
    loss = 0.0
    for i, (x, y) in enumerate(zip(xs[:num_layers], ys[:num_layers])):
        if equal_weights:
            weight = 1.0 / min(num_layers, len(xs))
        else:
            weight = 1 / (2 ** (min(num_layers, len(xs)) - i))
        loss = loss + (x - y).abs().flatten(1).mean(1) * weight
    return loss

class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, sigmoid_tau=0.3, include_bg=False):
        super().__init__()
        self.register_buffer('weight', weight)
        self.normalization = nn.Sigmoid()
        self.sigmoid_tau = sigmoid_tau
        self.include_bg = include_bg

    def _flatten(self, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)

    def _compute_per_channel_dice(self, input, target, epsilon=1e-6, weight=None):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
        Args:
                input (torch.Tensor): NxCxSpatial input tensor
                target (torch.Tensor): NxCxSpatial target tensor
                epsilon (float): prevents division by zero
                weight (torch.Tensor): Cx1 tensor of weight per channel/class
        """

        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = self._flatten(input)
        target = self._flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        return 2 * (intersect / denominator.clamp(min=epsilon))

    def dice(self, input, target, weight):
        return self._compute_per_channel_dice(input, target, weight=weight)
    
    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input / self.sigmoid_tau)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        if self.include_bg:
            return 1. - torch.mean(per_channel_dice)
        else:
            return 1. - torch.mean(per_channel_dice[1:])
            
class IntraImageNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, target):
        num_locations = min(query.size(2) * query.size(3), self.opt.intraimage_num_locations)
        bs = query.size(0)
        patch_ids = torch.randperm(num_locations, device=query.device)

        query = query.flatten(2, 3)
        target = target.flatten(2, 3)

        # both query and target are of size B x C x N
        query = query[:, :, patch_ids]
        target = target[:, :, patch_ids]

        cosine_similarity = torch.bmm(query.transpose(1, 2), target)
        cosine_similarity = cosine_similarity.flatten(0, 1)
        target_label = torch.arange(num_locations, dtype=torch.long, device=query.device).repeat(bs)
        loss = self.cross_entropy_loss(cosine_similarity / 0.07, target_label)
        return loss


class VGG16Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_convs = torchvision.models.vgg16(pretrained=True).features
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406])[None, :, None, None] - 0.5)
        self.register_buffer('stdev',
                             torch.tensor([0.229, 0.224, 0.225])[None, :, None, None] * 2)
        self.downsample = Downsample([1, 2, 1], factor=2)

    def copy_section(self, source, start, end):
        slice = torch.nn.Sequential()
        for i in range(start, end):
            slice.add_module(str(i), source[i])
        return slice

    def vgg_forward(self, x):
        x = (x - self.mean) / self.stdev
        features = []
        for name, layer in self.vgg_convs.named_children():
            if "MaxPool2d" == type(layer).__name__:
                features.append(x)
                if len(features) == 3:
                    break
                x = self.downsample(x)
            else:
                x = layer(x)
        return features

    def forward(self, x, y):
        y = y.detach()
        loss = 0
        weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]
        #weights = [1] * 5
        total_weights = 0.0
        for i, (xf, yf) in enumerate(zip(self.vgg_forward(x), self.vgg_forward(y))):
            loss += F.l1_loss(xf, yf) * weights[i]
            total_weights += weights[i]
        return loss / total_weights


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, target, negatives):
        query = util.normalize(query.flatten(1))
        target = util.normalize(target.flatten(1))
        negatives = util.normalize(negatives.flatten(1))
        bs = query.size(0)
        sim_pos = (query * target).sum(dim=1, keepdim=True)
        sim_neg = torch.mm(query, negatives.transpose(0, 1))
        all_similarity = torch.cat([sim_pos, sim_neg], axis=1) / 0.07
        #sim_target = util.compute_similarity_logit(query, target)
        #sim_target = torch.mm(query, target.transpose(0, 1)) / 0.07
        #sim_query = util.compute_similarity_logit(query, query)
        #util.set_diag_(sim_query, -20.0)

        #all_similarity = torch.cat([sim_target, sim_query], axis=1)

        #target_label = torch.arange(bs, dtype=torch.long,
        #                            device=query.device)
        target_label = torch.zeros(bs, dtype=torch.long, device=query.device)
        loss = self.cross_entropy_loss(all_similarity,
                                       target_label)
        return loss


class ScaleInvariantReconstructionLoss(nn.Module):
    def forward(self, query, target):
        query_flat = query.transpose(1, 3)
        target_flat = target.transpose(1, 3)
        dist = 1.0 - torch.bmm(
            query_flat[:, :, :, None, :].flatten(0, 2),
            target_flat[:, :, :, :, None].flatten(0, 2),
        )

        target_spatially_flat = target.flatten(1, 2)
        num_samples = min(target_spatially_flat.size(1), 64)
        random_indices = torch.randperm(num_samples, dtype=torch.long, device=target.device)
        randomly_sampled = target_spatially_flat[:, random_indices]
        random_indices = torch.randperm(num_samples, dtype=torch.long, device=target.device)
        another_random_sample = target_spatially_flat[:, random_indices]

        random_similarity = torch.bmm(
            randomly_sampled[:, :, None, :].flatten(0, 1),
            torch.flip(another_random_sample, [0])[:, :, :, None].flatten(0, 1)
        )

        return dist.mean() + random_similarity.clamp(min=0.0).mean()

class reconstruction_loss(object):
  def __init__(self, loss):
    self.loss = loss
    if self.loss == '1st gradient':
      sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]  # x
      sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # y
      sobel_x = torch.tensor(
        sobel_x, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      sobel_y = torch.tensor(
        sobel_y, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      self.kernel1 = sobel_x
      self.kernel2 = sobel_y
    elif self.loss == '2nd gradient':
      laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
      self.kernel1 = torch.tensor(
        laplacian, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      self.kernel2 = None
    else:
      self.kernel1 = None
      self.kernel2 = None

  def compute_loss(self, input, target):
    if self.loss == 'L1':
      reconstruction_loss = torch.mean(torch.abs(input - target))  # L1

    elif self.loss == '1st gradient':
      input_dfdx = sobel_op(input, kernel=self.kernel1.to(device=torch.cuda.current_device()))
      input_dfdy = sobel_op(input, kernel=self.kernel2.to(device=torch.cuda.current_device()))
      target_dfdx = sobel_op(target, kernel=self.kernel1.to(device=torch.cuda.current_device()))
      target_dfdy = sobel_op(target, kernel=self.kernel2.to(device=torch.cuda.current_device()))
      input_gradient = torch.sqrt(torch.pow(input_dfdx, 2) +
                                  torch.pow(input_dfdy, 2))
      target_gradient = torch.sqrt(torch.pow(
        target_dfdx, 2) + torch.pow(target_dfdy, 2))
      reconstruction_loss = torch.mean(torch.abs(
        input_gradient - target_gradient))  # L1

    elif self.loss == '2nd gradient':
      input_lap = laplacian_op(input.to(device=torch.cuda.current_device()), kernel=self.kernel1.to(device=torch.cuda.current_device()))
      target_lap = laplacian_op(target.to(device=torch.cuda.current_device()), kernel=self.kernel1.to(device=torch.cuda.current_device()))
      reconstruction_loss = torch.mean(torch.abs(input_lap - target_lap))  # L1
    else:
      reconstruction_loss = None

    return reconstruction_loss

def laplacian_op(x, kernel=None):
  if kernel is None:
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    channels = x.size()[1]
    kernel = torch.tensor(laplacian,
                          dtype=torch.float32).unsqueeze(0).expand(
      1, channels, 3, 3).to(device=torch.cuda.current_device())
  return F.conv2d(x, kernel, stride=1, padding=1)
def sobel_op(x, dir=0, kernel=None):
  if kernel is None:
    if dir == 0:
      sobel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]  # x
    elif dir == 1:
      sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # y
    channels = x.size()[1]
    kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(
      1, channels, 3, 3).to(device=torch.cuda.current_device())
  return F.conv2d(x, kernel, stride=1, padding=1)