import os
from packaging import version
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import math

def concat_all_gather(tensor, world_size):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor)
    tensors_gather[dist.get_rank()] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output

class rsclLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.queue_size = 128
        data0 = torch.randn(2048, self.queue_size)
        data0 = F.normalize(data0, dim=0)
        data1 = torch.randn(2048, self.queue_size)
        data1 = F.normalize(data1, dim=0)
        data2 = torch.randn(2048, self.queue_size)
        data2 = F.normalize(data2, dim=0)
        data3 = torch.randn(2048, self.queue_size)
        data3 = F.normalize(data3, dim=0)

        self.register_buffer("queue_data_A0", data0)
        self.register_buffer("queue_ptr_A0", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_A1", data1)
        self.register_buffer("queue_ptr_A1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_A2", data2)
        self.register_buffer("queue_ptr_A2", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_A3", data3)
        self.register_buffer("queue_ptr_A3", torch.zeros(1, dtype=torch.long))

    def forward(self, feat_q, feat_k, feat_k0=None, layer=-1):
        l_pos = torch.einsum("nc,nc->n", (feat_q, feat_k)).unsqueeze(-1)
        if layer == 0:
            queue = self.queue_data_A0.clone().detach()
        elif layer == 1:
            queue = self.queue_data_A1.clone().detach()
        elif layer == 2:
            queue = self.queue_data_A2.clone().detach()
        elif layer == 3:
            queue = self.queue_data_A3.clone().detach()
        if feat_k0 != None:
            queue = torch.cat((queue,feat_k0.T),dim=1)
        l_neg2 = torch.einsum("nc,ck->nk", (feat_q, queue))
        feat_q = feat_q.view(1, -1, 2048).contiguous()
        feat_k = feat_k.view(1, -1, 2048).contiguous()
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        diagonal = torch.eye(feat_q.size(0), device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg1 = l_neg_curbatch.view(-1, feat_q.size(1))
        
        l_neg = torch.cat((l_neg1,l_neg2),dim=1)
        logits = torch.cat((l_pos, l_neg), dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=feat_q.device)
        return F.cross_entropy(logits / self.opt.nce_T, labels)

    def dequeue_and_enqueue(self, keys, layer=-1):
        if torch.distributed.is_initialized():
            keys = concat_all_gather(keys, self.opt.num_gpus)
        batch_size = keys.size(0)
        if layer == 0:
            ptr = int(self.queue_ptr_A0)
            assert self.queue_size % batch_size == 0
            self.queue_data_A0[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A0[0] = (ptr + batch_size) % self.queue_size            
        elif layer == 1:
            ptr = int(self.queue_ptr_A1)
            assert self.queue_size % batch_size == 0
            self.queue_data_A1[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A1[0] = (ptr + batch_size) % self.queue_size 
        elif layer == 2:
            ptr = int(self.queue_ptr_A2)
            assert self.queue_size % batch_size == 0
            self.queue_data_A2[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A2[0] = (ptr + batch_size) % self.queue_size 
        elif layer == 3:
            ptr = int(self.queue_ptr_A3)
            assert self.queue_size % batch_size == 0
            self.queue_data_A3[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A3[0] = (ptr + batch_size) % self.queue_size 