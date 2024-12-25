import torch
from torch import nn
import torch.nn.functional as F

def loss_fn(logits, label):
    ori_loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), label.float().reshape(-1), reduction='none')
    pred_sigmoid = logits.view(-1).sigmoid()
    target = label.float().view(-1)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (0.25 * target + (1 - 0.25) * (1 - target)) * pt.pow(2)
    loss = torch.mean(ori_loss * focal_weight)
    return loss