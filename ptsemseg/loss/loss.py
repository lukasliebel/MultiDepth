import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy1d(input, target, reduction=None):
    loss_function = nn.CrossEntropyLoss(reduction=reduction)
    loss = loss_function(input, target)
    return loss


def mse(input, target, reduction=None):
    loss_function = nn.MSELoss(reduction=reduction)
    loss = loss_function(input.squeeze(), target.squeeze())
    return loss


def masked_mse(input, target, ignore_index=0):
    input = input.squeeze()
    target = target.squeeze()
    mask = (target != ignore_index).type(torch.cuda.FloatTensor)
    input = input * mask

    loss_function = nn.MSELoss(reduction="sum")
    loss = loss_function(input, target)
    n_px = float(mask.sum())
    if not n_px == 0:
        # print("INFO: Number of GT pixels in batch =", n_px)
        loss = loss / float(mask.sum())
    else:
        print("WARNING: Number of GT pixels in batch = 0")
        loss = 0
    return loss


def cross_entropy2d(input, target, weight=None, reduction=None):

    if len(target.size()) > 1 and len(input.size()) > 1:
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
    else:
        raise Exception("Too few dimensions in input to loss function. Loss function cross_entropy2d only works for 2D targets. Consider using cross_entropy1d instead!")
        # return cross_entropy1d(input, target)

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
