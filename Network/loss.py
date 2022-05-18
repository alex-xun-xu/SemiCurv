# import sys
#
# sys.path.append('/vision01/TopologyLayer')

import torch
from torch.autograd import Function
import torch.nn as nn
# from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
import numpy as np
import skimage.morphology as morph
import sys
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import time


def pairwise_L2(p1, p2):
    #
    # p1, p2 - D*N, D*M
    shape1 = p1.shape[0:2]
    shape2 = p2.shape[0:2]
    CostMatDist = np.repeat(np.sum(p1.transpose() ** 2, 1, keepdims=True), shape2[1], axis=1) + \
                  np.repeat(np.sum(p2 ** 2, 0, keepdims=True), shape1[1], axis=0) - \
                  2 * p1.transpose() @ p2
    CostMatDist = (CostMatDist > 0) * CostMatDist
    return np.sqrt(CostMatDist)


def LinearAssignWarp_ConsistLoss(img1, img2):
    shape1 = img1.shape[0:2]
    shape2 = img2.shape[0:2]

    img1_vec_ts = torch.reshape(img1, [img1.shape[0] * img1.shape[1], -1])
    img1_vec = img1_vec_ts.detach().cpu().numpy()
    img2_vec_ts = torch.reshape(img2, [img2.shape[0] * img2.shape[1], -1])
    img2_vec = img2_vec_ts.detach().cpu().numpy()

    ## Build Distance Cost Matrix
    indices1 = np.unravel_index(np.arange(0, img1.shape[0] * img1.shape[1]),
                                shape1, order='C')
    indices1 = np.array([indices1[0], indices1[1]])
    indices2 = np.unravel_index(np.arange(0, img2.shape[0] * img2.shape[1]),
                                shape2, order='C')
    indices2 = np.array([indices2[0], indices2[1]])
    CostMatDist = pairwise_L2(indices1, indices2)

    CostMatColor = pairwise_L2(img1_vec.transpose(), img2_vec.transpose())

    ## Compute Linear Assignment
    gamma = 0.01
    cost = CostMatColor + gamma * CostMatDist
    # start_time = time.time()
    row_ind, col_ind = linear_sum_assignment(cost)
    # end_time = time.time()
    # elapse_time = end_time - start_time
    # print('elapsed time {}'.format(elapse_time))

    img1_vec_ts_warp = img1_vec_ts[col_ind]

    return torch.norm(img1_vec_ts_warp - img2_vec_ts, 'fro')


class BCELoss(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt, **kwargs):
        # pred= kwargs['pred']
        # gt= kwargs['gt']

        # if self.logit:
        #     pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)
        gt = gt.view(-1)
        # dice = 2.0 * (pred * gt).sum() + self.smooth
        # dice = dice / (pred.sum() + gt.sum() + self.smooth)
        # loss = 1.0 - dice
        loss = self.loss(pred, gt)
        return loss


class CrossEntLoss(nn.Module):
    def __init__(self, logit=True, smooth=1e-1, ignore_label=255):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='mean')

    def forward(self, pred, gt, **kwargs):
        # pred= kwargs['pred']
        # gt= kwargs['gt']

        # if self.logit:
        #     pred = torch.sigmoid(pred)
        smooth = 1.0
        # pred = pred.view(-1)
        # gt = gt.view(-1).type(torch.int64)
        gt = gt.type(torch.long)
        # dice = 2.0 * (pred * gt).sum() + self.smooth
        # dice = dice / (pred.sum() + gt.sum() + self.smooth)
        # loss = 1.0 - dice
        loss = self.loss(pred, gt)
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, logit=True, beta=0.05):
        '''
        constructor
        :param logit: indicate  input is logit
        :param beta: indicate  the ratio between crack and all pixels
        '''
        super().__init__()
        self.logit = logit
        self.beta = beta
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1 / self.beta))

    def forward(self, pred, gt, **kwargs):
        # pred= kwargs['pred']
        # gt= kwargs['gt']

        # if self.logit:
        #     pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)
        gt = gt.view(-1)
        # dice = 2.0 * (pred * gt).sum() + self.smooth
        # dice = dice / (pred.sum() + gt.sum() + self.smooth)
        # loss = 1.0 - dice
        loss = self.loss(pred, gt)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth

    def forward(self, pred, gt, **kwargs):
        # pred= kwargs['pred']
        # gt= kwargs['gt']

        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)
        gt = gt.view(-1)
        dice = 2.0 * (pred * gt).sum() + self.smooth
        dice = dice / (pred.sum() + gt.sum() + self.smooth)
        loss = 1.0 - dice
        return loss


class MaskedDiceLoss(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth

    def forward(self, pred, gt, mask=None, **kwargs):

        # pred= kwargs['pred']
        # gt= kwargs['gt']
        # expand mask to the same dim as pred
        if mask is None:
            mask = torch.ones_like(pred).view(-1)

        else:
            mask = mask[..., np.newaxis, np.newaxis]
            mask = torch.tensor(np.tile(mask, [1, 1, pred.shape[2], pred.shape[3]]),
                                dtype=torch.float16, device=pred.device)
            mask = mask.view(-1)

        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)[mask == 1.]
        gt = gt.view(-1)[mask == 1.]
        dice = 2.0 * (pred * gt).sum() + self.smooth
        dice = dice / (pred.sum() + gt.sum() + self.smooth)
        loss = 1.0 - dice
        return loss


class MaskedDiceLoss_SliceConsist(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth

    def forward(self, pred, gt, mask=None,
                force_strong=None, force_weak=None, **kwargs):

        # pred= kwargs['pred']
        # gt= kwargs['gt']
        # expand mask to the same dim as pred
        if mask is None:
            mask = torch.ones_like(pred)
            mask_vec = mask.view(-1)

        else:
            mask = mask[..., np.newaxis, np.newaxis]
            mask = torch.tensor(np.tile(mask, [1, 1, pred.shape[2], pred.shape[3]]),
                                dtype=torch.float16, device=pred.device)
            mask_vec = mask.view(-1)

        ## ComputeDice loss for labeled slices
        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred_vec = pred.view(-1)[mask_vec == 1.]
        gt_vec = gt.view(-1)[mask_vec == 1.]
        dice = 2.0 * (pred_vec * gt_vec).sum() + self.smooth
        dice = dice / (pred_vec.sum() + gt_vec.sum() + self.smooth)
        loss = 1.0 - dice

        ## Compute consistency loss for unlabeled slices
        if force_strong is None:
            force_strong = 1.  # consistency strength between one labeled slice and one unlabeled slice
        if force_weak is None:
            force_weak = 0.0  # consistency strength between two unlabeled slices

        loss_consist = 0.
        N_conpair = 1
        for b_i, mask_i in enumerate(mask):
            for j in range(0, len(mask_i) - 1):
                # iterate all slices until the one before the last one
                if not (mask_i[j].sum() > 1. and mask_i[j + 1].sum() > 1.):
                    # if adjacent frames are not both labeled, enforce consistency
                    if (mask_i[j].sum() > 1.) != (mask_i[j + 1].sum() > 1.):
                        # strong consistency
                        loss_consist = force_strong * torch.norm(pred[b_i, j] - pred[b_i, j + 1], 'fro') / N_conpair + (
                                    N_conpair - 1) / N_conpair * loss_consist
                        N_conpair += 1
                    else:
                        # weak consistency
                        loss_consist = force_weak * torch.norm(pred[b_i, j] - pred[b_i, j + 1], 'fro') / N_conpair + (
                                    N_conpair - 1) / N_conpair * loss_consist
                        N_conpair += 1
        ## final loss
        loss += 1e-3 * loss_consist

        return loss


class MaskedDiceLoss_SliceConsistLinAssign(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth

    def forward(self, pred, gt, mask=None,
                force_strong=None, force_weak=None, consist_weight=1e-1,
                **kwargs):

        # pred= kwargs['pred']
        # gt= kwargs['gt']
        # expand mask to the same dim as pred
        if mask is None:
            mask = torch.ones_like(pred)
            mask_vec = mask.view(-1)

        else:
            mask = mask[..., np.newaxis, np.newaxis]
            mask = torch.tensor(np.tile(mask, [1, 1, pred.shape[2], pred.shape[3]]),
                                dtype=torch.float16, device=pred.device)
            mask_vec = mask.view(-1)

        ## ComputeDice loss for labeled slices
        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred_vec = pred.view(-1)[mask_vec == 1.]
        gt_vec = gt.view(-1)[mask_vec == 1.]
        dice = 2.0 * (pred_vec * gt_vec).sum() + self.smooth
        dice = dice / (pred_vec.sum() + gt_vec.sum() + self.smooth)
        loss = 1.0 - dice

        ## Compute consistency loss for unlabeled slices
        if force_strong is None:
            force_strong = 1.  # consistency strength between one labeled slice and one unlabeled slice
        if force_weak is None:
            force_weak = 0.0  # consistency strength between two unlabeled slices

        loss_consist = 0.
        N_conpair = 1
        for b_i, mask_i in enumerate(mask):
            # if b_i >4:
            #     # only compute 4 samples
            #     break
            for j in range(0, len(mask_i) - 1):
                # iterate all slices until the one before the last one
                if not (mask_i[j].sum() > 1. and mask_i[j + 1].sum() > 1.):
                    # if adjacent frames are not both labeled, enforce consistency
                    if (mask_i[j].sum() > 1.) != (mask_i[j + 1].sum() > 1.):
                        # strong consistency
                        # compute pixel correspondence
                        img1 = pred[b_i, j]
                        img1 = torch.nn.functional.interpolate(img1[None, None, ...], scale_factor=0.5)[0, 0]
                        img2 = pred[b_i, j + 1]
                        img2 = torch.nn.functional.interpolate(img2[None, None, ...], scale_factor=0.5)[0, 0]

                        loss_consist = force_strong * LinearAssignWarp_ConsistLoss(img1, img2) / N_conpair + \
                                       (N_conpair - 1) / N_conpair * loss_consist
                        N_conpair += 1
                    else:
                        # weak consistency
                        if force_weak == 0.:
                            continue
                        # compute pixel correspondence
                        img1 = pred[b_i, j]
                        img1 = torch.nn.functional.interpolate(img1[None, None, ...], scale_factor=0.5)[0, 0]
                        img2 = pred[b_i, j + 1]
                        img2 = torch.nn.functional.interpolate(img2[None, None, ...], scale_factor=0.5)[0, 0]

                        loss_consist = force_weak * LinearAssignWarp_ConsistLoss(img1, img2) / N_conpair + \
                                       (N_conpair - 1) / N_conpair * loss_consist
                        N_conpair += 1
        ## final loss
        loss += consist_weight * loss_consist

        return loss


class MaskedDiceLoss_BoundLoss(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth

    def forward(self, pred, gt, mask=None,
                force_strong=None, force_weak=None, consist_weight=1e-1,
                **kwargs):

        # pred= kwargs['pred']
        # gt= kwargs['gt']
        # expand mask to the same dim as pred
        if mask is None:
            mask = torch.ones_like(pred)
            mask_vec = mask.view(-1)

        else:
            mask = mask[..., np.newaxis, np.newaxis]
            mask = torch.tensor(np.tile(mask, [1, 1, pred.shape[2], pred.shape[3]]),
                                dtype=torch.float16, device=pred.device)
            mask_vec = mask.view(-1)

        ## ComputeDice loss for labeled slices
        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred_vec = pred.view(-1)[mask_vec == 1.]
        gt_vec = gt.view(-1)[mask_vec == 1.]
        dice = 2.0 * (pred_vec * gt_vec).sum() + self.smooth
        dice = dice / (pred_vec.sum() + gt_vec.sum() + self.smooth)
        loss = 1.0 - dice

        ## Compute consistency loss for unlabeled slices
        if force_strong is None:
            force_strong = 1.  # consistency strength between one labeled slice and one unlabeled slice
        if force_weak is None:
            force_weak = 0.0  # consistency strength between two unlabeled slices

        loss_consist = 0.
        N_conpair = 1
        for b_i, mask_i in enumerate(mask):
            # if b_i >4:
            #     # only compute 4 samples
            #     break
            for j in range(0, len(mask_i) - 1):
                # iterate all slices until the one before the last one
                if not (mask_i[j].sum() > 1. and mask_i[j + 1].sum() > 1.):
                    # if adjacent frames are not both labeled, enforce consistency
                    if (mask_i[j].sum() > 1.) != (mask_i[j + 1].sum() > 1.):
                        # strong consistency
                        # compute pixel correspondence
                        img1 = pred[b_i, j]
                        img1 = torch.nn.functional.interpolate(img1[None, None, ...], scale_factor=0.5)[0, 0]
                        img2 = pred[b_i, j + 1]
                        img2 = torch.nn.functional.interpolate(img2[None, None, ...], scale_factor=0.5)[0, 0]

                        ## lower bound loss
                        lowerbound = (img1 > 0.6).to(torch.float32)
                        loss_lowerbound = torch.sum(lowerbound * img2) / torch.sum(lowerbound)
                        ## upper bound loss
                        upperbound = (img1 > 0.4).to(torch.float32)
                        loss_upperbound = torch.sum(upperbound * img2) / torch.sum(img2)

                        loss_consist = (loss_lowerbound + loss_upperbound) / N_conpair + \
                                       (N_conpair - 1) / N_conpair * loss_consist
                        N_conpair += 1
                    else:
                        # weak consistency
                        if force_weak == 0.:
                            continue
                        # compute pixel correspondence
                        img1 = pred[b_i, j]
                        img1 = torch.nn.functional.interpolate(img1[None, None, ...], scale_factor=0.5)[0, 0]
                        img2 = pred[b_i, j + 1]
                        img2 = torch.nn.functional.interpolate(img2[None, None, ...], scale_factor=0.5)[0, 0]

                        ## lower bound loss
                        lowerbound = img1 > 0.6
                        loss_lowerbound = torch.sum(lowerbound * img2) / torch.sum(lowerbound)
                        ## upper bound loss
                        upperbound = img1 > 0.4
                        loss_upperbound = torch.sum(upperbound * img2) / torch.sum(img2)

                        loss_consist = (loss_lowerbound + loss_upperbound) / N_conpair + \
                                       (N_conpair - 1) / N_conpair * loss_consist

                        N_conpair += 1
        ## final loss
        loss += consist_weight * loss_consist

        return loss


class DiceLoss2(nn.Module):
    def __init__(self, logit=True, smooth=1e-1):
        '''
        constructor
        :param logit: indicate  input is logit
        '''
        super().__init__()
        self.logit = logit
        self.smooth = smooth

    def forward(self, pred, gt, **kwargs):
        # pred= kwargs['pred']
        # gt= kwargs['gt']

        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)
        gt = gt.view(-1)
        dice1 = (pred * gt).sum() + self.smooth
        dice1 = dice1 / (pred.sum() + gt.sum() + self.smooth)
        dice2 = ((1 - pred) * (1 - gt)).sum() + self.smooth
        dice2 = dice2 / ((1 - pred).sum() + (1 - gt).sum() + self.smooth)
        loss = 1.0 - dice1 - dice2
        return loss


class IoULoss(nn.Module):
    def __init__(self, logit=True):
        super().__init__()
        self.logit = logit

    def forward(self, **kwargs):
        pred = kwargs['pred']
        gt = kwargs['gt']

        if self.logit:
            pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)
        gt = gt.view(-1)
        iou = (pred * gt).sum()
        iou = (iou + smooth) / (pred.sum() + gt.sum() - iou + smooth)
        loss = 1.0 - iou
        return loss


class SupTopoLoss(nn.Module):
    def __init__(self, size=[480, 320]):
        super().__init__()
        self.pdfn = LevelSetLayer2D(size=size, sublevel=False)
        self.topfn3 = PartialSumBarcodeLengths(dim=0, skip=1)

    def forward(self, pred, gt, **kwargs):
        '''
        Forward pass for unsupervised Topology loss
        :param x: Segmentation Prediction B*H*W
        :param y:
        :return:
        '''

        loss = 0.
        for b in range(pred.shape[0]):
            dgminfo_pred = self.pdfn(pred[b])
            bd_pred = dgminfo_pred[0][0]
            bd_pred[bd_pred == -float('Inf')] = 0
            # bd_pred = self.topfn3(dgminfo_pred)  # encourage a connected component
            dgminfo_gt = self.pdfn(gt[b])
            bd_gt = dgminfo_gt[0][0]
            # bd_gt = dgminfo_gt[0][1]
            bd_gt[bd_gt == -float('Inf')] = 0
            num_connectcomp = torch.sum((bd_gt[:, 0] - bd_gt[:, 1]) > 0.8)

            ## define loss
            # dgms, issublevel = dgminfo
            lengths = bd_pred[:, 0] - bd_pred[:, 1]
            # sort lengths
            sortl, indl = torch.sort(lengths, descending=True)

            loss += torch.sum(sortl[num_connectcomp.int():])
            loss += torch.sum(1 - sortl[0:num_connectcomp])
        loss = loss / (b + 1)
        return loss


# class SupTopoLoss(nn.Module):
#     '''
#     define Topology loss
#     '''
#
#     def __init__(self,img_size=[480,320]):
#         super().__init__()
#         self.pdfn = LevelSetLayer2D(size=img_size, sublevel=False)
#         self.topfn = PartialSumBarcodeLengths(dim=1, skip=1)
#         self.topfn2 = SumBarcodeLengths(dim=0)
#         self.topfn3 = PartialSumBarcodeLengths(dim=0, skip=1)
#
#     def forward(self, **kwargs):
#         '''
#         Forward pass for unsupervised Topology loss
#         :param pred: Segmentation Prediction B*H*W
#         :param gt: Segmentation Ground-Truth B*H*W
#         :return:
#         '''
#         pred= kwargs['pred']
#         gt= kwargs['gt']
#
#         loss = 0.
#         for b in range(gt.shape[0]):
#             dgminfo = self.pdfn(pred[b])
#             loss += self.topfn3(dgminfo)
#         loss /= b
#         return loss # encourage a connected component

########################################################################################################################
#
#   Multi-Task Training Losses
#
class WeightedBCELoss(nn.Module):
    def __init__(self, logit=True, beta=0.05):
        '''
        constructor
        :param logit: indicate  input is logit
        :param beta: indicate  the ratio between crack and all pixels
        '''
        super().__init__()
        self.logit = logit
        self.beta = beta
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1 / self.beta))

    def forward(self, pred, gt, **kwargs):
        # pred= kwargs['pred']
        # gt= kwargs['gt']

        # if self.logit:
        #     pred = torch.sigmoid(pred)
        smooth = 1.0
        pred = pred.view(-1)
        gt = gt.view(-1)
        # dice = 2.0 * (pred * gt).sum() + self.smooth
        # dice = dice / (pred.sum() + gt.sum() + self.smooth)
        # loss = 1.0 - dice
        loss = self.loss(pred, gt)
        return loss


class WBCE_ConsistMSELoss(nn.Module):
    '''Dice with MSE for consistency loss'''

    def __init__(self, logit=True, beta=0.05):
        super().__init__()
        self.beta = beta
        # self.dice_loss = DiceLoss(logit)
        self.loss_mse = MSELoss(logit)
        self.loss_wbce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1 / self.beta))

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_wbce = 0.
        loss_mse_labeled = 0.

        pred = pred_labeled.view(-1)
        gt = gt_labeled.view(-1)
        # dice = 2.0 * (pred * gt).sum() + self.smooth
        # dice = dice / (pred.sum() + gt.sum() + self.smooth)
        # loss = 1.0 - dice
        loss_wbce = self.loss_wbce(pred, gt)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_mse_labeled = 0.
        loss_mse_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_mse_unlabeled = self.loss_mse(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_wbce + weight * (
                    loss_mse_labeled + loss_mse_unlabeled), loss_wbce, loss_mse_labeled, loss_mse_unlabeled


class BCE_ConsistMSELoss(nn.Module):
    '''Dice with MSE for consistency loss'''

    def __init__(self, logit=True, beta=0.05):
        super().__init__()
        self.beta = beta
        # self.dice_loss = DiceLoss(logit)
        self.loss_mse = MSELoss(logit)
        self.loss_bce = nn.BCEWithLogitsLoss()

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_wbce = 0.
        loss_mse_labeled = 0.

        pred = pred_labeled.view(-1)
        gt = gt_labeled.view(-1)
        # dice = 2.0 * (pred * gt).sum() + self.smooth
        # dice = dice / (pred.sum() + gt.sum() + self.smooth)
        # loss = 1.0 - dice
        loss_bce = self.loss_bce(pred, gt)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_mse_labeled = 0.
        loss_mse_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_mse_unlabeled = self.loss_mse(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_bce + weight * (
                    loss_mse_labeled + loss_mse_unlabeled), loss_bce, loss_mse_labeled, loss_mse_unlabeled


class DiceSupTopoLoss(nn.Module):
    def __init__(self, logit=True, imgsize=[480, 320]):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.sup_top_loss = SupTopoLoss(imgsize)

    def forward(self, **kwargs):

        pred = kwargs['pred']
        gt = kwargs['gt']

        if 'rampup' in kwargs:
            rampup = kwargs['rampup']
        else:
            rampup = 0.

        loss_dice = self.dice_loss(pred, gt)
        if rampup:
            loss_sup_top = self.sup_top_loss(pred, gt)
            return loss_dice + rampup * loss_sup_top
        else:
            return loss_dice


class DiceBCELoss(nn.Module):
    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, **kwargs):
        pred = kwargs['pred']
        gt = kwargs['gt']

        # if 'rampup' in kwargs:
        #     rampup = kwargs['rampup']
        # else:
        #     rampup = 0.

        loss_dice = self.dice_loss(pred, gt)
        loss_bce = self.bce_loss(pred, gt)

        return loss_dice + loss_bce


class Dice_ConsistVarLoss(nn.Module):
    '''Dice with MSE for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_variance = VarianceLoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_var_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        # for pred, gt in zip(pred_labeled, gt_labeled):
        #     loss_dice_labeled += self.dice_loss(pred, gt)
        #     # loss_var_labeled += self.loss_variance(pred)

        ## sum up the variance loss for all M views of unlabeled data
        loss_var_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_var_unlabeled = self.loss_variance(inputs=pred_unlabeled)
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + loss_var_labeled + weight * loss_var_unlabeled, loss_dice_labeled, loss_var_labeled, loss_var_unlabeled


class Dice_ConsistMSELoss(nn.Module):
    '''Dice with MSE for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_mse = MSELoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_mse_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_mse_labeled = 0.
        loss_mse_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_mse_unlabeled = self.loss_mse(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (
                    loss_mse_labeled + loss_mse_unlabeled), loss_dice_labeled, loss_mse_labeled, loss_mse_unlabeled


class Dice_ConsistDiscMSELoss(nn.Module):
    '''Dice with Discounted MSE for consistency loss'''

    def __init__(self, logit=True, Lp=2):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.consist_loss = DiscMSELoss(logit)
        self.Lp = Lp

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            # discount = torch.abs(1-(pred_unlabeled[0]+pred_unlabeled[1])/2)**self.Lp
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_consist_labeled + loss_consist_unlabeled), loss_dice_labeled, \
               loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistSCLLoss(nn.Module):
    '''Dice with SCL consistency loss'''

    def __init__(self, logit=True, Lp=2, beta=0.5, prior=None):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.consist_loss = WeightedMSELoss(logit)
        self.Lp = Lp
        self.beta = beta

        if prior is None:
            sys.exit('prior should not be None')
        else:
            self.prior = prior  # the ratio (N_min+N_max)/N_max

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            # discount = torch.abs(1-(pred_unlabeled[0]+pred_unlabeled[1])/2)**self.Lp
            ## Obtain pseudo labels
            pseudo_unlabeled = (pred_unlabeled[1] > 0).float()
            w = self.beta ** (pseudo_unlabeled * (1 - self.prior / (1 - self.prior)))
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1], weight=w)
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_consist_labeled + loss_consist_unlabeled), loss_dice_labeled, \
               loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistMILMSELoss(nn.Module):
    '''Dice with MIL MSE for consistency loss'''

    def __init__(self, logit=True, Lp=2):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.consist_loss = MSELoss(logit)
        self.weak_mil_loss = nn.BCEWithLogitsLoss()

        self.Lp = Lp

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs[
                'pred_unlabeled']  # a list of M views of logit predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            # discount = torch.abs(1-(pred_unlabeled[0]+pred_unlabeled[1])/2)**self.Lp
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        ## Weak Supervision with MIL loss
        avg_pred = (pred_unlabeled[0] + pred_unlabeled[1]) / 2
        loss_weak_mil = self.weak_mil_loss(torch.max(avg_pred), torch.tensor(1., device=avg_pred.device))

        return loss_dice_labeled + loss_weak_mil + weight * (loss_consist_labeled + loss_consist_unlabeled), \
               loss_dice_labeled + loss_weak_mil, \
               loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistPriorMSELoss(nn.Module):
    '''Dice with Prior MSE for consistency loss'''

    def __init__(self, logit=True, Lp=2, prior=None):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.consist_loss = MSELoss(logit)
        self.prior_loss = MSELoss(False)

        if prior is None:
            sys.exit('prior should not be None')
        else:
            self.prior = prior

        self.Lp = Lp

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs[
                'pred_unlabeled']  # a list of M views of logit predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            # discount = torch.abs(1-(pred_unlabeled[0]+pred_unlabeled[1])/2)**self.Lp
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        ## Prior Loss
        # avg_pred = (pred_unlabeled[0]+pred_unlabeled[1])/2
        loss_prior = 0.1 * self.prior_loss(input1=torch.mean(torch.sigmoid(pred_unlabeled[1])),
                                           input2=torch.tensor(self.prior, device=pred_unlabeled[1].device))

        return loss_dice_labeled + loss_prior + weight * (loss_consist_labeled + loss_consist_unlabeled), \
               loss_dice_labeled + loss_prior, \
               loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistHingeMSELoss(nn.Module):
    '''Dice with Hinged MSE for consistency loss'''

    def __init__(self, logit=True, C=0.05):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.consist_loss = HingeMSELoss(logit, C)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs[
                'pred_unlabeled']  # a list of M views of logit predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            # discount = torch.abs(1-(pred_unlabeled[0]+pred_unlabeled[1])/2)**self.Lp
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_consist_labeled + loss_consist_unlabeled), \
               loss_dice_labeled, \
               loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistDiceLoss(nn.Module):
    '''Dice with Dice for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.consist_loss = DiceConsistLoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs[
                'pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_mse_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (
                loss_consist_labeled + loss_consist_unlabeled), loss_dice_labeled, loss_consist_labeled, \
               loss_consist_unlabeled


class Dice_ConsistL1Loss(nn.Module):
    '''Dice with L1 for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_consist = L1Loss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_mse_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_consist_unlabeled = self.loss_consist(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_consist_labeled + loss_consist_unlabeled), \
               loss_dice_labeled, loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistLpLoss(nn.Module):
    '''Dice with L1 for consistency loss'''

    def __init__(self, lp=1, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_consist = LpLoss(lp, logit)
        # self.lp = lp    # Lp norm

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_mse_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_consist_unlabeled = self.loss_consist(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_consist_labeled + loss_consist_unlabeled), \
               loss_dice_labeled, loss_consist_labeled, loss_consist_unlabeled


class Dice_ConsistMSE_thin_Loss(nn.Module):
    '''Dice with MSE for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_mse = MSELoss(logit)
        self.loss_bce = nn.BCEWithLogitsLoss()

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1
        # thin_gt = kwargs['gt_labeled_thin'] # ground-truth for labeled thinned cracks B*N*H*W
        device = pred_labeled.device
        ## Obtain thinned targets and BCE loss
        loss_ce_labeled_thin = 0.
        for pred, gt in zip(pred_labeled, gt_labeled):
            gt_labeled_thin = torch.tensor(morph.thin(gt[0].detach().cpu().numpy()).astype(np.uint8), device=device)
            pred_labeled_thin = torch.masked_select(pred, gt_labeled_thin)
            target_labeled_thin = torch.ones_like(pred_labeled_thin)
            ## sum up the cross entropy loss for labeled thin crack
            loss_ce_labeled_thin += self.loss_bce(pred_labeled_thin, target_labeled_thin)

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_mse_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_mse_labeled = 0.
        loss_mse_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_mse_unlabeled = self.loss_mse(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + 0.5 * loss_ce_labeled_thin + weight * (
                    loss_mse_labeled + loss_mse_unlabeled), loss_dice_labeled, loss_mse_labeled, loss_mse_unlabeled


class Dice_Contrastive_Loss(nn.Module):
    '''Dice with Contrastive for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_contrast = ContrastiveLoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_contrast_labeled = 0.
        loss_contrast_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_contrast_unlabeled = self.loss_contrast(input1=pred_unlabeled[0], input2=pred_unlabeled[1],
                                                         mask=pred_unlabeled[2])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_contrast_labeled + loss_contrast_unlabeled), loss_dice_labeled, \
               loss_contrast_labeled, loss_contrast_unlabeled


class Dice_CosineContrastive_Loss(nn.Module):
    '''Dice with Cosine similarity based Contrastive for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_contrast = CosineContrastiveLoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_contrast_labeled = 0.
        loss_contrast_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_contrast_unlabeled = self.loss_contrast(input1=pred_unlabeled[0], input2=pred_unlabeled[1],
                                                         mask=pred_unlabeled[2])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_contrast_labeled + loss_contrast_unlabeled), loss_dice_labeled, \
               loss_contrast_labeled, loss_contrast_unlabeled


class Dice_CosineContrastiveNoExp_Loss(nn.Module):
    '''Dice with Cosine similarity based Contrastive for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_contrast = CosineContrastiveLoss_NoExp(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_contrast_labeled = 0.
        loss_contrast_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_contrast_unlabeled = self.loss_contrast(input1=pred_unlabeled[0], input2=pred_unlabeled[1],
                                                         mask=pred_unlabeled[2])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_contrast_labeled + loss_contrast_unlabeled), loss_dice_labeled, \
               loss_contrast_labeled, loss_contrast_unlabeled


class Dice_L2ContrastiveNoExp_Loss(nn.Module):
    '''Dice with L2 similarity based Contrastive for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_contrast = L2ContrastiveLoss_NoExp(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_contrast_labeled = 0.
        loss_contrast_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_contrast_unlabeled = self.loss_contrast(input1=pred_unlabeled[0], input2=pred_unlabeled[1],
                                                         mask=pred_unlabeled[2])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_contrast_labeled + loss_contrast_unlabeled), loss_dice_labeled, \
               loss_contrast_labeled, loss_contrast_unlabeled


class Dice_L2Contrastive_Loss(nn.Module):
    '''Dice with L2 similarity based Contrastive for consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.loss_contrast = L2ContrastiveLoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs['pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_contrast_labeled = 0.
        loss_contrast_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_contrast_unlabeled = self.loss_contrast(input1=pred_unlabeled[0], input2=pred_unlabeled[1],
                                                         mask=pred_unlabeled[2])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (loss_contrast_labeled + loss_contrast_unlabeled), loss_dice_labeled, \
               loss_contrast_labeled, loss_contrast_unlabeled


class Dice_OrientLoss(nn.Module):
    '''Dice with Orient for supervised loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.dice_loss = DiceLoss(logit)
        self.OrientLoss = OrientLoss(logit)

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1
        pred_vec_labeled = kwargs['pred_vec_labeled']
        gt_vec_labeled = kwargs['gt_vec_labeled']

        if 'pred_unlabeled' in kwargs:
            pred_unlabeled = kwargs[
                'pred_unlabeled']  # a list of M views of predictions for unlabeled data B2*M*H*W*1
        else:
            pred_unlabeled = None

        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 0.

        ## sum up the dice & variance loss for all N views of labeled data
        loss_dice_labeled = 0.
        loss_mse_labeled = 0.

        loss_dice_labeled = self.dice_loss(pred_labeled, gt_labeled)

        ## sum up the MSE loss for 2 views of labelked/unlabeled data
        loss_consist_labeled = 0.
        loss_consist_unlabeled = 0.
        if pred_unlabeled is not None:
            loss_consist_unlabeled = self.consist_loss(input1=pred_unlabeled[0], input2=pred_unlabeled[1])
        # for pred in pred_unlabeled:
        #     loss_var_unlabeled += self.loss_variance(pred)

        return loss_dice_labeled + weight * (
                loss_consist_labeled + loss_consist_unlabeled), loss_dice_labeled, loss_consist_labeled, \
               loss_consist_unlabeled


# class OrientLoss(nn.Module):
#     '''Orientation loss'''
#
#     def __init__(self, logit=True):
#         super().__init__()
#         self.logit = logit
#
#     def forward(self, **kwargs):
#
#         pred = kwargs['pred']
#         gt = kwargs['gt']
#
#         if self.logit:

class OrientLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255, reduce=True):
        super(OrientLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce)
        self.xeloss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # log_p = F.log_softmax(inputs, dim=1)
        # loss = self.nll_loss(log_p, targets)
        loss = self.xeloss(inputs, targets)
        return loss


class VarianceLoss(nn.Module):
    '''
    Compute the variance of multiple views as loss
    '''

    def __init__(self, logit=True):
        super().__init__()
        self.logit = logit

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''

        inputs = kwargs['inputs']  # N views of predictions over P pixels N*P

        if self.logit:
            inputs = torch.sigmoid(inputs)
        # compute the mean of predictions over all pixels
        inputs_mean = torch.mean(inputs, dim=0, keepdim=True)  # 1*P

        # compute the variance
        loss_variance = torch.mean((inputs - inputs_mean) ** 2)

        return loss_variance


class L1Loss(nn.Module):
    '''L1 error loss'''

    def __init__(self, logit=True):
        super().__init__()
        # self.loss_L1 =
        self.logit = logit

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        return torch.mean(torch.abs(input1 - input2))


class LpLoss(nn.Module):
    '''Lp error loss'''

    def __init__(self, lp=1, logit=True):
        super().__init__()
        # self.loss_L1 =
        self.logit = logit
        self.lp = lp

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        ## Check Nan
        # LpNorm = torch.mean(torch.abs(input1 - input2) ** self.lp)
        # if torch.isnan(LpNorm):
        #     a=1
        # for l in LpNorm:
        #     if torch.isnan(l):
        #         a=1

        # return torch.mean(torch.abs(input1-input2)**self.lp)
        return torch.mean(torch.pow(torch.abs(input1 - input2), self.lp))


class ContrastiveLoss(nn.Module):
    '''Contrastive loss'''

    def __init__(self, logit=True, tau=0.1):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.tau = tau

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
            mask = kwargs['mask']
        else:
            input1 = kwargs['input1']  # B*(H*W)
            input2 = kwargs['input2']

        ## Constrastive Loss

        # positive pair
        sim_pos = torch.exp(-torch.mean((mask * (input1 - input2)) ** 2, dim=[1, 2, 3]) / self.tau)  # B dim vec

        # negative pairs
        sim_neg = []
        for b1, x1 in enumerate(input1):
            sim = 0.
            for b2, x2 in enumerate(input2):
                if b1 == b2:
                    continue
                sim += torch.exp(-torch.mean((x1 - x2) ** 2) / self.tau)
            sim_neg.append(sim)
        sim_neg = torch.stack(sim_neg)  # B dim vec

        # constrastive loss
        loss_contrastive = -torch.log(sim_pos / (sim_pos + sim_neg))

        return torch.mean(loss_contrastive)


class CosineContrastiveLoss_NoExp(nn.Module):
    '''Cosine Similarity based Contrastive loss'''

    def __init__(self, logit=True, tau=0.1):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.tau = tau

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
            mask = kwargs['mask']
        else:
            input1 = kwargs['input1']  # B*(H*W)
            input2 = kwargs['input2']

        ## Constrastive Loss

        # positive pair
        # sim_pos = torch.exp(-torch.mean((mask * (input1 - input2))**2, dim=[1,2,3])/self.tau)   # B dim vec

        sim_pos = torch.sqrt(torch.sum((mask * input1 * input2) ** 2, dim=[1, 2, 3])) / (
                    torch.sqrt(torch.sum((mask * input1) ** 2, dim=[1, 2, 3])) *
                    torch.sqrt(torch.sum((mask * input2) ** 2, dim=[1, 2, 3])))

        # negative pairs
        sim_neg = []
        for b1, x1 in enumerate(input1):
            sim = 0.
            for b2, x2 in enumerate(input2):
                if b1 == b2:
                    continue
                sim += torch.sqrt(torch.sum((x1 * x2) ** 2, dim=[1, 2])) / (
                            torch.sqrt(torch.sum((x1) ** 2, dim=[1, 2])) *
                            torch.sqrt(torch.sum((x2) ** 2, dim=[1, 2])))
            sim_neg.append(sim)
        sim_neg = torch.stack(sim_neg)  # B dim vec

        # constrastive loss
        loss_contrastive = -torch.log(sim_pos / (sim_pos + sim_neg))

        return torch.mean(loss_contrastive)


class L2ContrastiveLoss_NoExp(nn.Module):
    '''L2 Similarity based Contrastive loss'''

    def __init__(self, logit=True, tau=0.1):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.tau = tau

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
            mask = kwargs['mask']
        else:
            input1 = kwargs['input1']  # B*(H*W)
            input2 = kwargs['input2']

        ## Constrastive Loss

        # positive pair
        # sim_pos = torch.exp(-torch.mean((mask * (input1 - input2))**2, dim=[1,2,3])/self.tau)   # B dim vec
        sim_pos = torch.mean((mask * (input1 - input2)) ** 2, dim=[1, 2, 3])  # B dim vec

        # sim_pos = torch.sqrt(torch.sum((mask*input1*input2)**2,dim=[1,2,3]))/(torch.sqrt(torch.sum((mask*input1)**2,dim=[1,2,3]))*
        #                               torch.sqrt(torch.sum((mask*input2)**2,dim=[1,2,3])))

        # negative pairs
        sim_neg = []
        for b1, x1 in enumerate(input1):
            sim = 0.
            for b2, x2 in enumerate(input2):
                if b1 == b2:
                    continue
                # sim += torch.sqrt(torch.sum((x1*x2)**2,dim=[1,2]))/(torch.sqrt(torch.sum((x1)**2,dim=[1,2]))*
                #                       torch.sqrt(torch.sum((x2)**2,dim=[1,2])))
                sim += torch.mean(((x1 - x2)) ** 2, dim=[1, 2])
                # sim += torch.exp(-torch.mean((mask * (x1 - x2))**2, dim=[1,2,3])/self.tau)

            sim_neg.append(sim)
        sim_neg = torch.stack(sim_neg)  # B dim vec

        # constrastive loss
        loss_contrastive = -torch.log(sim_pos / (sim_pos + sim_neg))

        return torch.mean(loss_contrastive)


class L2ContrastiveLoss(nn.Module):
    '''L2 Similarity based Contrastive loss'''

    def __init__(self, logit=True, tau=0.1):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.tau = tau

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
            mask = kwargs['mask']
        else:
            input1 = kwargs['input1']  # B*(H*W)
            input2 = kwargs['input2']

        ## Constrastive Loss

        # positive pair
        # sim_pos = torch.exp(-torch.mean((mask * (input1 - input2))**2, dim=[1,2,3])/self.tau)   # B dim vec
        sim_pos = torch.exp(-torch.mean((mask * (input1 - input2)) ** 2, dim=[1, 2, 3]) / self.tau)  # B dim vec

        # sim_pos = torch.sqrt(torch.sum((mask*input1*input2)**2,dim=[1,2,3]))/(torch.sqrt(torch.sum((mask*input1)**2,dim=[1,2,3]))*
        #                               torch.sqrt(torch.sum((mask*input2)**2,dim=[1,2,3])))

        # negative pairs
        sim_neg = []
        for b1, x1 in enumerate(input1):
            sim = 0.
            for b2, x2 in enumerate(input2):
                if b1 == b2:
                    continue
                # sim += torch.sqrt(torch.sum((x1*x2)**2,dim=[1,2]))/(torch.sqrt(torch.sum((x1)**2,dim=[1,2]))*
                #                       torch.sqrt(torch.sum((x2)**2,dim=[1,2])))
                sim += torch.exp(-torch.mean(((x1 - x2)) ** 2, dim=[1, 2]) / self.tau)
                # sim += torch.exp(-torch.mean((mask * (x1 - x2))**2, dim=[1,2,3])/self.tau)

            sim_neg.append(sim)
        sim_neg = torch.stack(sim_neg)  # B dim vec

        # constrastive loss
        loss_contrastive = -torch.log(sim_pos / (sim_pos + sim_neg))

        return torch.mean(loss_contrastive)


class CosineContrastiveLoss(nn.Module):
    '''Cosine Similarity based Contrastive loss'''

    def __init__(self, logit=True, tau=0.1):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.tau = tau

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
            mask = kwargs['mask']
        else:
            input1 = kwargs['input1']  # B*(H*W)
            input2 = kwargs['input2']

        ## Constrastive Loss
        # input1 += 1e-5
        # input2 += 1e-5
        # positive pair
        # sim_pos = torch.exp(-torch.mean((mask * (input1 - input2))**2, dim=[1,2,3])/self.tau)   # B dim vec

        sim_pos = torch.sum((mask * input1 * input2) ** 2, dim=[1, 2, 3]) / (
                    torch.sqrt(torch.sum((mask * input1) ** 2, dim=[1, 2, 3])) *
                    torch.sqrt(torch.sum((mask * input2) ** 2, dim=[1, 2, 3])))

        # negative pairs
        sim_neg = []
        for b1, x1 in enumerate(input1):
            sim = 0.
            for b2, x2 in enumerate(input2):
                if b1 == b2:
                    continue
                sim += torch.sum((x1 * x2) ** 2, dim=[1, 2]) / (torch.sqrt(torch.sum((x1) ** 2, dim=[1, 2])) *
                                                                torch.sqrt(torch.sum((x2) ** 2, dim=[1, 2])))
            sim_neg.append(sim)
        sim_neg = torch.cat(sim_neg)  # B dim vec

        # constrastive loss
        loss_contrastive = -torch.log(torch.exp(sim_pos / self.tau) /
                                      (torch.exp(sim_pos / self.tau) + torch.exp(sim_neg / self.tau)))

        return torch.mean(loss_contrastive)


class MSELoss(nn.Module):
    '''Mean sqaure error loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        return self.loss_mse(input1, input2)


class HingeMSELoss(nn.Module):
    '''Hinged Mean sqaure error loss'''

    def __init__(self, logit=True, C=0.05):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.C = C

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        se = (input1 - input2) ** 2

        return torch.sum((se > self.C) * se) / (torch.sum(se > self.C) + 1e-5)


class DiscMSELoss(nn.Module):
    '''Discounted mean sqaure error loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        discount = (1 - (input1 + input2) / 2) ** 2

        return torch.mean(discount * (input1 - input2) ** 2)


class WeightedMSELoss(nn.Module):
    '''Weighted mean sqaure error loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        weight = kwargs['weight']

        return torch.mean(weight * (input1 - input2) ** 2)


class FocalMSELoss(nn.Module):
    '''Focal Mean sqaure error loss'''

    def __init__(self, logit=True, alpha=0.9, gamma=1):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.logit = logit
        self.alpha = alpha  # positive class weight
        self.gamma = gamma  # focal loss power

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        return torch.mean(torch.abs(input1 - input2) ** self.gamma * (input1 - input2) ** 2)


class DiceConsistLoss(nn.Module):
    '''Dice consistency loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.logit = logit

    def forward(self, **kwargs):

        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

            # dice = (2*torch.sum(input1*input2)+1)/(torch.sum(input1+input2)+1)

        return 1 - (2 * torch.sum(input1 * input2) + 0.1) / (torch.sum(input1 + input2) + 0.1)


class JSDivLoss(nn.Module):
    '''Jensen Shannon Divergence  loss'''

    def __init__(self, logit=True, alpha=0.9, gamma=1):
        super().__init__()
        self.loss_kld = nn.KLDivLoss(reduction='mean')
        self.logit = logit

    def forward(self, **kwargs):
        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        p1 = torch.stack([input1, 1 - input1], dim=1)
        p2 = torch.stack([input2, 1 - input2], dim=1)

        jsd = 0.5 * (self.loss_kld(torch.log(p1 + 1e-8), p2) + self.loss_kld(torch.log(p2 + 1e-8), p1))

        return jsd


class FocalJSDivLoss(nn.Module):
    '''Jensen Shannon Divergence  loss'''

    def __init__(self, logit=True, alpha=0.9, gamma=1):
        super().__init__()
        self.loss_kld = nn.KLDivLoss(reduction='none')
        self.logit = logit
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, **kwargs):
        if self.logit:
            input1 = torch.sigmoid(kwargs['input1'])
            input2 = torch.sigmoid(kwargs['input2'])
        else:
            input1 = kwargs['input1']
            input2 = kwargs['input2']

        p1 = torch.stack([input1, 1 - input1], dim=1)
        p2 = torch.stack([input2, 1 - input2], dim=1)

        jsd = 0.5 * (self.loss_kld(torch.log(p1 + 1e-8), p2) + self.loss_kld(torch.log(p2 + 1e-8), p1))
        jsd = (torch.abs(p1 - p2) ** self.gamma) * jsd

        return torch.mean(jsd)


# class KLDivLoss(nn,ModuleNotFoundError):

class TopoLoss(nn.Module):
    def __init__(self, size):
        try:
            from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
        except:
            pass
        super(TopoLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size, sublevel=False)
        # self.topfn = PartialSumBarcodeLengths(dim=1, skip=1)
        self.topfn2 = PartialSumBarcodeLengths(dim=0, skip=0)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        # return self.topfn2(dgminfo), dgminfo
        return torch.sum(dgminfo[0][0][1::, 0] - dgminfo[0][0][1::, 1]), dgminfo


from torch.autograd import Variable


class IoU_OrientLoss(nn.Module):
    '''Dice with Orient for supervised loss'''

    def __init__(self, logit=True):
        super().__init__()
        self.IoULoss = mIoULoss()
        self.OrientLoss = OrientLoss()

    def forward(self, **kwargs):
        '''
        multiple views are acceptable
        :param kwargs:
        :return:
        '''
        pred_labeled = kwargs['pred_labeled']  # a list of N views of predictions for labeled data B1*N*H*W*1
        gt_labeled = kwargs['gt_labeled']  # a list of N views of predictions for ground-truth B1*N*H*W*1
        pred_vec_labeled = kwargs['pred_vec_labeled']
        gt_vec_labeled = kwargs['gt_vec_labeled']

        loss1 = self.IoULoss(pred_labeled, gt_labeled)
        loss2 = self.OrientLoss(pred_vec_labeled, gt_vec_labeled)

        return loss1 + loss2


def to_one_hot_var(tensor, nClasses, requires_grad=False):
    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)


class mIoULoss(nn.Module):
    def __init__(self, weight=1.0, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = weight

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        # if is_target_variable:
        #     target_oneHot = to_one_hot_var(target.data, self.classes).float()
        # else:
        #     target_oneHot = to_one_hot_var(target, self.classes).float()
        target_oneHot = target

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)
        inputs = torch.sigmoid(inputs)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (self.weights * union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=1.0, size_average=True, ignore_index=255, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(size_average=size_average, ignore_index=ignore_index, reduce=reduce)
        self.weight = weight

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        loss = self.weight * self.nll_loss(log_p, targets)
        return loss
