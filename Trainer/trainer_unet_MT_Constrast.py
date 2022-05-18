## define trainer class and methods
import sys
import torch.nn as nn
import numpy as np
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(),'../Network'))

import torch
import unet_model as model
from torch import optim
import evaluator as eval
from trainer_unet_MT import Trainer as MTUnetTrainer


class Trainer(MTUnetTrainer):

    def __init__(self, args, Loader, device):
        '''

        :param lr:
        :param epochs:
        :param Loader:
        :param device:
        :param alpha:   EMA parameter
        :param gamma:   The weight for consistency term
        :param rampup_type: Rampup type
        :param rampup_epoch:    Rampup epoch
        :param writer:
        :param target_data:
        '''
        super(Trainer, self).__init__(args, Loader, device)


    def TrainOneEpoch(self, writer=None, max_itr=np.inf):
        '''
        Train One Epoch for MT model
        :return:
        '''

        epoch_loss = 0.
        epoch_loss_consist = 0.

        train_labeled_pred_all = []
        train_labeled_gt_all = []

        # enable network for training
        self.net.train()
        self.net_ema.train()

        itr = 0

        while True:
            ## Get next batch train samples
            FinishEpoch, data = \
                self.Loader.NextTrainBatch()
            if FinishEpoch or itr > max_itr:
                break

            train_labeled_data = data['labeled']['data']
            train_labeled_gt = data['labeled']['gt']
            # train_labeled_name = data['labeled']['name']
            train_unlabeled_data = data['unlabeled']['data']
            train_unlabeled_gt = data['unlabeled']['gt']
            # train_unlabeled_name = data['unlabeled']['name']

            ## Update weight for consistency loss
            w = self.W(epoch=float(self.epoch_cnt))
            self.weight = torch.tensor(w, device=self.device)

            #### Apply Augmentation for both labeled and unlabeled data
            #   data ~ transformed image data, gt ~ transformed segmentation ground-truth, mask ~ mask for original frame
            #   Tform ~ transformation matrix
            train_labeled_data_view1, train_labeled_gt_view1, train_labeled_mask_view1, train_labeled_Tform_view1 = \
                self.ApplyAugmentation_Mask(train_labeled_data, train_labeled_gt)
            train_labeled_data_view2, train_labeled_gt_view2, train_labeled_mask_view2, train_labeled_Tform_view2 = \
                self.ApplyAugmentation_Mask(train_labeled_data, train_labeled_gt)

            if w != 0.0:
                train_unlabeled_data_view1, train_unlabeled_gt_view1, train_unlabeled_mask_view1, train_unlabeled_Tform_view1 = \
                    self.ApplyAugmentation_Mask(train_unlabeled_data, train_unlabeled_gt)
                train_unlabeled_data_view2, train_unlabeled_gt_view2, train_unlabeled_mask_view2, train_unlabeled_Tform_view2 = \
                    self.ApplyAugmentation_Mask(train_unlabeled_data, train_unlabeled_gt)

            #### train one iteration
            # move data to GPU
            train_labeled_data_view1 = torch.tensor(train_labeled_data_view1,device=self.device, dtype=torch.float32)
            train_labeled_data_view2 = torch.tensor(train_labeled_data_view2,device=self.device, dtype=torch.float32)
            train_labeled_gt_view1 = torch.tensor(train_labeled_gt_view1,device=self.device, dtype=torch.float32)
            train_labeled_gt_view2 = torch.tensor(train_labeled_gt_view2,device=self.device, dtype=torch.float32)
            if w != 0.0:
                train_unlabeled_data_view1 = torch.tensor(train_unlabeled_data_view1,device=self.device, dtype=torch.float32)
                train_unlabeled_data_view2 = torch.tensor(train_unlabeled_data_view2,device=self.device, dtype=torch.float32)
                # train_unlabeled_gt_view1 = torch.tensor(train_unlabeled_gt_view1,device=self.device, dtype=torch.float32)
                # train_unlabeled_gt = torch.tensor(train_unlabeled_gt,device=self.device, dtype=torch.float32)

            ## forward pass for all views
            # Augment with location
            if self.Location:
                x,y=np.meshgrid(np.arange(0,train_labeled_data_view1.shape[3]), np.arange(0,train_labeled_data_view1.shape[2]))
                x = x[np.newaxis,np.newaxis,...].astype(np.float32)
                y = y[np.newaxis,np.newaxis,...].astype(np.float32)
                x = torch.tensor(np.tile(x,[train_labeled_data_view1.shape[0],1,1,1]),device=train_labeled_data_view1.device)
                y = torch.tensor(np.tile(y,[train_labeled_data_view1.shape[0],1,1,1]),device=train_labeled_data_view1.device)
                train_labeled_data_view1=torch.cat([train_labeled_data_view1, x, y], dim=1)
                train_labeled_data_view2=torch.cat([train_labeled_data_view2, x, y], dim=1)
                train_unlabeled_data_view1=torch.cat([train_unlabeled_data_view1, x, y], dim=1)
                train_unlabeled_data_view2=torch.cat([train_unlabeled_data_view2, x, y], dim=1)

            # forward pass labeled samples
            train_labeled_pred_view1 = self.net(train_labeled_data_view1)   # forward pass prediction of train labeled set view1
            train_labeled_pred_view2 = self.net_ema(train_labeled_data_view2)   # forward pass prediction of train labeled set view2
            train_unlabeled_pred_allview_sel = None
            if w != 0.0:
                # forward pass unlabeled samples
                train_unlabeled_pred_view1 = self.net(train_unlabeled_data_view1)   # forward pass prediction of train unlabeled set view1
                train_unlabeled_pred_view2 = self.net_ema(train_unlabeled_data_view2)   # forward pass prediction of train unlabeled set view2

                ## inverse transform of predictions (tensors)
                # labeled data
                # train_labeled_pred_view1_aligned = self.gt_model.invtransform_image_tensor(train_labeled_pred_view1,
                #                                                                              train_labeled_Tform_view1)
                # train_labeled_pred_view2_aligned = self.gt_model.invtransform_image_tensor(train_labeled_pred_view2,
                #                                                                              train_labeled_Tform_view2)
                # labeled_mask = torch.tensor(train_labeled_mask_view1 * train_labeled_mask_view2, dtype=torch.uint8,
                #                               device=self.device)  # the overlap mask between all views
                # unlabeled data
                train_unlabeled_pred_view1_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view1, train_unlabeled_Tform_view1)
                train_unlabeled_pred_view2_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view2, train_unlabeled_Tform_view2)
                unlabeled_mask = torch.tensor(train_unlabeled_mask_view1 * train_unlabeled_mask_view2,dtype=torch.uint8,device=self.device)   # the overlap mask between all views

                ## select overlapped pixels
                # labeled data
                # train_labeled_pred_view1_sel = torch.masked_select(train_labeled_pred_view1_aligned, labeled_mask)
                # train_labeled_pred_view2_sel = torch.masked_select(train_labeled_pred_view2_aligned, labeled_mask)
                # unlabeled data
                # train_unlabeled_pred_view1_sel = train_unlabeled_pred_view1_aligned * unlabeled_mask.unsqueeze(1)
                # train_unlabeled_pred_view2_sel = train_unlabeled_pred_view2_aligned * unlabeled_mask.unsqueeze(1)
                # train_unlabeled_pred_view2_sel = train_unlabeled_pred_view2_sel/self.Temperature    # apply temperature scaling
                train_unlabeled_pred_allview_sel = [train_unlabeled_pred_view1_aligned,
                                                    train_unlabeled_pred_view2_aligned/self.Temperature,
                                                    unlabeled_mask]

            ## compute loss
            _, loss_class, loss_consist_labeled, loss_consist_unlabeled = self.criterion(
                pred_labeled=train_labeled_pred_view1, gt_labeled=train_labeled_gt_view1[:, None, ...],
                pred_unlabeled=train_unlabeled_pred_allview_sel, weight=self.weight)

            # loss_consist = loss_consist_labeled + loss_consist_unlabeled
            loss_consist = loss_consist_unlabeled*self.weight*self.gamma
            loss = loss_class + loss_consist

            epoch_loss = epoch_loss * itr/(itr+1) + float(loss.item())/(itr+1)
            epoch_loss_consist = epoch_loss_consist * itr/(itr+1) + float(loss_consist.item())/(itr+1)

            # accumulate predictions and ground-truths
            train_labeled_pred_all.append(train_labeled_pred_view2.detach().cpu().numpy())
            train_labeled_gt_all.append(train_labeled_gt_view2.detach().cpu().numpy())

            # del train_labeled_pred, train_labeled_gt
            # torch.cuda.empty_cache()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
            self.optimizer.step()

            # update ema model
            self.update_ema_variables()

            # print batch
            print('\rFinished {}-th epoch {}-th batch'.format(self.epoch_cnt,itr),end='')

            # update global and local train steps
            self.global_step += 1
            itr += 1

        ## Evaluate current epoch performance
        train_labeled_pred_all = np.concatenate(train_labeled_pred_all)
        if self.swap_label:
            train_labeled_pred_all = np.squeeze(train_labeled_pred_all<0,axis=1).astype(float)  # binarize predictions
            train_labeled_gt_all = 1-np.concatenate(train_labeled_gt_all).astype(float)
        else:
            train_labeled_pred_all = np.squeeze(train_labeled_pred_all>0,axis=1).astype(float)  # binarize predictions
            train_labeled_gt_all = np.concatenate(train_labeled_gt_all).astype(float)
        perpix_acc = self.evaluator.perpixel_acc(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per sample iou

        self.tr_loss = epoch_loss
        self.tr_loss_consist = epoch_loss_consist
        self.tr_Acc = perpix_acc
        self.tr_macro_IoU = persamp_iou
        self.tr_micro_IoU = micro_iou

        self.epoch_cnt += 1 # increase epoch counter by 1

        return epoch_loss, perpix_acc, persamp_iou, micro_iou

    def EvalTrainSet(self, writer=None, max_itr=np.inf):
        '''
        Evaluate on train set only
        :return:
        '''

        epoch_loss = 0.
        epoch_loss_consist = 0.

        train_labeled_pred_all = []
        train_labeled_gt_all = []

        # enable network for training
        self.net.eval()
        self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch train samples
            FinishEpoch, data = \
                self.Loader.NextTrainBatch()
            if FinishEpoch or itr > max_itr:
                break

            train_labeled_data = data['labeled']['data']
            train_labeled_gt = data['labeled']['gt']
            # train_labeled_name = data['labeled']['name']
            train_unlabeled_data = data['unlabeled']['data']
            train_unlabeled_gt = data['unlabeled']['gt']
            # train_unlabeled_name = data['unlabeled']['name']

            ## Update weight for consistency loss
            w = 1.
            self.weight = torch.tensor(w, device=self.device)

            #### Apply Augmentation for both labeled and unlabeled data
            #   data ~ transformed image data, gt ~ transformed segmentation ground-truth, mask ~ mask for original frame
            #   Tform ~ transformation matrix
            train_labeled_data_view1, train_labeled_gt_view1, train_labeled_mask_view1, train_labeled_Tform_view1 = \
                self.ApplyAugmentation_Mask(train_labeled_data, train_labeled_gt)
            train_labeled_data_view2, train_labeled_gt_view2, train_labeled_mask_view2, train_labeled_Tform_view2 = \
                self.ApplyAugmentation_Mask(train_labeled_data, train_labeled_gt)

            if w != 0.0:
                train_unlabeled_data_view1, train_unlabeled_gt_view1, train_unlabeled_mask_view1, train_unlabeled_Tform_view1 = \
                    self.ApplyAugmentation_Mask(train_unlabeled_data, train_unlabeled_gt)
                train_unlabeled_data_view2, train_unlabeled_gt_view2, train_unlabeled_mask_view2, train_unlabeled_Tform_view2 = \
                    self.ApplyAugmentation_Mask(train_unlabeled_data, train_unlabeled_gt)

            #### train one iteration
            # move data to GPU
            train_labeled_data_view1 = torch.tensor(train_labeled_data_view1,device=self.device, dtype=torch.float32)
            train_labeled_data_view2 = torch.tensor(train_labeled_data_view2,device=self.device, dtype=torch.float32)
            train_labeled_gt_view1 = torch.tensor(train_labeled_gt_view1,device=self.device, dtype=torch.float32)
            train_labeled_gt_view2 = torch.tensor(train_labeled_gt_view2,device=self.device, dtype=torch.float32)
            if w != 0.0:
                train_unlabeled_data_view1 = torch.tensor(train_unlabeled_data_view1,device=self.device, dtype=torch.float32)
                train_unlabeled_data_view2 = torch.tensor(train_unlabeled_data_view2,device=self.device, dtype=torch.float32)
                # train_unlabeled_gt_view1 = torch.tensor(train_unlabeled_gt_view1,device=self.device, dtype=torch.float32)
                # train_unlabeled_gt = torch.tensor(train_unlabeled_gt,device=self.device, dtype=torch.float32)

            ## forward pass for all views
            if self.Location:
                x, y = np.meshgrid(np.arange(0, train_labeled_data_view1.shape[3]),
                                   np.arange(0, train_labeled_data_view1.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
                x = torch.tensor(np.tile(x, [train_labeled_data_view1.shape[0], 1, 1, 1]), device=train_labeled_data_view1.device)
                y = torch.tensor(np.tile(y, [train_labeled_data_view1.shape[0], 1, 1, 1]), device=train_labeled_data_view1.device)
                train_labeled_data_view1 = torch.cat([train_labeled_data_view1, x, y], dim=1)
                train_labeled_data_view2 = torch.cat([train_labeled_data_view2, x, y], dim=1)
                train_unlabeled_data_view1 = torch.cat([train_unlabeled_data_view1, x, y], dim=1)
                train_unlabeled_data_view2 = torch.cat([train_unlabeled_data_view2, x, y], dim=1)

            # forward pass labeled samples
            train_labeled_pred_view1 = self.net(train_labeled_data_view1)  # forward pass prediction of train labeled set view1
            train_labeled_pred_view2 = self.net_ema(train_labeled_data_view2)  # forward pass prediction of train labeled set view2
            train_unlabeled_pred_allview_sel = None
            if w != 0.0:
                # forward pass unlabeled samples
                train_unlabeled_pred_view1 = self.net(
                    train_unlabeled_data_view1)  # forward pass prediction of train unlabeled set view1
                train_unlabeled_pred_view2 = self.net_ema(
                    train_unlabeled_data_view2)  # forward pass prediction of train unlabeled set view2

                ## inverse transform of predictions (tensors)
                train_unlabeled_pred_view1_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view1,
                                                                                             train_unlabeled_Tform_view1)
                train_unlabeled_pred_view2_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view2,
                                                                                             train_unlabeled_Tform_view2)
                unlabeled_mask = torch.tensor(train_unlabeled_mask_view1 * train_unlabeled_mask_view2,
                                              dtype=torch.uint8,
                                              device=self.device)  # the overlap mask between all views

                ## select overlapped pixels
                train_unlabeled_pred_allview_sel = [train_unlabeled_pred_view1_aligned,
                                                    train_unlabeled_pred_view2_aligned / self.Temperature,
                                                    unlabeled_mask]

            ## compute loss
            _, loss_class, loss_consist_labeled, loss_consist_unlabeled = self.criterion(
                pred_labeled=train_labeled_pred_view1, gt_labeled=train_labeled_gt_view1[:, None, ...],
                pred_unlabeled=train_unlabeled_pred_allview_sel, weight=self.weight)

            # loss_consist = loss_consist_labeled + loss_consist_unlabeled
            loss_consist = loss_consist_unlabeled*self.weight*self.gamma
            loss = loss_class + loss_consist

            epoch_loss = epoch_loss * itr/(itr+1) + float(loss.item())/(itr+1)
            epoch_loss_consist = epoch_loss_consist * itr/(itr+1) + float(loss_consist.item())/(itr+1)

            # accumulate predictions and ground-truths
            train_labeled_pred_all.append(train_labeled_pred_view2.detach().cpu().numpy())
            train_labeled_gt_all.append(train_labeled_gt_view2.detach().cpu().numpy())

            # del train_labeled_pred, train_labeled_gt
            # torch.cuda.empty_cache()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

            # backward propagation
            # self.optimizer.zero_grad()
            # loss.backward()
            # nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
            # self.optimizer.step()

            # update ema model
            # self.update_ema_variables()

            # update global and local train steps
            self.global_step += 1
            itr += 1

        ## Evaluate current epoch performance
        train_labeled_pred_all = np.concatenate(train_labeled_pred_all)
        if self.swap_label:
            train_labeled_pred_all = np.squeeze(train_labeled_pred_all < 0, axis=1).astype(
                float)  # binarize predictions
            train_labeled_gt_all = 1 - np.concatenate(train_labeled_gt_all).astype(float)
        else:
            train_labeled_pred_all = np.squeeze(train_labeled_pred_all > 0, axis=1).astype(
                float)  # binarize predictions
            train_labeled_gt_all = np.concatenate(train_labeled_gt_all).astype(float)

        # train_labeled_pred_all = np.concatenate(train_labeled_pred_all)
        # train_labeled_pred_all = np.squeeze(train_labeled_pred_all>0,axis=1).astype(float)  # binarize predictions
        # train_labeled_gt_all = np.concatenate(train_labeled_gt_all).astype(float)
        perpix_acc = self.evaluator.perpixel_acc(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per sample iou

        self.tr_loss = epoch_loss
        self.tr_loss_consist = epoch_loss_consist
        self.tr_Acc = perpix_acc
        self.tr_macro_IoU = persamp_iou
        self.tr_micro_IoU = micro_iou

        self.epoch_cnt += 1 # increase epoch counter by 1

        return epoch_loss, perpix_acc, persamp_iou, micro_iou

