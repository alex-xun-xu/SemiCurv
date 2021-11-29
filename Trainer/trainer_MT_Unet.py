## define trainer class and methods
import sys
import tqdm
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from PIL import Image
import socket
import pathlib
import scipy
from scipy import ndimage

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(),'../Network'))

import torch
import unet_model as model
import loss as Loss
from Trainer.trainer_Unet import Trainer as UnetTrainer

class Trainer(UnetTrainer):

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

        self.rampup_epoch = args.RampupEpoch    # rampup epoch
        self.gamma = args.Gamma  # The weight for consistency term
        self.alpha = args.Alpha  # ema moving average weight
        self.rampup_type = args.RampupType  # rampup type
        self.lp = args.lp   # lp norm for consistency loss
        self.W = self.WeightRampup(self.rampup_type,self.rampup_epoch)    # rampup weight
        self.HingeC = args.HingeC
        self.mean_pos = Loader.mean_pos
        self.Temperature = args.Temperature # temperature applied to teacher's output


    def DefineNetwork(self, net_name, loss_name):
        '''
        define backbone network
        :param loss:  specified loss to trian the network
        :return:
        '''
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N

        if net_name == 'Unet':
            self.net = model.UNet(n_channels=3, n_classes=1, bilinear=False)
        elif net_name == 'ResUnet':
            if self.Location:
                self.net = model.ResUNet(n_channels=3+2, n_classes=1, bilinear=False)
            else:
                self.net = model.ResUNet(n_channels=3, n_classes=1, bilinear=False)
        elif net_name == 'ResUnet_Location':
            self.net = model.ResUNet_Location(n_channels=3 + 2, n_classes=1, bilinear=False)
        elif net_name == 'ResUnet_SinusoidLocation':
            self.net = model.ResUNet_SinusoidLocation(n_channels=3 + 2, n_classes=1, bilinear=False, SinPeriod=self.SinPeriod)
        elif net_name == 'LK34MTL':
            self.net = model.LinkNet34MTL()
        elif net_name == 'LK34':
            if self.Location:
                self.net = model.LinkNet34(in_channels=3+2, num_classes=1)
            else:
                self.net = model.LinkNet34(in_channels=3, num_classes=1)
        elif net_name == 'LK34_SinusoidLocation':
            # if self.Location:
                self.net = model.LinkNet34_SinusoidLocation(in_channels=3 + 2, num_classes=1, SinPeriod=self.SinPeriod)
            # else:
            #     self.net = model.LinkNet34_SinusoidLocation(in_channels=3, num_classes=1, SinPeriod=self.SinPeriod)

        self.net_ema = copy.deepcopy(self.net)

        self.net.to(device=self.device)
        self.net_ema.to(device=self.device)

        # detach all variables in teacher model
        self.loss_name = loss_name

        for param in self.net_ema.parameters():
            param.detach_()

        if loss_name == 'BCEloss':
            # self.criterion = loss.DiceCoeff()
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_name == 'Diceloss':
            self.criterion = Loss.DiceLoss(logit=True)
        elif loss_name == 'IoUloss':
            self.criterion = Loss.IoULoss(logit=True)
        elif loss_name == 'Dice+SupTopo_loss':
            self.criterion = Loss.DiceSupTopoLoss(logit=True, imgsize=self.masksize)
        elif loss_name == 'Dice+BCE_loss':
            self.criterion = Loss.DiceBCELoss(logit=True)
        elif loss_name == 'Dice+ConsistMSE_loss':
            self.criterion = Loss.Dice_ConsistMSELoss(logit=True)
        elif loss_name == 'Dice+ConsistMSEall_loss':
            self.criterion = Loss.Dice_ConsistMSELoss(logit=True)
        elif loss_name == 'Dice+ConsistDiscMSE_loss':
            self.criterion = Loss.Dice_ConsistMSELoss(logit=True)
        elif loss_name == 'Dice+ConsistSCL_loss':
            self.criterion = Loss.Dice_ConsistSCLLoss(logit=True, prior=self.mean_pos)
        elif loss_name == 'Dice+ConsistDice_loss':
            self.criterion = Loss.Dice_ConsistDiceLoss(logit=True)
        elif loss_name == 'Dice+ConsistMILMSE_loss':
            self.criterion = Loss.Dice_ConsistMILMSELoss(logit=True)
        elif loss_name == 'Dice+ConsistPriorMSE_loss':
            self.criterion = Loss.Dice_ConsistPriorMSELoss(logit=True, prior=self.mean_pos)
        elif loss_name == 'Dice+ConsistHingeMSE_loss':
            self.criterion = Loss.Dice_ConsistHingeMSELoss(logit=True, C=self.HingeC)
        elif loss_name == 'Dice+ConsistVar_loss':
            self.criterion = Loss.Dice_ConsistVarLoss(logit=True)
        elif loss_name == 'Dice+ConsistMSE+thin_loss':
            self.criterion = Loss.Dice_ConsistMSE_thin_Loss(logit=True)
        elif loss_name == 'Dice+ConsistL1_loss':
            self.criterion = Loss.Dice_ConsistL1Loss(logit=True)
        elif loss_name == 'Dice+ConsistLp_loss':
            self.criterion = Loss.Dice_ConsistLpLoss(lp=self.lp, logit=True)
        elif loss_name == 'Dice+ConsistDice_loss':
            self.criterion = Loss.Dice_ConsistDiceLoss(logit=True)
        elif loss_name == 'Dice+Contrastive_loss':
            self.criterion = Loss.Dice_Contrastive_Loss(logit=True)
        elif loss_name == 'Dice+CosineContrastive_loss':
            self.criterion = Loss.Dice_CosineContrastive_Loss(logit=True)
        elif loss_name == 'Dice+CosineContrastiveNoExp_loss':
            self.criterion = Loss.Dice_CosineContrastiveNoExp_Loss(logit=True)
        elif loss_name == 'Dice+L2ContrastiveNoExp_loss':
            self.criterion = Loss.Dice_L2ContrastiveNoExp_Loss(logit=True)
        elif loss_name == 'Dice+L2Contrastive_loss':
            self.criterion = Loss.Dice_L2Contrastive_Loss(logit=True)

        self.DiceLoss = Loss.DiceLoss(logit=True)
        try: self.TopoLoss = Loss.TopoLoss(size=[512,512])
        except:
            self.TopoLoss = None
            pass
        # self.MSELoss = loss.MSELoss(logit=True)
        # self.JSDLoss = loss.JSDivLoss(logit=True)
        # self.FocalMSELoss = loss.FocalMSELoss(logit=True)

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
                if self.loss_name == 'Dice+ConsistMSEall_loss':

                    train_labeled_pred_view1_aligned = self.gt_model.invtransform_image_tensor(train_labeled_pred_view1,
                                                                                                 train_labeled_Tform_view1)
                    train_labeled_pred_view2_aligned = self.gt_model.invtransform_image_tensor(train_labeled_pred_view2,
                                                                                                 train_labeled_Tform_view2)
                    labeled_mask = torch.tensor(train_labeled_mask_view1 * train_labeled_mask_view2, dtype=torch.uint8,
                                                  device=self.device)  # the overlap mask between all views
                # unlabeled data
                train_unlabeled_pred_view1_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view1, train_unlabeled_Tform_view1)
                train_unlabeled_pred_view2_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view2, train_unlabeled_Tform_view2)
                unlabeled_mask = torch.tensor(train_unlabeled_mask_view1 * train_unlabeled_mask_view2,dtype=torch.uint8,device=self.device)   # the overlap mask between all views

                ## select overlapped pixels
                # labeled data
                if self.loss_name == 'Dice+ConsistMSEall_loss':

                    train_labeled_pred_view1_sel = torch.masked_select(train_labeled_pred_view1_aligned, labeled_mask)
                    train_labeled_pred_view2_sel = torch.masked_select(train_labeled_pred_view2_aligned, labeled_mask)
                    train_labeled_pred_view2_sel = train_labeled_pred_view2_sel / self.Temperature  # apply temperature scaling
                    # train_labeled_pred_allview_sel = [train_labeled_pred_view1_sel, train_labeled_pred_view2_sel]

                # unlabeled data
                train_unlabeled_pred_view1_sel = torch.masked_select(train_unlabeled_pred_view1_aligned, unlabeled_mask==1)
                train_unlabeled_pred_view2_sel = torch.masked_select(train_unlabeled_pred_view2_aligned, unlabeled_mask==1)
                train_unlabeled_pred_view2_sel = train_unlabeled_pred_view2_sel/self.Temperature    # apply temperature scaling
                # train_unlabeled_pred_allview_sel = [train_unlabeled_pred_view1_sel,train_unlabeled_pred_view2_sel]

                if self.loss_name == 'Dice+ConsistMSEall_loss':

                    train_all_pred_allview_sel = [torch.cat((train_labeled_pred_view1_sel,train_unlabeled_pred_view1_sel)),
                                                  torch.cat((train_labeled_pred_view2_sel,train_unlabeled_pred_view2_sel))]

                else:
                    train_all_pred_allview_sel = [train_unlabeled_pred_view1_sel,train_unlabeled_pred_view2_sel]

            ## compute loss
            _, loss_class, loss_consist_labeled, loss_consist_unlabeled = self.criterion(
                pred_labeled=train_labeled_pred_view1, gt_labeled=train_labeled_gt_view1[:, None, ...],
                pred_unlabeled=train_all_pred_allview_sel, weight=self.weight)

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

        # enable network for evaluation
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
                train_unlabeled_pred_view1 = self.net(train_unlabeled_data_view1)  # forward pass prediction of train unlabeled set view1
                train_unlabeled_pred_view2 = self.net_ema(train_unlabeled_data_view2)  # forward pass prediction of train unlabeled set view2

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
            unlabeled_mask = torch.tensor(train_unlabeled_mask_view1 * train_unlabeled_mask_view2,dtype=torch.bool,device=self.device)   # the overlap mask between all views

            ## select overlapped pixels
            # labeled data
            # train_labeled_pred_view1_sel = torch.masked_select(train_labeled_pred_view1_aligned, labeled_mask)
            # train_labeled_pred_view2_sel = torch.masked_select(train_labeled_pred_view2_aligned, labeled_mask)
            # unlabeled data
            train_unlabeled_pred_view1_sel = torch.masked_select(train_unlabeled_pred_view1_aligned, unlabeled_mask)
            train_unlabeled_pred_view2_sel = torch.masked_select(train_unlabeled_pred_view2_aligned, unlabeled_mask)
            train_unlabeled_pred_allview_sel = [train_unlabeled_pred_view1_sel,train_unlabeled_pred_view2_sel]

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

    def ValOneEpoch(self):
        '''
        Evaluate one epoch on validation set
        :return:
        '''

        epoch_loss = 0.

        val_pred_all = []
        val_gt_all = []

        # enable network for evaluation
        self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch val samples
            FinishEpoch, data = \
                self.Loader.NextValBatch()
            if FinishEpoch:
                break

            ## Apply Normalization
            val_data = self.ApplyNormalization(data['data'])

            val_data = np.transpose(val_data, [0, 3, 1, 2])
            val_gt = np.transpose(data['gt'], [0, 1, 2])

            ## val current batch
            val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
            val_gt = torch.tensor(val_gt, device=self.device, dtype=torch.float32)

            # Augment with location
            if self.Location:
                x, y = np.meshgrid(np.arange(0, val_data.shape[3]),
                                   np.arange(0, val_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
                x = torch.tensor(np.tile(x, [val_data.shape[0], 1, 1, 1]),
                                 device=val_data.device)
                y = torch.tensor(np.tile(y, [val_data.shape[0], 1, 1, 1]),
                                 device=val_data.device)
                val_data = torch.cat([val_data, x, y], dim=1)

            # forward pass
            val_pred = self.net_ema(val_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=val_pred[:, 0, ...], gt_labeled=val_gt)
            loss_class = self.DiceLoss(pred=val_pred[:, 0, ...],gt=val_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            # accumulate predictions and ground-truths
            val_pred = torch.sigmoid(val_pred)
            val_pred_all.append(val_pred.detach().cpu().numpy())
            val_gt_all.append(val_gt.detach().cpu().numpy())

            del val_pred, val_gt
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

            itr += 1

        ## Evaluate val performance
        val_pred_all = np.concatenate(val_pred_all)
        val_pred_all = np.squeeze(val_pred_all > 0.5, axis=1).astype(float)  # binarize predictions
        val_gt_all = np.concatenate(val_gt_all).astype(float)
        perpix_acc = self.evaluator.perpixel_acc(val_pred_all,
                                                 val_gt_all)  # evaluate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(val_pred_all,
                                                 val_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(val_pred_all, val_gt_all)  # evaluate micro-average iou

        self.val_loss = epoch_loss
        self.val_Acc = perpix_acc
        self.val_macro_IoU = persamp_iou
        self.val_micro_IoU = micro_iou

        return epoch_loss, perpix_acc, persamp_iou, micro_iou

    def TestAll_SavePred(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        :return:
        '''

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))

        self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            # accumulate predictions and ground-truths
            test_pred = torch.sigmoid(test_pred)
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or \
                self.target_data == 'MITRoadClean' or self.target_data == 'Gaps384' or \
                ('EM' in self.target_data) or ('DRIVE' in self.target_data) or self.target_data=='CORN1':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.array(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
            micro_F1, macro_F1 = self.evaluator.F1(test_pred_bin_all, test_gt_all)

        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU, micro_F1, macro_F1 = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU
        self.best_te_micro_F1 = micro_F1
        self.best_te_macro_F1 = macro_F1

        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_SavePred_debug(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        :return:
        '''

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0
        Max_Test_Samp = 5


        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch or itr>= Max_Test_Samp:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            # accumulate predictions and ground-truths
            test_pred = torch.sigmoid(test_pred)
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or \
                self.target_data == 'MITRoadClean' or self.target_data == 'Gaps384':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.array(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_TopoPostProc(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        :return:
        '''

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        tloss = Loss.TopoLoss([512,512])  # topology penalty
        gloss = nn.MSELoss()

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            test_pred = torch.sigmoid(test_pred)

            # Topology Post Processing
            for pred_i in test_pred:
                # x_init = pred_i[0].detach()
                x_init = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                x_t = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                x_t = torch.autograd.Variable(torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device).type(torch.float),
                                              requires_grad=True)

                optimizer = torch.optim.Adam([x_t], lr=1e-2)

                for i in range(500):
                    # tick = time.time()
                    optimizer.zero_grad()
                    tlossi, dgminfo = tloss(x_t)
                    glossi = gloss(x_t, x_init)
                    loss = 1 * tlossi + glossi
                    loss.backward()
                    optimizer.step()


            # accumulate predictions and ground-truths
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or self.target_data == 'MITRoadClean':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.array(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_CropTopoPostProc(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        Use Topology loss to refine the prediction in a cropped region
        :return:
        '''

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        tloss = Loss.TopoLoss([50,50])  # topology penalty
        gloss = nn.MSELoss()

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            test_pred = torch.sigmoid(test_pred)

            # Topology Post Processing
            for pred_i in test_pred:
                # x_init = pred_i[0].detach()
                x_init = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                # crop prediction map
                x_init = x_init[100:150,100:150]
                x_t = x_init.clone()
                x_t.requires_grad = True
                # x_t = torch.autograd.Variable(x_init.data, device=pred_i.device, requires_grad=True)
                # x_t = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                # x_t = torch.autograd.Variable(torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device).type(torch.float),
                #                               requires_grad=True)

                optimizer = torch.optim.Adam([x_t], lr=1e-2)

                for i in range(500):
                    # tick = time.time()
                    optimizer.zero_grad()
                    tlossi, dgminfo = tloss(x_t)
                    glossi = gloss(x_t, x_init)
                    loss = 1 * tlossi + glossi
                    loss.backward()
                    optimizer.step()


            # accumulate predictions and ground-truths
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or self.target_data == 'MITRoadClean':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.array(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_RandCropTopoPostProc(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        Use Topology loss to refine the prediction in randomly cropped regions
        :return:
        '''

        ## Load Topology Refinement Settings
        self.LoadTopoRefinementSetting()

        ## Export Topology Refinement Settings
        self.ExportTopoRefinementSetting(topo_setting_expfilepath=os.path.join(self.result_path, 'topo_settings.txt'))

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        tloss = Loss.TopoLoss([int(self.TopoSettings['CropSize']/2**self.TopoSettings['DownsampTimes']),
                               int(self.TopoSettings['CropSize']/2**self.TopoSettings['DownsampTimes'])])  # topology penalty
        gloss = nn.MSELoss()

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0
        sampcnt = 1 # inference sample counter
        Max_Test_Itr = np.inf  # maximal testing iterations (to save time, so test on fewer samples)

        # self.Downsamp = nn.functional.interpolate([2, 2])
        # self.Upsamp = nn.Upsample(scale_factor=2, mode='bilinear')

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch or itr>= Max_Test_Itr:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            test_pred = torch.sigmoid(test_pred)

            # Topology Post Processing
            for name_i, pred_i in zip(test_names, test_pred):

                print('\nPost Process {}-th sample {}\n'.format(sampcnt, name_i))

                ## Repeat random crop and optimization for multiple times
                for rep in range(self.TopoSettings['RandCrop']):

                    # x_init = pred_i[0].detach()
                    x_init = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=False).type(torch.float)
                    # crop prediction map
                    # random crop a region
                    x, y = [np.floor(pred_i.shape[1] * np.random.random()),
                            np.floor(pred_i.shape[2] * np.random.random())]
                    if x <= self.TopoSettings['CropSize'] / 2:
                        x = self.TopoSettings['CropSize'] / 2
                    if y <= self.TopoSettings['CropSize'] / 2:
                        y = self.TopoSettings['CropSize'] / 2
                    if x >= pred_i.shape[1] - self.TopoSettings['CropSize'] / 2:
                        x = pred_i.shape[1] - self.TopoSettings['CropSize'] / 2
                    if y >= pred_i.shape[2] - self.TopoSettings['CropSize'] / 2:
                        y = pred_i.shape[2] - self.TopoSettings['CropSize'] / 2
                    x = int(x)
                    y = int(y)
                    x_init_crop = x_init[x - int(self.TopoSettings['CropSize'] / 2):x + int(self.TopoSettings['CropSize'] / 2),
                                  y - int(self.TopoSettings['CropSize'] / 2):y + int(self.TopoSettings['CropSize'] / 2)]  # detach from graph

                    ## Downsample
                    x_init_crop = x_init_crop.unsqueeze(0).unsqueeze(0)
                    for i_down in range(0,self.TopoSettings['DownsampTimes']):
                        x_init_crop = nn.functional.interpolate(x_init_crop, scale_factor=0.5)
                    x_init_crop = x_init_crop[0, 0]
                    # x_init_crop = nn.functional.interpolate(x_init_crop.unsqueeze(0).unsqueeze(0),scale_factor=0.5)[0,0]
                    # x_init_crop = nn.functional.interpolate(x_init_crop,scale_factor=0.5)[0,0]

                    # x_init = x_init[100:150,100:150]
                    x_t = x_init_crop.clone()

                    max_val = torch.max(x_t)  # maximal logit

                    x_t[0, :] = max_val
                    x_t[ -1, :] = max_val
                    x_t[:, 0] = max_val
                    x_t[ :, -1] = max_val

                    x_t.requires_grad = True
                    # x_t = Variable(x_t.data, requires_grad=True)

                    # x_t = torch.autograd.Variable(x_init.data, device=pred_i.device, requires_grad=True)
                    # x_t = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                    # x_t = torch.autograd.Variable(torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device).type(torch.float),
                    #                               requires_grad=True)

                    optimizer = torch.optim.Adam([x_t], lr=1e-2)

                    for i in range(self.TopoSettings['OptIters']):
                        # tick = time.time()
                        optimizer.zero_grad()
                        tlossi, dgminfo = tloss(x_t)
                        glossi = gloss(x_t, x_init_crop)
                        loss = (1 * tlossi + 1* glossi)/(1+1)
                        loss.requres_grad = True
                        loss.backward()
                        optimizer.step()

                        print('\rRandom Crop Rep-{} Itr-{}'.format(rep,i),end='')

                    ## Upsample
                    x_t = x_t[1: -1, 1: -1] # remove boundary
                    x_t = x_t.unsqueeze(0).unsqueeze(0)
                    for i_up in range(0,self.TopoSettings['DownsampTimes']):
                        x_t = nn.functional.interpolate(x_t, scale_factor=2)
                    x_t = x_t[0,0]
                    # x_t = nn.functional.interpolate(x_t, scale_factor=2)[0, 0]

                    ## Put refined results to prediction output
                    pred_i[0,x - int(self.TopoSettings['CropSize'] / 2) + 2**self.TopoSettings['DownsampTimes']:
                             x + int(self.TopoSettings['CropSize'] / 2) - 2**self.TopoSettings['DownsampTimes'],
                             y - int(self.TopoSettings['CropSize'] / 2) + 2**self.TopoSettings['DownsampTimes']:
                             y + int(self.TopoSettings['CropSize'] / 2) - 2**self.TopoSettings['DownsampTimes']] = x_t

                ## Apped results after refinement
                test_pred_all.append(pred_i[0].detach().cpu().numpy())
                sampcnt += 1

            # accumulate predictions and ground-truths
            # for pred_i in test_pred.detach().cpu().numpy():
            #     test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred_topo')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or self.target_data == 'MITRoadClean':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.stack(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        # print('IoU: {}'.format(np.mean(persamp_iou)))


        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_RandCropTopoPostProc_v1(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        Use Topology loss to refine the prediction in randomly cropped regions
        :return:
        '''

        ## Load Topology Refinement Settings
        self.LoadTopoRefinementSetting()

        ## Export Topology Refinement Settings
        self.ExportTopoRefinementSetting(topo_setting_expfilepath=os.path.join(self.result_path, 'topo_settings.txt'))

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        tloss = Loss.TopoLoss([int(self.TopoSettings['CropSize']/2**self.TopoSettings['DownsampTimes']),
                               int(self.TopoSettings['CropSize']/2**self.TopoSettings['DownsampTimes'])])  # topology penalty
        gloss = nn.MSELoss()

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0
        Max_Test_Itr = 5  # maximal testing iterations (to save time, so test on fewer samples)

        # self.Downsamp = nn.functional.interpolate([2, 2])
        # self.Upsamp = nn.Upsample(scale_factor=2, mode='bilinear')

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch or itr>= Max_Test_Itr:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            test_pred = torch.sigmoid(test_pred)

            # Topology Post Processing
            for name_i, pred_i in zip(test_names, test_pred):

                print('\nPost Proc {}\n'.format(name_i))

                ## Repeat random crop and optimization for multiple times
                for rep in range(self.TopoSettings['RandCrop']):

                    # x_init = pred_i[0].detach()
                    x_init = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=False).type(torch.float)
                    # crop prediction map
                    # random crop a region
                    x, y = [np.floor(pred_i.shape[1] * np.random.random()),
                            np.floor(pred_i.shape[2] * np.random.random())]
                    if x <= self.TopoSettings['CropSize'] / 2:
                        x = self.TopoSettings['CropSize'] / 2
                    if y <= self.TopoSettings['CropSize'] / 2:
                        y = self.TopoSettings['CropSize'] / 2
                    if x >= pred_i.shape[1] - self.TopoSettings['CropSize'] / 2:
                        x = pred_i.shape[1] - self.TopoSettings['CropSize'] / 2
                    if y >= pred_i.shape[2] - self.TopoSettings['CropSize'] / 2:
                        y = pred_i.shape[2] - self.TopoSettings['CropSize'] / 2
                    x = int(x)
                    y = int(y)
                    x_init_crop = x_init[x - int(self.TopoSettings['CropSize'] / 2):x + int(self.TopoSettings['CropSize'] / 2),
                                  y - int(self.TopoSettings['CropSize'] / 2):y + int(self.TopoSettings['CropSize'] / 2)]  # detach from graph

                    ## Downsample
                    x_init_crop = x_init_crop.unsqueeze(0).unsqueeze(0)
                    for i_down in range(0,self.TopoSettings['DownsampTimes']):
                        x_init_crop = nn.functional.interpolate(x_init_crop, scale_factor=0.5)
                    x_init_crop = x_init_crop[0, 0]
                    # x_init_crop = nn.functional.interpolate(x_init_crop.unsqueeze(0).unsqueeze(0),scale_factor=0.5)[0,0]
                    # x_init_crop = nn.functional.interpolate(x_init_crop,scale_factor=0.5)[0,0]

                    # x_init = x_init[100:150,100:150]
                    x_t = x_init_crop.clone()

                    max_val = torch.max(x_t)  # maximal logit

                    x_t[0, :] = max_val
                    x_t[ -1, :] = max_val
                    x_t[:, 0] = max_val
                    x_t[ :, -1] = max_val

                    x_t.requires_grad = True
                    # x_t = Variable(x_t.data, requires_grad=True)

                    # x_t = torch.autograd.Variable(x_init.data, device=pred_i.device, requires_grad=True)
                    # x_t = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                    # x_t = torch.autograd.Variable(torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device).type(torch.float),
                    #                               requires_grad=True)

                    optimizer = torch.optim.Adam([x_t], lr=1e-2)

                    for i in range(self.TopoSettings['OptIters']):
                        # tick = time.time()
                        optimizer.zero_grad()
                        tlossi, dgminfo = tloss(x_t)
                        glossi = gloss(x_t, x_init_crop)
                        loss = (self.TopoSettings['TopologyWeight'] * tlossi + self.TopoSettings['GeometricWeight']* glossi)\
                               /(self.TopoSettings['TopologyWeight']+self.TopoSettings['GeometricWeight'])
                        loss.requres_grad = True
                        loss.backward()
                        optimizer.step()

                        print('\rRandom Crop Rep-{} Itr-{}'.format(rep,i),end='')

                    ## Upsample
                    x_t = x_t[1: -1, 1: -1] # remove boundary
                    x_t = x_t.unsqueeze(0).unsqueeze(0)
                    for i_up in range(0,self.TopoSettings['DownsampTimes']):
                        x_t = nn.functional.interpolate(x_t, scale_factor=2)
                    x_t = x_t[0,0]
                    # x_t = nn.functional.interpolate(x_t, scale_factor=2)[0, 0]

                    ## Put refined results to prediction output
                    pred_i[0,x - int(self.TopoSettings['CropSize'] / 2) + 2**self.TopoSettings['DownsampTimes']:
                             x + int(self.TopoSettings['CropSize'] / 2) - 2**self.TopoSettings['DownsampTimes'],
                             y - int(self.TopoSettings['CropSize'] / 2) + 2**self.TopoSettings['DownsampTimes']:
                             y + int(self.TopoSettings['CropSize'] / 2) - 2**self.TopoSettings['DownsampTimes']] = x_t

                ## Apped results after refinement
                test_pred_all.append(pred_i[0].detach().cpu().numpy())

            # accumulate predictions and ground-truths
            # for pred_i in test_pred.detach().cpu().numpy():
            #     test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred_topo')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or self.target_data == 'MITRoadClean':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.stack(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        print('IoU: {}'.format(np.mean(persamp_iou)))


        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_RandCropTopoPostProc_Debug(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        Use Topology loss to refine the prediction in randomly cropped regions
        :return:
        '''

        ## Load Topology Refinement Settings
        self.LoadTopoRefinementSetting()

        ## Export Topology Refinement Settings
        self.ExportTopoRefinementSetting(topo_setting_expfilepath=os.path.join(self.result_path, 'topo_settings.txt'))

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        tloss = Loss.TopoLoss([int(self.TopoSettings['CropSize']/2**self.TopoSettings['DownsampTimes']),
                               int(self.TopoSettings['CropSize']/2**self.TopoSettings['DownsampTimes'])])  # topology penalty
        gloss = nn.MSELoss()

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0
        Max_Test_Itr = 5  # maximal testing iterations (to save time, so test on fewer samples)

        # self.Downsamp = nn.functional.interpolate([2, 2])
        # self.Upsamp = nn.Upsample(scale_factor=2, mode='bilinear')

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch or itr>= Max_Test_Itr:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            test_pred = torch.sigmoid(test_pred)

            # Topology Post Processing
            for name_i, pred_i in zip(test_names, test_pred):

                print('\nPost Proc {}\n'.format(name_i))

                ## Repeat random crop and optimization for multiple times
                # Initialize all random croppings
                x_all = []
                y_all = []
                x_init_crops = []
                x_ts = []
                for rep in range(self.TopoSettings['RandCrop']):

                    x_init = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=False).type(torch.float)
                    # random crop a region
                    x, y = [np.floor(pred_i.shape[1] * np.random.random()),
                            np.floor(pred_i.shape[2] * np.random.random())]
                    if x <= self.TopoSettings['CropSize'] / 2:
                        x = self.TopoSettings['CropSize'] / 2
                    if y <= self.TopoSettings['CropSize'] / 2:
                        y = self.TopoSettings['CropSize'] / 2
                    if x >= pred_i.shape[1] - self.TopoSettings['CropSize'] / 2:
                        x = pred_i.shape[1] - self.TopoSettings['CropSize'] / 2
                    if y >= pred_i.shape[2] - self.TopoSettings['CropSize'] / 2:
                        y = pred_i.shape[2] - self.TopoSettings['CropSize'] / 2
                    x = int(x)
                    y = int(y)
                    x_init_crop = x_init[x - int(self.TopoSettings['CropSize'] / 2):x + int(self.TopoSettings['CropSize'] / 2),
                                  y - int(self.TopoSettings['CropSize'] / 2):y + int(self.TopoSettings['CropSize'] / 2)]  # detach from graph

                    ## Downsample
                    x_init_crop = x_init_crop.unsqueeze(0).unsqueeze(0)
                    for i_down in range(0,self.TopoSettings['DownsampTimes']):
                        x_init_crop = nn.functional.interpolate(x_init_crop, scale_factor=0.5)
                    x_init_crop = x_init_crop[0, 0]

                    ## Initialize optimization variable
                    x_t = x_init_crop.clone()

                    max_val = torch.max(x_t)  # maximal logit

                    x_t[0, :] = max_val
                    x_t[ -1, :] = max_val
                    x_t[:, 0] = max_val
                    x_t[ :, -1] = max_val

                    x_t.requires_grad = True

                    ## Accumulate All Croppings
                    x_all.append(x)
                    y_all.append(y)
                    x_init_crops.append(x_init_crop)
                    x_ts.append(x_t)

                    # x_t = Variable(x_t.data, requires_grad=True)

                    # x_t = torch.autograd.Variable(x_init.data, device=pred_i.device, requires_grad=True)
                    # x_t = torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device, requires_grad=True).type(torch.float)
                    # x_t = torch.autograd.Variable(torch.tensor(pred_i[0].detach().cpu().numpy(),device=pred_i.device).type(torch.float),
                    #                               requires_grad=True)

                x_init_crops = torch.stack(x_init_crops)
                x_ts_1 = torch.stack(x_ts)
                nCrops = len(x_ts)

                optimizer = torch.optim.Adam(x_ts, lr=1e-2)

                for i in range(self.TopoSettings['OptIters']):
                        # tick = time.time()
                        optimizer.zero_grad()
                        top_loss = 0.
                        for x_t in x_ts:
                            tlossi, dgminfo = tloss(x_t)
                            top_loss += tlossi
                        top_loss /= nCrops

                        geo_loss = 0.
                        for x_t, x_init_crop in zip(x_ts, x_init_crops):
                            geo_loss += gloss(x_t, x_init_crops)
                        geo_loss /= nCrops

                        loss = (1 * top_loss + 1* geo_loss)/(1+1)
                        # loss.requires_grad = True
                        loss.backward(retain_graph=True)
                        optimizer.step()

                        print('\rRandom Crop Rep-{} Itr-{}'.format(rep,i),end='')

                ## Put the results back into output
                for x_t, x, y in zip(x_ts, x_all, y_all):

                    ## Upsample
                    x_t = x_t[1: -1, 1: -1] # remove boundary
                    x_t = x_t.unsqueeze(0).unsqueeze(0)
                    for i_up in range(0,self.TopoSettings['DownsampTimes']):
                        x_t = nn.functional.interpolate(x_t, scale_factor=2)
                    x_t = x_t[0,0]
                    # x_t = nn.functional.interpolate(x_t, scale_factor=2)[0, 0]

                    ## Put refined results to prediction output
                    pred_i[0,x - int(self.TopoSettings['CropSize'] / 2) + 2**self.TopoSettings['DownsampTimes']:
                             x + int(self.TopoSettings['CropSize'] / 2) - 2**self.TopoSettings['DownsampTimes'],
                             y - int(self.TopoSettings['CropSize'] / 2) + 2**self.TopoSettings['DownsampTimes']:
                             y + int(self.TopoSettings['CropSize'] / 2) - 2**self.TopoSettings['DownsampTimes']] = x_t

                ## Apped results after refinement
                test_pred_all.append(pred_i[0].detach().cpu().numpy())

            # accumulate predictions and ground-truths
            # for pred_i in test_pred.detach().cpu().numpy():
            #     test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred_topo')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or self.target_data == 'MITRoadClean':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.stack(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
            test_gt_all = np.array(test_gt_all).astype(float)
            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        print('IoU: {}'.format(np.mean(persamp_iou)))


        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU


    def TestAll_ConnectComp(self,exp_fig=False,best_ckpt_filepath=None):
        '''
        Test all samples in the test set with optimal threshold minimizing the #connected component
        :return:
        '''

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            # test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            ## forward pass
            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),
                                   np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]),
                                 device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            # accumulate predictions and ground-truths
            test_pred = torch.sigmoid(test_pred/0.01)
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)
            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())
            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # test_pred_all.append(test_pred.detach().cpu().numpy())
            # test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(self.export_path,'gt','{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path,'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    # plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    # plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred.png'.format(img_i))
                    img = Image.fromarray(255*pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0]>0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        if self.target_data == 'CrackForest' or self.target_data == 'MITRoad' or self.target_data == 'MITRoadClean':
            # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
            test_pred_all = np.array(test_pred_all)
            test_gt_all = np.array(test_gt_all).astype(float)

            nConnComps = []
            iou = []
            # Find optimal threshold
            for tau in [1e-3,1e-2,1e-1,2e-1,5e-1,6e-1,7e-1,1e0]:
                test_pred_bin_all = (test_pred_all > tau).astype(float)  # binarize predictions
                micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
                nr_objects = []
                for pred_bin in test_pred_bin_all:
                    _, objects = ndimage.label(pred_bin)
                    nr_objects.append(objects)
                nConnComps.append(np.mean(nr_objects))
                iou.append(micro_iou)

            # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,
                                                     test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,
                                                     test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
            AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU

        return epoch_loss, perpix_acc, persamp_iou, micro_iou, AIU

    def TestAll_SavePred_bk(self,export_path=None,best_ckpt_filepath=None):
        '''
        Test all samples in the test set and export predictions
        :return:
        '''

        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net_ema.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))
            self.net_ema.eval()

        itr = 0

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_thin = data['gt_thin']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            # forward pass
            test_pred = self.net_ema(test_data)  # forward pass prediction of train labeled set
            # losses = self.criterion(pred_labeled=test_pred[:, 0, ...], gt_labeled=test_gt)
            loss_class = self.DiceLoss(pred=test_pred[:, 0, ...],gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            # accumulate predictions and ground-truths
            test_pred = torch.sigmoid(test_pred)
            test_pred_all.append(test_pred.detach().cpu().numpy())
            test_gt_all.append(test_gt.detach().cpu().numpy())

            # export prediction and gt figures
            if export_path is not None:
                for img_i, gt_i, pred_i in zip(test_names,test_gt,test_pred):
                    exp_gt_filepath = os.path.join(export_path,'{}_gt.png'.format(img_i))
                    exp_pred_filepath = os.path.join(export_path,'{}_pred.png'.format(img_i))
                    plt.imsave(exp_gt_filepath,gt_i.detach().cpu().numpy())
                    plt.imsave(exp_pred_filepath,pred_i.detach().cpu().numpy()[0])

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        test_pred_all = np.concatenate(test_pred_all)
        test_pred_all = np.squeeze(test_pred_all > 0.5, axis=1).astype(float)  # binarize predictions
        test_gt_all = np.concatenate(test_gt_all).astype(float)
        perpix_acc = self.evaluator.perpixel_acc(test_pred_all,
                                                 test_gt_all)  # etestuate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(test_pred_all,
                                                 test_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(test_pred_all, test_gt_all)  # evaluate micro-average iou

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou

        return epoch_loss, perpix_acc, persamp_iou, micro_iou


    def PrintTrValInfo(self):
        '''
        Print train and validation set information
        :return:
        '''

        ## Print out Training Info
        print('\nEpoch: {}'.format(self.epoch_cnt))
        print('loss/train: {:.2f}  train_consist: {:.2f}  val: {:.2f}'.format(self.tr_loss, self.tr_loss_consist, self.val_loss))
        print('perpix_accuracy/train: {:.2f}%    val: {:.2f}%'.format(100 * self.tr_Acc, 100 * self.val_Acc))
        print('macro average IoU/train: {:.2f}%   val:{:.2f}%'.format(100 * np.mean(self.tr_macro_IoU),
                                                                      100 * np.mean(self.val_macro_IoU)))
        print('micro average IoU/train: {:.2f}%   val:{:.2f}%'.format(100 * self.tr_micro_IoU, 100 * self.val_micro_IoU))

    def ApplyAugmentation_Mask(self, data, gt):
        '''
        Apply augmentation to batch data and obtain the valid region mask
        :return:
        '''

        ## Apply Augmentatoin
        # image level augmentation
        data, gt = self.augmentor.augment(images=data.astype(np.float32), masks=gt.astype(np.float32))

        # geometric transformation augmentation
        self.gt_model.construct_random_transform(data)
        Tform = self.gt_model.Tform # transformation matrices
        data = self.gt_model.transform_images(data, extrapolation='reflect')    # transformed image data
        gt = self.gt_model.transform_images(gt[..., np.newaxis], extrapolation='reflect')[..., 0]   # transformed ground-truth masks
        mask = np.ones_like(gt)
        mask = self.gt_model.transform_images(mask[..., np.newaxis], extrapolation='constant')[..., 0]
        mask = self.gt_model.invtransform_images(mask[..., np.newaxis], extrapolation='constant')[..., 0]   # valid mask in original frame

        ## Apply Normalization
        data = (data - self.Loader.mean)/self.Loader.stddev

        ## new dimension order
        data = np.transpose(data, [0, 3, 1, 2])
        gt = np.transpose(gt, [0, 1, 2])
        mask = np.transpose(mask, [0, 1, 2])

        return data, gt, mask, Tform


    class WeightRampup(nn.Module):
        '''
        Weight Rampup class
        :return:
        '''

        def __init__(self,RampupType='Exp',RampupEpoch=50):
            '''

            :param RampupType: Exp(Exponential), Step(Step increase)
            :param RampupEpoch: A rampup epoch constant
            '''
            super().__init__()
            self.RampupType = RampupType
            self.RampupEpoch = RampupEpoch

        def forward(self, **kwargs):

            epoch = kwargs['epoch']

            if self.RampupType == 'Exp':
                return np.exp(-10.*(1-np.clip(epoch, 0.0, self.RampupEpoch)/self.RampupEpoch)**2)
            elif self.RampupType == 'Step':
                return 1.0 if epoch >= self.RampupEpoch else 0.0

    def UpdateLatestModel(self):
        '''
        Update Latest Model
        :return:
        '''
        # Save model
        todelete_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(self.epoch_cnt - self.ValFreq * self.MaxKeepCkpt))
        if os.path.exists(todelete_ckpt):
            os.remove(todelete_ckpt)
        current_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(self.epoch_cnt))
        torch.save(self.net_ema.state_dict(), current_ckpt)  # for MT model, we save the ema model
        # save the best up-to-date model
        if self.best_val_IoU < self.val_micro_IoU:
            self.best_val_IoU = self.val_micro_IoU
            best_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format('best'))
            os.system('cp {} {}'.format(current_ckpt, best_ckpt))
            # update best tr and val performance
            self.UpdateBest()
            print('saved current epoch as the best up-to-date model')

    def ExportTensorboard(self):
        '''
        Export train/val and inference results to tensorboard file
        :param result_filepath:
        :return:
        '''
        self.writer.add_scalar('loss/train', self.tr_loss, self.epoch_cnt)
        self.writer.add_scalar('loss/val', self.val_loss, self.epoch_cnt)
        self.writer.add_scalar('loss/train_consist', self.tr_loss_consist, self.epoch_cnt)
        self.writer.add_scalar('loss/rampup_weight', self.weight, self.epoch_cnt)
        self.writer.add_scalar('perpix_accuracy/train', 100 * self.tr_Acc, self.epoch_cnt)
        self.writer.add_scalar('perpix_accuracy/val', 100 * self.val_Acc, self.epoch_cnt)
        self.writer.add_scalar('macro average IoU/train', np.mean(self.tr_macro_IoU), self.epoch_cnt)
        self.writer.add_scalar('macro average IoU/val', np.mean(self.val_macro_IoU), self.epoch_cnt)
        self.writer.add_scalar('micro average IoU/train', np.mean(self.tr_micro_IoU), self.epoch_cnt)
        self.writer.add_scalar('micro average IoU/val', np.mean(self.val_micro_IoU), self.epoch_cnt)


    def update_ema_variables(self):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for ema_param, param in zip(self.net_ema.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def RestoreModelByPath(self, ckpt_path=None, model_epoch='best'):

        # if ckpt_path is None:
        #     mdl_filepath = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(model_epoch))
        # else:
        #     mdl_filepath = os.path.join(ckpt_path, 'model_epoch-{}.pt'.format(model_epoch))
        #
        # net_ema = getattr(self, 'net_ema')
        net = getattr(self, 'net')
        net_ema = getattr(self, 'net_ema')
        state_dict = torch.load(ckpt_path, map_location=str(self.device))
        # net_ema.load_state_dict(state_dict)
        net_state_dict = net.state_dict()
        emanet_state_dict = net_ema.state_dict()
        missing_keys = []
        mismatch_keys = []
        for k, v in net_state_dict.items():
            if k in state_dict:
                if v.shape == state_dict[k].shape:
                    net_state_dict[k].copy_(state_dict[k])
                    emanet_state_dict[k].copy_(state_dict[k])
                else:
                    mismatch_keys.append(k)
            else:
                missing_keys.append(k)
        print('Missing keys: {}'.format(missing_keys))
        print('Mismatch keys: {}'.format(mismatch_keys))


    def SaveAllSettings(self, args):
        '''
        Save All settings
        :param args:
        :return:
        '''

        if args.SaveRslt:

            self.settings_filepath = os.path.join(self.result_path,'settings.txt')

            with open(self.settings_filepath,'w') as fid:
                fid.write('Host:{}\n'.format(socket.gethostname()))
                fid.write('GPU:{}\n'.format(args.GPU))
                fid.write('SplitSeed:{}\n'.format(args.seed_split))
                fid.write('Network:{}\n'.format(args.net))
                fid.write('LearningRate:{}\n'.format(args.LearningRate))
                fid.write('Epoch:{}\n'.format(args.Epoch))
                fid.write('batchsize:{}\n'.format(args.batchsize))
                fid.write('labelpercent:{}\n'.format(args.labelpercent))
                fid.write('loss:{}\n'.format(args.loss))
                fid.write('HingeC:{}\n'.format(args.HingeC))
                fid.write('Temperature:{}\n'.format(args.Temperature))
                fid.write('lp:{}\n'.format(args.lp))
                fid.write('ssl:{}\n'.format(args.ssl))
                fid.write('EmaAlpha:{}\n'.format(self.alpha))
                fid.write('Gamma:{}\n'.format(args.Gamma))
                fid.write('RampupEpoch:{}\n'.format(args.RampupEpoch))
                fid.write('RampupType:{}\n'.format(args.RampupType))
                fid.write('Target Dataset:{}\n'.format(args.TargetData))
                fid.write('Aux Dataset:{}\n'.format(args.AddUnlab))
                fid.write('AddLocation:{}\n'.format(args.Location))
                fid.write('SinPeriod:{}\n'.format(args.SinPeriod))
                fid.write('Augment:{}\n'.format(args.Augment))
                if 'Elastic' in self.Augment:
                    fid.write('Elastic Alpha:{}\n'.format(self.ElasticPara['alpha']))
                    fid.write('Elastic Sigma:{}\n'.format(self.ElasticPara['sigma']))
