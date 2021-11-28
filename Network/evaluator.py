import torch
import numpy as np
import skimage.morphology as morph
import cv2
from sklearn import metrics


class evaluate():

    def __init__(self):

        pass

    def perpixel_acc(self, pred, gt):
        '''
        calculate per pixel accuracy given binary inputs
        :param pred:    B*H*W
        :param gt:      B*H*W
        :return:
        '''

        perpix_acc = np.mean((pred == gt).astype(float))

        return perpix_acc

    def persamp_iou(self, pred, gt):
        '''
        calculate per sample/image segmentatoin iou
        :param pred:    B*H*W
        :param gt:      B*H*W
        :return:
        '''

        interesect = np.sum(pred.astype(float) * gt.astype(float), axis=(1, 2))
        union = np.sum(pred.astype(float) + gt.astype(float), axis=(1, 2)) - interesect
        persamp_iou = interesect / (union + 1e-10)

        return persamp_iou

    def micro_iou(self, pred, gt):
        '''
        evaluate mirco-average iou. the tp, fp are cumulated over all samples.
        :param pred: prediction  B*H*W
        :param gt:  ground-truth B*H*W
        :return:
        '''

        pred = pred.reshape(-1)
        gt = gt.reshape(-1)
        intersect = np.sum(pred.astype(float) * gt.astype(float))
        union = np.sum(pred.astype(float) + gt.astype(float)) - intersect
        micro_iou = intersect / (union + 1e-10)
        return micro_iou

    def ODS(self, pred, gt):
        '''
        Evaluate F-measure with the dataset-level optimal threshold
        :param pred: prediction  B*H*W
        :param gt:  ground-truth B*H*W
        :return:
        '''

        thresholds = np.arange(0, 1, 0.01)
        fm_best = 0.

        for threshold in thresholds:
            pred_bin = (pred > threshold).astype(int)  # binarize prediction
            pred_bin_thin = morph.thin(pred_bin).astype(np.uint8)  # thin prediction

            tp = np.sum((pred > threshold).astype(int) * gt)
            fp = np.sum((pred > threshold).astype(int) * (1 - gt))
            tn = np.sum((pred <= threshold).astype(int) * (1 - gt))
            fn = np.sum((pred <= threshold).astype(int) * (gt))
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            fm = 2 * prec * recall / (prec + recall)
            if fm_best < fm:
                fm_best = fm
                threshold_best = threshold

        return fm_best, threshold_best

    def OIS(self, preds, gts):
        '''
        Evaluate F-measure with the sample-level optimal threshold
        :param preds: prediction  B*H*W
        :param gts:  ground-truth B*H*W
        :return:
        '''

        thresholds = np.arange(0, 1, 100)
        fm_best = 0.

        for pred, gt in zip(preds, gts):
            fm_all = []
            threshold_all = []
            fm_best = 0.
            for threshold in thresholds:
                tp = np.sum((pred > threshold).astype(int) * gt)
                fp = np.sum((pred > threshold).astype(int) * (1 - gt))
                tn = np.sum((pred <= threshold).astype(int) * (1 - gt))
                fn = np.sum((pred <= threshold).astype(int) * (gt))
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                fm = 2 * prec * recall / (prec + recall)
                if fm_best < fm:
                    fm_best = fm
                    threshold_best = threshold
                fm_all.append(fm_best)
                threshold_all.append(threshold_best)

        return fm_all, threshold_all

    def AIU(self, preds, gts):
        '''
        Compute Average Intersect Over Union
        :param pred: probabilistic output
        :param gt:
        :return:
        '''

        thresholds = np.arange(0, 1, 0.01)
        fm_best = 0.
        persamp_aiu = []
        for pred, gt in zip(preds, gts):
            persamp_iou = []
            for threshold in thresholds:
                interesect = np.sum((pred > threshold).astype(float) * gt.astype(float))
                union = np.sum((pred > threshold).astype(float) + gt.astype(float)) - interesect
                persamp_iou.append(interesect / (union + 1e-10))
            persamp_aiu.append(np.mean(persamp_iou))

        return np.mean(persamp_aiu)

    def F1(self, preds, gts):
        '''
        Compute F1 measure
        :param pred: binary predictions
        :param gt: binary ground-truth
        :return:
        '''

        preds = preds.astype(int)
        gts = gts.astype(int)

        F1_micro = metrics.f1_score(gts.reshape([-1]).astype(int), preds.reshape([-1]).astype(int))

        F1_macro = []

        for pred, gt in zip(preds, gts):
            F1_macro.append(metrics.f1_score(gt.reshape([-1]).astype(int), pred.reshape([-1]).astype(int)))

        F1_macro = np.mean(F1_macro)

        return F1_micro, F1_macro

    def Dice(self, preds, gts):
        '''
        Compute Dice coefficient
        :param pred: binary predictions
        :param gt: binary ground-truth
        :return:
        '''

        preds = preds.astype(int)
        gts = gts.astype(int)

        macro_dice = 2 * np.sum(preds * gts, axis=(1, 2))
        macro_dice = macro_dice / (np.sum(preds, axis=(1, 2)) + np.sum(gts, axis=(1, 2)))
        macro_dice = np.mean(macro_dice)

        micro_dice = 2 * np.sum(preds * gts)
        micro_dice = micro_dice / (np.sum(preds) + np.sum(gts))

        return micro_dice, macro_dice


class evaluate_list():
    '''
    class to evaluate a list of predictions and ground-truths. Each sample may have different dimension.
    '''

    def __init__(self):

        pass

    def perpixel_acc(self, preds, gts):
        '''
        calculate per pixel accuracy given binary inputs
        :param preds:    list of B samples H*W
        :param gts:      list of B samples H*W
        :return:
        '''

        perpix_acc = 0.
        nPixs = 0.
        for pred, gt in zip(preds, gts):
            ## resize prediction to original gt dimension
            pred = (cv2.resize(pred, (gt.shape[1], gt.shape[0])) > 0.5).astype(float)
            perpix_acc += np.sum((pred == gt).astype(float))
            nPixs += pred.size
        perpix_acc /= nPixs

        return perpix_acc

    def persamp_iou(self, preds, gts):
        '''
        calculate per sample/image segmentatoin iou
        :param preds:    list of B samples H*W
        :param gts:      list of B samples H*W
        :return:
        '''

        persamp_iou = []
        for pred, gt in zip(preds, gts):
            ## resize prediction to original gt dimension
            pred = (cv2.resize(pred, (gt.shape[1], gt.shape[0])) > 0.5).astype(float)
            interesect = np.sum(pred * gt.astype(float))
            union = np.sum(pred + gt.astype(float)) - interesect
            persamp_iou.append(interesect / (union + 1e-10))
        persamp_iou = np.array(persamp_iou)

        return persamp_iou

    def micro_iou(self, preds, gts):
        '''
        evaluate mirco-average iou. the tp, fp are cumulated over all samples.
        :param preds:    list of B samples H*W
        :param gts:      list of B samples H*W
        :return:
        '''

        intersect = 0.
        union = 0.
        for pred, gt in zip(preds, gts):
            ## resize prediction to original gt dimension
            pred = (cv2.resize(pred, (gt.shape[1], gt.shape[0])) > 0.5).astype(float)
            pred = pred.reshape(-1)
            gt = gt.reshape(-1)
            intersect += np.sum(pred * gt.astype(float))
            union += np.sum(pred + gt.astype(float))
        union -= intersect
        micro_iou = intersect / (union + 1e-10)
        return micro_iou

    def ODS(self, pred, gt):
        '''
        Evaluate F-measure with the dataset-level optimal threshold
        :param pred: prediction  B*H*W
        :param gt:  ground-truth B*H*W
        :return:
        '''

        thresholds = np.arange(0, 1, 0.01)
        fm_best = 0.

        for threshold in thresholds:
            pred_bin = (pred > threshold).astype(int)  # binarize prediction
            pred_bin_thin = morph.thin(pred_bin).astype(np.uint8)  # thin prediction

            tp = np.sum((pred > threshold).astype(int) * gt)
            fp = np.sum((pred > threshold).astype(int) * (1 - gt))
            tn = np.sum((pred <= threshold).astype(int) * (1 - gt))
            fn = np.sum((pred <= threshold).astype(int) * (gt))
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            fm = 2 * prec * recall / (prec + recall)
            if fm_best < fm:
                fm_best = fm
                threshold_best = threshold

        return fm_best, threshold_best

    def OIS(self, preds, gts):
        '''
        Evaluate F-measure with the sample-level optimal threshold
        :param preds: prediction  B*H*W
        :param gts:  ground-truth B*H*W
        :return:
        '''

        thresholds = np.arange(0, 1, 100)
        fm_best = 0.

        for pred, gt in zip(preds, gts):
            fm_all = []
            threshold_all = []
            fm_best = 0.
            for threshold in thresholds:
                tp = np.sum((pred > threshold).astype(int) * gt)
                fp = np.sum((pred > threshold).astype(int) * (1 - gt))
                tn = np.sum((pred <= threshold).astype(int) * (1 - gt))
                fn = np.sum((pred <= threshold).astype(int) * (gt))
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                fm = 2 * prec * recall / (prec + recall)
                if fm_best < fm:
                    fm_best = fm
                    threshold_best = threshold
                fm_all.append(fm_best)
                threshold_all.append(threshold_best)

        return fm_all, threshold_all

    def AIU(self, preds, gts):
        '''
        Compute Average Intersect Over Union
        :param preds:    list of B samples H*W
        :param gts:      list of B samples H*W
        :return:
        '''

        thresholds = np.arange(0, 1, 0.01)
        fm_best = 0.
        persamp_aiu = []
        for pred, gt in zip(preds, gts):
            ## resize prediction to original gt dimension
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            persamp_iou = []
            for threshold in thresholds:
                interesect = np.sum((pred > threshold).astype(float) * gt.astype(float))
                union = np.sum((pred > threshold).astype(float) + gt.astype(float)) - interesect
                persamp_iou.append(interesect / (union + 1e-10))
            persamp_aiu.append(np.mean(persamp_iou))

        return np.mean(persamp_aiu)

    def F1(self, preds, gts):
        '''
        Compute F1 measure
        :param pred: binary predictions
        :param gt: binary ground-truth
        :return:
        '''

        # preds = preds.astype(int)
        # gts = gts.astype(int)
        preds_bin = []
        preds_bin_all = []
        gts_all = []

        for pred, gt in zip(preds, gts):
            pred_bin = (cv2.resize(pred, (gt.shape[1], gt.shape[0])) > 0.5).astype(float)
            preds_bin.append(pred_bin)
            preds_bin_all.append(pred_bin.reshape([-1]))
            gts_all.append(gt.reshape([-1]))
        # preds = np.stack(preds_bin)
        preds_bin_all = np.concatenate(preds_bin_all)
        gts_all = np.concatenate(gts_all)

        F1_micro = metrics.f1_score(gts_all.reshape([-1]).astype(int), preds_bin_all.reshape([-1]).astype(int))

        F1_macro = []

        for pred, gt in zip(preds_bin, gts):
            F1_macro.append(metrics.f1_score(gt.reshape([-1]).astype(int), pred.reshape([-1]).astype(int)))

        F1_macro = np.mean(F1_macro)

        return F1_micro, F1_macro