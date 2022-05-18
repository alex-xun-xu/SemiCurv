## script to Inference Unet with Dice Coefficient Loss

import os
import sys
import numpy as np

sys.path.append('./Trainer')
sys.path.append('./DataLoader')

import torch
import argparse as arg


## Take input arguments
parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--Mode',type=str,help='Running mode default: train [train, test]',default='train')
parser.add_argument('--Ckpt',type=str,help='Ckpt to resotre for test/inference',
                    default='')
parser.add_argument('--GPU',type=int,help='GPU to use',default=0)
# parser.add_argument('--ExpSum','-es',type=int,help='Flag to indicate if export summary',default=0)    # bool
parser.add_argument('--SaveRslt','-sr',type=int,help='Flag to indicate if save trained model and results',default=1)    # bool
parser.add_argument('--ExportFigure','-xf',type=int,help='Flag to indicate if export figures',default=0)    # bool
parser.add_argument('--LearningRate', '-lr', type=float,help='Learning Rate',
                    default=1e-3)
parser.add_argument('--Epoch','-ep',type=int,help='Number of epochs to train [default: 51]',default=150)
parser.add_argument('--batchsize','-bs',type=int,help='Training batchsize [default: 1]',default=8)
parser.add_argument('--labelpercent','-m',type=float,help='the ratio of samples selected to be labelled [default: 0.05]',
                    default=0.01)
parser.add_argument('--seed_split','-ssp',type=int,help='the seed to generate train/val/test split [default: 0]',
                    default=1)
# parser.add_argument('--seed_label','-ssl',type=int,help='the seed to generate train/val/test split [default: 0]',
#                     default=1)
parser.add_argument('--net','-n',type=str,help='network architecture [default: ResUnet] candidate: ResUnet, ResUnet_Location,'
                                               'ResUnet_SinusoidLocation, HED, RCF, DeepLabV3Plus',
                    default='ResUnet')
parser.add_argument('--loss','-l',type=str ,
                    help='Loss type [default: Diceloss] candidates: BCEloss, WeightedBCEloss, WeightedBCE+ConsistMSE_loss, '
                         'BCE+ConsistMSE_loss, Diceloss, Diceloss2, IoUloss, Dice+SupTopo_loss, '
                         'Dice+BCE_loss, Dice+ConsistVar_loss, Dice+ConsistMSE_loss, Dice+ConsistMSEall_loss, Dice+ConsistDiscMSE_loss,'
                         'Dice+ConsistMILMSE_loss, Dice+ConsistPriorMSE_loss, Dice+ConsistL1_loss, '
                         'Dice+ConsistMSE+thin_loss, Dice+ConsistDiscMSE_loss, Dice+ConsistHingeMSE_loss,'
                         'Dice+ConsistDice_loss, Dice+ConsistSCL_loss, Dice+Contrastive_loss, Dice+CosineContrastive_loss,'
                         'Dice+CosineContrastiveNoExp_loss',
                    default='Dice+ConsistMSE_loss')
parser.add_argument('--lp','-lp',type=float ,
                    help='Lp norm for consistency loss',
                    default=2)
parser.add_argument('--ssl','-ssl',type=str ,
                    help='Semi-supervised learning strategy [default: FullSup] candidates: FullSup, FullSup_nonorm, '
                         'PiMdl, MeanTeacher, MeanTeacher+NoGeoTform, MeanTeacher+Cycle, MeanTeacher+Bin, VAT, Mix-Match, MeanTeacher+Focal, '
                         'MeanTeacher, CutMix, SEM',
                    default='MeanTeacher')
parser.add_argument('--Gamma','-gm',type=float ,help='The weight applied to consistency loss',default=1.)
parser.add_argument('--CycleW','-cw',type=float ,help='The weight applied to cycle consistency loss',default=0.1)
parser.add_argument('--HingeC','-hc',type=float ,help='The threshold applied to hingeMSEloss',default=0.01)
parser.add_argument('--Temperature','-temp',type=float ,help='The temperature applied to teacher''s output [default:1.]',default=1.)
parser.add_argument('--Alpha','-alp',type=float ,help='The EMA parameter applied to MT [default: 0.99',default=0.99)
parser.add_argument('--RampupEpoch','-rpe',type=int ,help='Rampup epoch [default: 100]',default=100)
parser.add_argument('--RampupType','-rpt',type=str ,help='Rampup type [default: exp] candidates: ',default='Exp')
parser.add_argument('--MaxKeepCkpt','-mc',type=int ,help='maximum number of checkpoint to keep',default=2)
parser.add_argument('--AddUnlab','-au',type=str ,help='Additional Unlabeled data [default: None] candidates: None, Crack500',default='None')
parser.add_argument('--TargetData','-ta',type=str ,help='Target dataset [default: CrackForest] candidates: CrackForest,'
                                                        ' Crack500, Gaps384, EM, EM128, DRIVE128, CORN1',
                    default='EM')
parser.add_argument('--ValFreq','-vf',type=int ,help='Validation frequency (every n epochs) [default: 1]',default=3)
parser.add_argument('--MaxTrIter','-mti',type=int ,help='Maximal training iterations [default: inf]',default=10)
parser.add_argument('--SwapLabel','-sb',type=int ,help='Swap label 0 with 1 (It is not compatible with DiceLoss)[default: False]',default=0)
parser.add_argument('--Location','-loc',type=int ,help='Add location as additional feature [default: False]'
                    ,default=0)
parser.add_argument('--SinPeriod','-sp',type=int ,help='Sinusoid spatial encoding period [default: 2]',default=4)
parser.add_argument('--Augment', '-ag', type=str, help='Image Augmentation Type [default: Affine|Elastic] candidates None, Affine, Elastic', default='Affine')
parser.add_argument('--RunName','-rn',type=str ,help='filename to existing experiment results',
                    default='')

args = parser.parse_args()

lab_ratio = 0.5
if 'Contrastive' not in args.loss:
    if args.ssl == 'FullSup':
        from Trainer import trainer_unet as trainer

        lab_ratio = 1.
    elif args.ssl == 'FullSup_noaug':
        import trainer_unet_noaug as trainer
        lab_ratio = 1.
    elif args.ssl == 'FullSup_nonorm':
        import trainer_unet_nonorm as trainer
        lab_ratio = 1.
    elif args.ssl == 'PiMdl':
        import trainer_unet_PiMdl as trainer
    elif args.ssl == 'MeanTeacher':
        from Trainer import trainer_unet_MT as trainer
    elif args.ssl == 'MeanTeacher+Focal':
        import trainer_unet_MT_Focal as trainer
    elif args.ssl == 'MeanTeacher+Cycle':
        import trainer_unet_CycleMT as trainer
    elif args.ssl == 'MeanTeacher+Bin':
        import trainer_unet_BinMT as trainer
    elif args.ssl == 'MeanTeacher+Bin':
        import trainer_unet_BinMT as trainer
    elif args.ssl == 'VAT':
        import trainer_unet_VAT as trainer
    elif args.ssl == 'MeanTeacher+Topo':
        import trainer_unet_MT_Topology as trainer
    elif args.ssl == 'CutMix':
        import trainer_unet_CutMix as trainer
    elif args.ssl == 'SEM':
        import trainer_unet_SEM as trainer
    elif args.ssl == 'MeanTeacher+NoGeoTform':
        import trainer_unet_MTnoGeoTform as trainer

elif args.loss == 'Dice+Contrastive_loss' \
        or args.loss == 'Dice+CosineContrastive_loss' \
        or args.loss == 'Dice+CosineContrastiveNoExp_loss':
    if args.ssl == 'MeanTeacher':
        import trainer_unet_MT_Constrast as trainer


#### the max number of iterations in a single epoch
# too large number will result in slowing training in particular when unlabeled data is large
if args.AddUnlab == 'None':
    max_itr = args.MaxTrIter
elif args.AddUnlab == 'Crack500':
    max_itr = 30
else:
    max_itr = args.MaxTrIter

#### Load Target Dataset
n_classes = 1
if args.TargetData == 'CrackForest':
    from DataLoader import DataIO_CrackForest as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    # Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)
    Loader.InitDataset_EqLabUnlab(split_filepath=os.path.abspath('Dataset/split_0'),
                                  lab_ratio=lab_ratio)
elif args.TargetData == 'Crack500':
    import DataIO_Crack500 as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif args.TargetData == 'Gaps384':
    import DataIO_Gaps384 as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif 'EM' in args.TargetData:
    import DataIO_EM as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif 'DRIVE' in args.TargetData:
    import DataIO_DRIVE as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif args.TargetData == 'CORN1':
    import DataIO_CORN1 as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

#### Load Target Dataset
elif args.TargetData == 'MITRoad':
    import DataIO_MITRoad as DataIO

    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif args.TargetData == 'MITRoadClean':
    import DataIO_MITRoad as DataIO

    print('Data loading! ...')
    Loader = DataIO.DataIO_Clean(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif args.TargetData == 'MITRoadCleanVecMap':
    import DataIO_MITRoad as DataIO

    print('Data loading! ...')
    Loader = DataIO.DataIO_Clean_VecMap(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)

elif args.TargetData == 'PascalVOC':
    import DataIO_PascalVOC as DataIO

    print('Data loading! ...')
    Loader = DataIO.DataIO(batch_size=args.batchsize, label_percent=args.labelpercent, add_unlab=args.AddUnlab)
    Loader.InitDataset_EqLabUnlab(lab_ratio=lab_ratio)
    n_classes = 21

print('Data loaded!')

# Loader.InitDataset(split_filepath='/vision02/SSL/Simplified/Experiments/DataSplit/split_0')

device = torch.device('cuda:{}'.format(args.GPU) if torch.cuda.is_available() else 'cpu')

## Define Network
if 'PiMdl' in args.ssl or 'MeanTeacher' in args.ssl or 'VAT' in args.ssl:
    Trainer = trainer.Trainer(args=args, Loader=Loader, device=device)
else:
    Trainer = trainer.Trainer(args=args, Loader=Loader, device=device, n_classes=n_classes)

Trainer.DefineNetwork(net_name=args.net, loss_name=args.loss)   # define network architecture
Trainer.DefineOptimizer()       # define optimizer
Trainer.DefineAugmentation()    # define data augmentation

## Determine Running Mode
if args.Mode == 'train':

    ## Saving results path
    base_path = os.path.abspath('./')
    Trainer.PrepareSaveResults(base_path,args)
    if args.SaveRslt:
        Trainer.SaveAllSettings(args)   # save experiment settings

    ## Train Model
    best_val_miou = 0.  # the best validation mean IoU
    for epoch in range(args.Epoch):
        Loader.InitNewEpoch()   # Shuffle Training Data and reset all data sample pointers

        ## Train One Epoch
        loss_tr, perpix_acc_tr, macro_iou_tr, micro_iou_tr = Trainer.TrainOneEpoch(max_itr=max_itr)

        # torch.cuda.empty_cache()

        ## Evaluate One Epoch
        if epoch % args.ValFreq != 0:
            #### Skip validation
            Trainer.PrintTrInfo()

        else:
            #### Do validation
            with torch.no_grad():
                loss_val, perpix_acc_val, macro_iou_val, micro_iou_val = Trainer.ValOneEpoch()

            ## Print Train/Val Info
            Trainer.PrintTrValInfo()

            ## Export Training Info
            if args.SaveRslt:
                Trainer.ExportTensorboard()
                Trainer.UpdateLatestModel()


    ## Export Predictions Results
    # Test All Data
    with torch.no_grad():
        Trainer.RestoreModelByPath(model_epoch='best')
        loss_te, perpix_acc_te, macro_iou_te, micro_iou_te, AIU = Trainer.TestAll_SavePred(exp_fig=True)
        macro_miou_te = np.mean(macro_iou_te)
        Trainer.PrintTeInfo()

    if args.SaveRslt:

        ## Export Results into file
        # result_filepath = os.path.join(result_path,'results.txt')
        Trainer.ExportResults()

elif args.Mode == 'test':
    ## Export Predictions Results
    # Test All Data
    with torch.no_grad():
        # Trainer.RestoreModelByPath(ckpt_path, model_epoch='best')
        Trainer.RestoreModelByPath(ckpt_path=args.Ckpt)

        # Trainer.EvalTrainSet()    # evaluate on train set
        # Trainer.ValOneEpoch()  # evaluate on test set
        Trainer.UpdateBest()

        if args.ExportFigure:
            loss_te, perpix_acc_te, macro_iou_te, micro_iou_te, AIU = Trainer.TestAll_SavePred(exp_fig=True)
        else:
            loss_te, perpix_acc_te, macro_iou_te, micro_iou_te, AIU = Trainer.TestAll_SavePred(exp_fig=False)

        macro_miou_te = np.mean(macro_iou_te)

    print('Inference')
    print('loss/test: {:.2f}'.format(loss_te))
    print('perpix_accuracy/test: {:.2f}%'.format(100 * perpix_acc_te))
    print('Macro Average IoU/test: {:.2f}%'.format(100 * macro_miou_te))
    print('Micro Average IoU/test: {:.2f}%'.format(100 * micro_iou_te))
    print('AIU/test: {:.2f}%'.format(100 * AIU))