GPU=0
Mode=test
Ckpt=/home/xuxun/Dropbox/GitHub/SemiCurv/Results/CrackForest/MeanTeacher/MeanTeacher_ResUnet_Dice+ConsistMSE_loss_ep-1000_m-0.01_2021-11-29_01-50-48/ckpt/model_epoch-best.pt

python main.py --GPU $GPU --Mode $Mode --Ckpt $Ckpt