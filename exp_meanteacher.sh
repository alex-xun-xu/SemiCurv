#### Run experiment for fully supervised baseline

GPU=1
Epoch=2000
lr=1e-3 # init learning rate
bs=8  # batchsize
SaveRslt=1  # save experiment results
labelpercent=0.01 # labelled ratio (0 to 1)
seed_split=1  # data split seed
loss=Dice+ConsistMSE_loss # supervised loss
ssl=MeanTeacher # semi-supervised learner
EmaAlpha=0.999
Gamma=5
RampupEpoch=300
RampupType=Exp


python main.py --GPU $GPU --Epoch $Epoch --LearningRate $lr --batchsize $bs --SaveRslt $SaveRslt \
 --labelpercent $labelpercent --seed_split $seed_split --loss $loss --ssl $ssl --Alpha $EmaAlpha --Gamma $Gamma \
 --RampupEpoch $RampupEpoch --RampupType $RampupType