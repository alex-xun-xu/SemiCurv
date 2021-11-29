#### Run experiment for fully supervised baseline

GPU=1
seed_split=1  # data split seed
Mode=train  # training (train) or testing (test) mode

Epoch=1000
lr=1e-3 # init learning rate
bs=12  # batchsize
SaveRslt=1  # save experiment results
labelpercent=0.01 # labelled ratio (0 to 1)
loss=Dice+ConsistMSE_loss # supervised loss
ssl=MeanTeacher # semi-supervised learner
EmaAlpha=0.999
Gamma=5
RampupEpoch=50
RampupType=Exp


python main.py --GPU $GPU --Mode $Mode --Epoch $Epoch --LearningRate $lr --batchsize $bs --SaveRslt $SaveRslt \
 --labelpercent $labelpercent --seed_split $seed_split --loss $loss --ssl $ssl --Alpha $EmaAlpha --Gamma $Gamma \
 --RampupEpoch $RampupEpoch --RampupType $RampupType