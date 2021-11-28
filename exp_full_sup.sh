#### Run experiment for fully supervised baseline

GPU=0
Epoch=200
lr=1e-3 # init learning rate
bs=8  # batchsize
SaveRslt=1  # save experiment results
labelpercent=0.01 # labelled ratio (0 to 1)
seed_split=1  # data split seed
loss=Diceloss # supervised loss
ssl=FullSup # semi-supervised learner

python main.py --GPU $GPU --Epoch $Epoch --LearningRate $lr --batchsize $bs --SaveRslt $SaveRslt \
 --labelpercent $labelpercent --seed_split $seed_split --loss $loss --ssl $ssl
