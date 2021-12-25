#!/usr/bin/env bash

#GPUs
gpus=0

#Set paths
checkpoint_root=/media/lidan/ssd2/ChangeFormer/checkpoints
vis_root=/media/lidan/ssd2/ChangeFormer/vis
data_name=LEVIR #LEVIR, DSIFN


img_size=256                #Choices=128, 256, 512
batch_size=16               #Choices=8, 16, 32, 64
lr=0.001         
max_epochs=200

net_G=SiamUnet_diff        #Choices=ChangeFormerV1/2/3/4/5

lr_policy=linear
optimizer=sgd               #Choices: sgd, adam, adamw
loss=ce                     #Choices: ce, fl (Focal Loss), miou
multi_scale_train=False
multi_scale_infer=False
shuffle_AB=False

#Train and Validation splits
split=train         #trainval
split_val=test      #test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}

CUDA_VISIBLE_DEVICES=0 python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G}  --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}