#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=ChangeFormerV4
split=test
vis_root=/media/lidan/ssd2/ChangeFormer/vis
project_name=CD_ChangeFormerV4_LEVIR_b8_lr0.01_sgd_train_val_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False
checkpoints_root=/media/lidan/ssd2/ChangeFormer/checkpoints
checkpoint_name=best_ckpt.pt
img_size=128


python eval_cd.py --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


