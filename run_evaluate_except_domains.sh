#!/bin/bash
# get all filename in folder
 
# 配合config 中 exp_domains, 记得加/
# 0819
# path='experiments_Xdomain/attraction_0819_FromSrcatch_sd11_lr0.001_bs2_ga1/'
# path='experiments_Xdomain/hotel_0819_FromSrcatch_sd11_lr0.001_bs2_ga1/'
# path='experiments_Xdomain/restaurant_0819_FromSrcatch_sd11_lr0.001_bs2_ga1/'
path='experiments_Xdomain/taxi_0819_FromSrcatch_sd11_lr0.001_bs2_ga1/'
# path='experiments_Xdomain/train_0819_FromSrcatch_sd11_lr0.001_bs2_ga1/'
#
# 0818
# path='experiments_Xdomain/hotel_0818_FS_sd11_lr5e-05_bs2_ga1/'
# path='experiments_Xdomain/train_0818_FS_sd11_lr5e-05_bs2_ga1/'
# path='experiments_Xdomain/attraction_0818_FS_sd11_lr5e-05_bs2_ga1/'
# path='experiments_Xdomain/restaurant_0818_FS_sd11_lr5e-05_bs2_ga1/'
# path='experiments_Xdomain/taxi_0818_FS_sd11_lr5e-05_bs2_ga1/'
#
# path='experiments_Xdomain/except-train_0806_sd11_lr5e-05_bs2_ga16/'
# path='experiments_Xdomain/except-attraction_0805_sd11_lr0.0001_bs2_ga16/'
# path='experiments_Xdomain/except-restaurant_0805_sd11_lr0.0001_bs2_ga16/'
# path='experiments_Xdomain/except-taxi_0805_sd11_lr0.0001_bs2_ga16/'
# path='experiments_Xdomain/hotel_0806-ft_sd11_lr0.0001_bs2_ga16/'
# path='experiments_Xdomain/attraction_0806-ft_sd11_lr0.0001_bs2_ga1/'

 
#获取文件夹下所有文件
files=$(ls $path)
 
#遍历文件夹中文件
# count=0
for filename in $files
do
    load_path=$path$filename'/'
    # 0818/0819
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=0 exp_domains=hotel log_path=logs_xd && echo $load_path is done
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=0 exp_domains=train log_path=logs_xd && echo $load_path is done
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=0 exp_domains=attraction log_path=logs_xd && echo $load_path is done
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=0 exp_domains=restaurant log_path=logs_xd && echo $load_path is done
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=2 exp_domains=taxi log_path=logs_xd && echo $load_path is done
    
    #
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=2 exp_domains=except,attraction && echo $load_path is done
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=2 exp_domains=except,restaurant && echo $load_path is done
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=2 exp_domains=except,taxi && echo $load_path is done
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -mode test -cfg eval_load_path=$load_path cuda_device=0 exp_domains=attraction && echo $load_path is done # hotel finetune


done


# load_path='experiments/all_with_nodelex_resp_sd11_lr0.0001_bs2_ga16/epoch33_trloss0.65_gpt2/'
# log_path=$load_path'logFFFT.file'
# CUDA_VISIBLE_DEVICES=3 nohup python train.py -mode test -cfg eval_load_path=$load_path model_output='model_output_e2e_FFFT.json' use_true_prev_aspn=False >$log_path &