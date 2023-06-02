#!/bin/bash

# dataset_choice='all'
# python data_process_ll.py
world_sizes=4
batch_sizes=4
dp_delta_1=0.0000001
dp_delta_2=0.0001
dp_delta_3=0
dataset_choice='all'
root='./headneck_male'
root0='./headneck_male'
root3='./headneck_oropharynx'
root2='./headneck_female'
root1='./headneck_larynx'
epoch_count=1
noniid='noniid'
setting='ours'
weight=1
standalone='false'

for i in `seq 0 3`
do
    {
    python save_client_main_test.py \
        --setting $setting \
        --noniid $noniid \
        --root $root \
        --root0 $root0 \
        --root1 $root1 \
        --root2 $root2 \
        --root3 $root3 \
        --low_bound_ensure 1 \
        --ours_dir 1 \
        --ours_loss 1 \
        --ours_pccs 1 \
        --q 0.75 \
        --ours_dis 0 \
        --weight $weight \
        --DP no \
        --gpu_id '1' \
        --dataset_named 'petct' \
        --model_choice 'navie_model_concat' \
        --dataset_choice $dataset_choice \
        --backend gloo \
        --epoch_count $epoch_count \
        --init_method tcp://localhost:18116 \
        --world_size $world_sizes \
        --rank $i \
        --standalone $standalone \
        --batchSize $batch_sizes
    }&
done



