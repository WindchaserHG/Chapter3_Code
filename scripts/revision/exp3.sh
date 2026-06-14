#!/bin/bash
# 实验目的：噪声数据集上的实验
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建 logs 目录
if [ ! -d "./logs_exp" ]; then
    mkdir ./logs_exp
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs_exp/exp3" ]; then
    mkdir ./logs_exp/exp3
fi

# 设置模型和数据路径
model_name=MTCL
root_path_name=./dataset/ALFA_dataset1
model_id_name=dataset1
data_name=ALFA_ad
anomaly_ratio=5
d_model=16
seq_len=96
learning_rate=0.00001
train_epochs=100
patience=20
batch_size=128
residual_connection=1
batch_norm=1
temp=200
lambda_contrastive=1
k=2
no_multi=1

for sub_path in gau_005 gau_01 gau_05 uni_005 uni_01 uni_05; do

    root_path_name=./dataset/noisy/${sub_path}

    # 构建日志文件路径
    log_file="logs_exp/exp3/${model_id_name}_ns${sub_path}_ad${anomaly_ratio}_${model_name}_d${d_model}_s${seq_len}_lr${learning_rate}_e${train_epochs}_p${patience}_b${batch_size}_rc${residual_connection}_bn${batch_norm}_temp${temp}_lc${lambda_contrastive}_k${k}.log"

    # 确保日志目录存在
    mkdir -p "$(dirname "$log_file")"

    # 运行 Python 脚本并将输出重定向到日志文件
    python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --model_id $sub_path \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $seq_len \
    --num_nodes 18 \
    --layer_nums 3 \
    --batch_norm $batch_norm \
    --residual_connection $residual_connection \
    --k $k \
    --temp $temp \
    --lambda_contrastive $lambda_contrastive \
    --d_model $d_model \
    --d_ff $d_model \
    --train_epochs $train_epochs \
    --patience $patience \
    --lradj 'TST' \
    --itr 1 \
    --anomaly_ratio $anomaly_ratio \
    --batch_size $batch_size \
    --learning_rate $learning_rate > "$log_file"
done
