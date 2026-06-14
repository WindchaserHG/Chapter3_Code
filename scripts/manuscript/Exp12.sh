#!/bin/bash
# 实验目的：消融实验
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建 logs 目录
if [ ! -d "./logs_exp" ]; then
    mkdir ./logs_exp
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs_exp/Exp12" ]; then
    mkdir ./logs_exp/Exp12
fi

# 设置模型和数据路径
model_name=PathFormer
model_id_name=dataset1
data_name=ALFA_ad
anomaly_ratio=5
d_model=16
seq_len=96
learning_rate=0.00001
train_epochs=100
patience=10
batch_size=128
residual_connection=1
batch_norm=1
k=2
temp=10
lambda_contrastive=1
no_inter_atten=0
no_intra_atten=0
no_contrastive=0

for model_id_name in dataset1 dataset3 dataset4; do
    for temp in 10 100 200 1000; do
        root_path_name="./dataset/ALFA_${model_id_name}"

        # 构建日志文件路径
        log_file="logs_exp/Exp12/${model_id_name}_temp${temp}.log"

        # 确保日志目录存在
        mkdir -p "$(dirname "$log_file")"

        # 运行 Python 脚本并将输出重定向到日志文件
        python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --model_id $model_id_name \
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
        --no_inter_atten $no_inter_atten \
        --no_intra_atten $no_intra_atten \
        --no_contrastive $no_contrastive \
        --learning_rate $learning_rate > "$log_file"
    done
done
