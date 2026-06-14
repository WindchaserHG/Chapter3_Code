#!/bin/bash
# 实验目的：最优参数下跑实验,与其他方法进行对比
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建 logs 目录
if [ ! -d "./logs_exp" ]; then
    mkdir ./logs_exp
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs_exp/Exp10" ]; then
    mkdir ./logs_exp/Exp10
fi

# 设置模型和数据路径
model_name=PathFormer
root_path_name=./dataset/ALFA_dataset1
model_id_name=dataset1
data_name=ALFA_ad
anomaly_ratio=10
d_model=16
seq_len=96
learning_rate=0.00001
train_epochs=100
patience=10
batch_size=128
residual_connection=1
batch_norm=1
temp=10
lambda_contrastive=1

for anomaly_ratio in 1 5 10; do
    for d_model in 8 16 64; do

        # 构建日志文件路径
        log_file="logs_exp/Exp10/${model_id_name}_d_model${d_model}_ar${anomaly_ratio}.log"

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
        --k 3 \
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
done


root_path_name=./dataset/ALFA_dataset2
model_id_name=dataset2

for anomaly_ratio in 1 5 10; do
    for d_model in 8 16 64; do

        # 构建日志文件路径
        log_file="logs_exp/Exp10/${model_id_name}_d_model${d_model}_ar${anomaly_ratio}.log"

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
        --k 3 \
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
done

root_path_name=./dataset/ALFA_dataset3
model_id_name=dataset3

for anomaly_ratio in 1 5 10; do
    for d_model in 8 16 64; do

        # 构建日志文件路径
        log_file="logs_exp/Exp10/${model_id_name}_d_model${d_model}_ar${anomaly_ratio}.log"

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
        --k 3 \
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
done

root_path_name=./dataset/ALFA_dataset4
model_id_name=dataset4

for anomaly_ratio in 1 5 10; do
    for d_model in 8 16 64; do

        # 构建日志文件路径
        log_file="logs_exp/Exp10/${model_id_name}_d_model${d_model}_ar${anomaly_ratio}.log"

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
        --k 3 \
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
done
