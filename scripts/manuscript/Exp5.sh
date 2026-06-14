#!/bin/bash
# 实验目的：探寻d_model的影响
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建 logs 目录
if [ ! -d "./logs_exp" ]; then
    mkdir ./logs_exp
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs_exp/Exp5" ]; then
    mkdir ./logs_exp/Exp5
fi

# 设置模型和数据路径
model_name=PathFormer
root_path_name=./dataset/ALFA_dataset1
model_id_name=dataset1_exp5
data_name=ALFA_ad
anomaly_ratio=3
d_model=16
seq_len=96
learning_rate=0.0001
train_epochs=100
patience=20
batch_size=128
residual_connection=1
batch_norm=1
temp=1

# 遍历不同的异常比率和序列长度
for d_model in 8 16 32 64; do
    # 构建日志文件路径
    log_file="logs_exp/Exp5/${model_id_name}_dmodel${d_model}.log"

    # 确保日志目录存在
    mkdir -p "$(dirname "$log_file")"

    # 运行 Python 脚本并将输出重定向到日志文件
    python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --model_id $model_id_name \
    --model $model_name \
    --data $data_name \
    --loss_save_name "${model_id_name}_dmodel${d_model}.csv" \
    --features M \
    --seq_len $seq_len \
    --pred_len $seq_len \
    --num_nodes 18 \
    --layer_nums 3 \
    --batch_norm $batch_norm \
    --residual_connection $residual_connection\
    --k 3 \
    --temp $temp \
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

# 若释放前要保存环境并命名
# export $(cat /proc/1/environ |tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name ALFA_AD_0914

# 若释放前不需要保存环境 
# export $(cat /proc/1/environ |tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node