#!/bin/bash
# 创建 logs 目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs/AnomalyDetection" ]; then
    mkdir ./logs/AnomalyDetection
fi

# 设置模型和数据路径
model_name=PathFormer
root_path_name=./dataset/ALFA/
data_path_name=ALFA.csv
model_id_name=ALFA_ad
data_name=ALFA_ad

# 遍历不同的异常比率和序列长度
for anomaly_ratio in 2 3 4 5 10; do
    for seq_len in 96 192; do
        # 构建日志文件路径
        log_file="logs/AnomalyDetection/${model_name}_${model_id_name}_${seq_len}_${anomaly_ratio}.log"

        # 确保日志目录存在
        mkdir -p "$(dirname "$log_file")"

        # 运行 Python 脚本并将输出重定向到日志文件
        python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$anomaly_ratio \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $seq_len \
        --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 \
        --num_nodes 18 \
        --layer_nums 3 \
        --batch_norm 1 \
        --residual_connection 0\
        --k 3\
        --d_model 16 \
        --d_ff 64 \
        --train_epochs 100\
        --patience 10\
        --lradj 'TST'\
        --itr 1 \
        --anomaly_ratio $anomaly_ratio \
        --batch_size 64 --learning_rate 0.0005 > "$log_file"
    done
done

# 若释放前要保存环境并命名
export $(cat /proc/1/environ |tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name ALFA_AD_0914

# 若释放前不需要保存环境 
# export $(cat /proc/1/environ |tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node