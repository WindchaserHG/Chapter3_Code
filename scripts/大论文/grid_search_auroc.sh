#!/bin/bash
# 以 AUROC 为目标的超参数网格搜索（v2：精简搜索空间 + 动态 batch_size + train_epochs）

cd "$(dirname "$0")/../.." || exit 1

mkdir -p ./logs_exp/大论文/grid_search_v2

python -u scripts/大论文/grid_search_auroc.py \
  --output_dir ./logs_exp/大论文/grid_search_v2 \
  --resume \
  "$@"
