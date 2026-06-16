#!/usr/bin/env python3
"""以 AUROC 为目标的超参数网格搜索脚本。"""

import argparse
import csv
import io
import json
import random
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

FIX_SEED = 1024

# 固定超参（与 exp1.sh 一致；temp / lambda_contrastive 对 AUROC 几乎无影响，故固定）
BASE_CONFIG = {
    'model': 'MTCL',
    'model_id': 'dataset1',
    'data': 'ALFA_ad',
    'root_path': './dataset/ALFA_dataset1',
    'data_path': 'ETTh1.csv',
    'target': 'OT',
    'freq': 's',
    'features': 'M',
    'num_nodes': 18,
    'layer_nums': 3,
    'num_experts_list': [4, 4, 4],
    'anomaly_ratio': 5.0,
    'patience': 20,
    'residual_connection': 1,
    'batch_norm': 1,
    'lradj': 'TST',
    'itr': 1,
    'embed': 'timeF',
    'revin': 1,
    'drop': 0.1,
    'no_inter_atten': 0,
    'no_intra_atten': 0,
    'no_contrastive': 0,
    'no_multi': 1,
    'individual': False,
    'metric': 'mse',
    'do_predict': False,
    'test_flop': False,
    'num_workers': 10,
    'use_amp': False,
    'pct_start': 0.4,
    'use_gpu': True,
    'gpu': 0,
    'use_multi_gpu': False,
    'devices': '0',
    'checkpoints': './checkpoints/',
    'patch_size_list': [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2],
    'temp': 200,
    'lambda_contrastive': 1.0,
}

# 参与网格搜索的参数
# - 移除 temp / lambda_contrastive（实验中结果完全一致）
# - 加入 train_epochs（短训练往往 AUROC 更高）
# - anomaly_ratio 不影响 AUROC，保持固定
PARAM_GRID = {
    'seq_len': [96, 192, 384],
    'd_model': [16, 32, 64],
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'train_epochs': [1, 5, 10],
    'k': [2, 3],
}

EXTRA_RESULT_COLUMNS = ['batch_size']
RESULT_COLUMNS = [
    'run_id', 'status', 'auroc', 'accuracy', 'precision', 'recall', 'f1',
    'fpr', 'fnr', 'setting', 'elapsed_sec', *EXTRA_RESULT_COLUMNS, *sorted(PARAM_GRID.keys()),
]


def set_seed():
    random.seed(FIX_SEED)
    torch.manual_seed(FIX_SEED)
    np.random.seed(FIX_SEED)


def normalize_param_value(name, value):
    """统一参数键，便于 --resume 匹配。"""
    if value in (None, ''):
        return ''
    if name in ('learning_rate', 'lambda_contrastive'):
        return f"{float(value):g}"
    if name in ('seq_len', 'd_model', 'train_epochs', 'k', 'temp', 'batch_size'):
        return str(int(float(value)))
    return str(value)


def param_key(params):
    return tuple(normalize_param_value(name, params[name]) for name in sorted(PARAM_GRID.keys()))


def resolve_batch_size(seq_len, d_model):
    """根据序列长度和模型宽度动态调整 batch_size，避免 OOM。"""
    if seq_len >= 384:
        batch_size = 32
    elif seq_len >= 192:
        batch_size = 64
    else:
        batch_size = 128

    if d_model >= 64 and batch_size > 64:
        batch_size = 64
    if seq_len >= 384 and d_model >= 64:
        batch_size = 16
    return batch_size


def build_setting(args, ii=0):
    return '{}_bs{}_sl{}_dm{}_df{}_ln{}_k{}_emb{}_ar{}_lr{}_pt{}_te{}_bn{}_{}'.format(
        args.model_id,
        args.batch_size,
        args.seq_len,
        args.d_model,
        args.d_ff,
        args.layer_nums,
        args.k,
        args.embed,
        args.anomaly_ratio,
        args.learning_rate,
        args.patience,
        args.train_epochs,
        args.batch_norm,
        ii,
    )


def make_model_id(base_id, params, batch_size):
    tag = '_'.join(
        f"{key}{normalize_param_value(key, params[key]).replace('.', 'p')}"
        for key in sorted(PARAM_GRID.keys())
    )
    return f"{base_id}_gs_bs{batch_size}_{tag}"


def build_args(config):
    args = argparse.Namespace(**config)
    args.is_training = 1
    args.use_gpu = torch.cuda.is_available() and args.use_gpu
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    return args


def load_completed_runs(results_path):
    if not results_path.exists():
        return set()

    completed = set()
    with results_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'FAILED':
                continue
            if not row.get('auroc') or row['auroc'] == 'nan':
                continue
            params = {name: row[name] for name in PARAM_GRID if name in row}
            if len(params) == len(PARAM_GRID):
                completed.add(param_key(params))
    return completed


def append_result(results_path, row):
    write_header = not results_path.exists()
    with results_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_best(results_path, summary_path):
    if not results_path.exists():
        return None

    rows = []
    with results_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'FAILED':
                continue
            try:
                auroc = float(row['auroc'])
            except (TypeError, ValueError):
                continue
            if np.isnan(auroc):
                continue
            rows.append(row)

    if not rows:
        return None

    best = max(rows, key=lambda x: float(x['auroc']))
    summary = {
        'objective': 'AUROC',
        'best_auroc': float(best['auroc']),
        'best_params': {
            key: normalize_param_value(key, best[key]) for key in sorted(PARAM_GRID.keys())
        },
        'batch_size': int(best.get('batch_size', 0) or 0),
        'setting': best['setting'],
        'run_id': best['run_id'],
        'accuracy_with_pa': float(best['accuracy']),
        'f1_with_pa': float(best['f1']),
        'successful_runs': len(rows),
        'search_space': PARAM_GRID,
    }

    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def clean_experiment_artifacts(args, setting):
    """网格搜索会产生大量 checkpoint，每次实验后清理以节省磁盘。"""
    checkpoint_dir = Path(args.checkpoints) / setting
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    for root in ('./test_results', './label_results'):
        artifact_dir = Path(root) / setting
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)


def run_single_experiment(params, output_dir, run_id, clean_artifacts=True):
    batch_size = resolve_batch_size(int(params['seq_len']), int(params['d_model']))

    config = BASE_CONFIG.copy()
    config.update(params)
    config['pred_len'] = config['seq_len']
    config['d_ff'] = config['d_model']
    config['batch_size'] = batch_size
    config['patience'] = max(int(params['train_epochs']), 5)
    config['model_id'] = make_model_id(BASE_CONFIG['model_id'], params, batch_size)

    set_seed()
    torch.cuda.empty_cache()
    args = build_args(config)
    setting = build_setting(args, ii=0)

    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run{run_id:04d}_{setting}.log"

    start = time.time()
    log_buffer = io.StringIO()
    metrics = None

    try:
        with log_file.open('w', encoding='utf-8') as lf:
            lf.write(f"Params: {json.dumps(params, ensure_ascii=False)}\n")
            lf.write(f"Resolved batch_size: {batch_size}\n")
            lf.write(f"Setting: {setting}\n\n")
            lf.flush()

            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                exp = Exp_Anomaly_Detection(args)
                print('>>>>>>>start training')
                exp.train(setting)
                print('>>>>>>>testing')
                metrics = exp.test(setting)

            lf.write(log_buffer.getvalue())
            lf.flush()
    finally:
        if clean_artifacts:
            clean_experiment_artifacts(args, setting)
        torch.cuda.empty_cache()

    elapsed = time.time() - start

    row = {
        'run_id': run_id,
        'status': 'OK',
        'auroc': f"{metrics['auroc']:.6f}",
        'accuracy': f"{metrics['accuracy']:.6f}",
        'precision': f"{metrics['precision']:.6f}",
        'recall': f"{metrics['recall']:.6f}",
        'f1': f"{metrics['f1']:.6f}",
        'fpr': f"{metrics['fpr']:.6f}",
        'fnr': f"{metrics['fnr']:.6f}",
        'setting': setting,
        'elapsed_sec': f"{elapsed:.1f}",
        'batch_size': batch_size,
    }
    for key in sorted(PARAM_GRID.keys()):
        row[key] = params[key]
    return row


def build_fail_row(run_id, params, batch_size, error_msg):
    row = {
        'run_id': run_id,
        'status': 'FAILED',
        'auroc': 'nan',
        'accuracy': '',
        'precision': '',
        'recall': '',
        'f1': '',
        'fpr': '',
        'fnr': '',
        'setting': 'FAILED',
        'elapsed_sec': '0',
        'batch_size': batch_size,
        'error': error_msg,
    }
    for key in sorted(PARAM_GRID.keys()):
        row[key] = params[key]
    for col in RESULT_COLUMNS:
        row.setdefault(col, '')
    return row


def parse_args():
    parser = argparse.ArgumentParser(description='Grid search hyperparameters by AUROC')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./logs_exp/大论文/grid_search_v2',
        help='directory to save csv/logs/summary',
    )
    parser.add_argument('--max_runs', type=int, default=None, help='limit number of new runs (for quick testing)')
    parser.add_argument('--resume', action='store_true', help='skip successful parameter sets already in results.csv')
    parser.add_argument(
        '--keep_checkpoints',
        action='store_true',
        help='keep checkpoint/test artifacts after each run (default: delete to save disk)',
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.csv'
    summary_path = output_dir / 'best_params.json'

    completed = load_completed_runs(results_path) if cli_args.resume else set()
    param_names = sorted(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]
    all_combinations = [dict(zip(param_names, values)) for values in product(*param_values)]
    pending = [p for p in all_combinations if param_key(p) not in completed]

    print('Grid search objective: AUROC')
    print(f'Total combinations: {len(all_combinations)}')
    print(f'Already completed: {len(completed)}')
    print(f'Pending: {len(pending)}')
    print(f'Output directory: {output_dir.resolve()}')
    print(f'Fixed params: temp={BASE_CONFIG["temp"]}, lambda_contrastive={BASE_CONFIG["lambda_contrastive"]}')
    print(f'Search space: {json.dumps(PARAM_GRID, ensure_ascii=False)}')

    run_id = 0
    if results_path.exists():
        with results_path.open(newline='', encoding='utf-8') as f:
            run_id = sum(1 for _ in csv.DictReader(f))
    executed = 0

    for params in all_combinations:
        key = param_key(params)
        if key in completed:
            print(f'[skip] {params}')
            continue

        if cli_args.max_runs is not None and executed >= cli_args.max_runs:
            print(f'\nReached --max_runs={cli_args.max_runs}, stopping.')
            break

        run_id += 1
        executed += 1
        batch_size = resolve_batch_size(int(params['seq_len']), int(params['d_model']))
        print(f'\n[{executed}/{len(pending)}] Running: {params} | batch_size={batch_size}')

        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            row = run_single_experiment(
                params, output_dir, run_id, clean_artifacts=not cli_args.keep_checkpoints
            )
            append_result(results_path, row)
            completed.add(key)
            summarize_best(results_path, summary_path)
            print(
                f"  OK | AUROC={row['auroc']} | "
                f"Acc={row['accuracy']} Prec={row['precision']} "
                f"Rec={row['recall']} F1={row['f1']}"
            )
        except Exception as exc:
            err = traceback.format_exc()
            print(f'  FAILED: {exc}')
            fail_row = build_fail_row(run_id, params, batch_size, str(exc))
            append_result(results_path, fail_row)

            fail_log = log_dir / f'run{run_id:04d}_FAILED.log'
            fail_log.write_text(
                f"Params: {json.dumps(params, ensure_ascii=False)}\n"
                f"batch_size: {batch_size}\n\n{err}",
                encoding='utf-8',
            )
            torch.cuda.empty_cache()

    summary = summarize_best(results_path, summary_path)
    if summary is None:
        print('\nNo successful runs recorded.')
        return

    print('\n' + '=' * 60)
    print('Grid search finished.')
    print(f"Best AUROC: {summary['best_auroc']:.6f}")
    print(f"Best params: {json.dumps(summary['best_params'], ensure_ascii=False)}")
    print(f"Best batch_size: {summary['batch_size']}")
    print(f"Saved to: {summary_path.resolve()}")


if __name__ == '__main__':
    main()
