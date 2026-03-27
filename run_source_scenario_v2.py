"""
run_source_scenario_v2.py — 源端场景生成 (改进版)
===================================================
改进点:
  1. 按天切割 (stride=24) 替代滑动窗口 (stride=1)
     → 每个样本都是完整日历天, PV/GHI的日内模式不再被打乱
  2. HIDDEN_DIM 增大到 48
  3. 评估方法与参考论文一致

用法: python run_source_scenario_v2.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper.mctimegan import MCTimeGAN
from helper.vine_copula import VineCopulaModel

# ===================== 参数配置 =====================
DATABASE_PATH  = "data/Database.csv"
LABEL_CSV      = "helper/data/raw_labels/source_labels.csv"
HORIZON        = 24
HIDDEN_DIM     = 48
NUM_LAYERS     = 3
EPOCHS         = 60000
BATCH_SIZE     = 128         # 恢复128, batch=32梯度太noisy
LR             = 5e-4
LR_D           = 2e-4
D_THRESHOLD    = 0.15
USE_LR_SCHED   = True
USE_VINE       = True
W_CORR         = 80          # 从50温和提升到80 (150太激进)
CKPT_INTERVAL  = 200
OUTPUT_DIR     = "helper/synthetic_data"
CKPT_DIR       = "helper/checkpoints"

SRC_COLS = ['Wind_speed', 'GHI', 'Temperature', 'PV_production', 'Wind_production']
VAR_UNITS = ['m/s', 'W/m²', '°C', 'kW', 'kW']
# ====================================================


def load_day_based_data(db_path, label_csv_path):
    """
    按天切割数据 (stride=24), 替代 data_processing.py 的滑动窗口
    
    关键区别:
      滑动窗口: 样本0=hour[0:24], 样本1=hour[1:25], ... → 样本起点不对应午夜
      按天切割: 样本0=Day1[00:00-23:00], 样本1=Day2[00:00-23:00] → 每个样本都是完整天
    
    这保证了: 第0个时间步=00:00, 第12个=12:00, 第23个=23:00
    → GHI/PV的钟形日内模式可以被正确学习
    """
    print("[数据加载] 按天切割 (stride=24)...")
    
    # 读取原始数据
    df = pd.read_csv(db_path, index_col=0)
    df['Time'] = pd.to_datetime(df['Time'].str.replace('-T', ' '))
    df.set_index('Time', inplace=True)
    df_h = df.resample("1h").mean().dropna()
    
    # 读取标签
    labels_all = pd.read_csv(label_csv_path)['weather_label'].values
    
    # 按天切割
    df_h['date'] = df_h.index.date
    source_days = []
    label_days = []
    
    dates = sorted(df_h['date'].unique())
    idx = 0
    for date in dates:
        day_data = df_h[df_h['date'] == date][SRC_COLS].values
        day_labels = labels_all[idx:idx+len(day_data)]
        idx += len(day_data)
        
        if len(day_data) == 24:
            source_days.append(day_data)
            # 取这天中出现最多的标签作为该天的标签
            if len(day_labels) >= 24:
                label_val = int(np.median(day_labels[:24]))
            else:
                label_val = 0
            # 扩展为 (24, 1) — MC-TimeGAN需要逐时间步的标签
            label_days.append(np.full((24, 1), label_val))
    
    source_data = np.array(source_days, dtype=np.float32)   # (N_days, 24, 5)
    label_data = np.array(label_days, dtype=np.float32)     # (N_days, 24, 1)
    
    print(f"  完整天数: {len(source_data)}")
    print(f"  形状: data={source_data.shape}, labels={label_data.shape}")
    
    # 标签分布
    day_labels_flat = label_data[:, 0, 0].astype(int)
    for lv in range(int(day_labels_flat.max()) + 1):
        cnt = (day_labels_flat == lv).sum()
        print(f"  标签{lv}: {cnt}天 ({cnt/len(day_labels_flat)*100:.1f}%)")
    
    return source_data, label_data


def normalize_data(data):
    """MinMax归一化到[0,1], 返回归一化数据和缩放参数"""
    N, T, D = data.shape
    flat = data.reshape(-1, D)
    scaler = MinMaxScaler()
    flat_norm = scaler.fit_transform(flat)
    data_norm = flat_norm.reshape(N, T, D).astype(np.float32)
    return data_norm, scaler


def denormalize_data(data_norm, scaler):
    N, T, D = data_norm.shape
    flat = data_norm.reshape(-1, D)
    flat_orig = scaler.inverse_transform(flat)
    return flat_orig.reshape(N, T, D)


def evaluate_per_variable(ori, gen, var_name, var_idx):
    """按变量评估, 与参考论文方法一致"""
    # 每个场景的日均值
    ori_daily = ori[:, :, var_idx].mean(axis=1)  # (N,)
    gen_daily = gen[:, :, var_idx].mean(axis=1)  # (N,)
    
    # 排序对齐 (分位数匹配)
    o_s = np.sort(ori_daily)
    g_s = np.sort(gen_daily)
    
    mae = mean_absolute_error(o_s, g_s)
    rmse = np.sqrt(mean_squared_error(o_s, g_s))
    r2 = r2_score(o_s, g_s)
    
    # MAPE (排除极小值)
    threshold = max(np.percentile(np.abs(o_s), 10), 1.0)
    mask = np.abs(o_s) > threshold
    mape = np.mean(np.abs((o_s[mask] - g_s[mask]) / o_s[mask])) * 100 if mask.sum() > 0 else 0
    
    # JS divergence
    bins = 100
    rng = (min(o_s.min(), g_s.min()), max(o_s.max(), g_s.max()))
    h_o, _ = np.histogram(o_s, bins=bins, range=rng, density=True)
    h_g, _ = np.histogram(g_s, bins=bins, range=rng, density=True)
    h_o, h_g = h_o + 1e-10, h_g + 1e-10
    h_o, h_g = h_o / h_o.sum(), h_g / h_g.sum()
    m = 0.5 * (h_o + h_g)
    js = 0.5 * entropy(h_o, m) + 0.5 * entropy(h_g, m)
    
    # FID (1D)
    mu_diff = (ori_daily.mean() - gen_daily.mean()) ** 2
    fid = mu_diff + ori_daily.var() + gen_daily.var() - 2 * np.sqrt(ori_daily.var() * gen_daily.var())
    
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2, 'JS': js, 'FID': fid}


def evaluate_checkpoint(model, data_norm, labels, scaler, ori_raw):
    """评估一个checkpoint"""
    gen_norm = model.transform(data_norm.shape, cond=labels)
    gen_raw = denormalize_data(gen_norm, scaler)
    
    results = {}
    for j, name in enumerate(SRC_COLS):
        results[name] = evaluate_per_variable(ori_raw, gen_raw, name, j)
    
    # 综合得分
    avg_mape = np.mean([v['MAPE'] for v in results.values()])
    avg_r2 = np.mean([v['R2'] for v in results.values()])
    score = avg_mape + (1 - avg_r2) * 100  # 越小越好
    
    return results, gen_raw, score


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(CKPT_DIR):
        import shutil; shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {dev}")
    
    # ===== Step 1: 按天加载数据 =====
    print("\n" + "=" * 60)
    print("Step 1: 按天切割加载数据")
    print("=" * 60)
    source_raw, labels = load_day_based_data(DATABASE_PATH, LABEL_CSV)
    # source_raw: (1096, 24, 5), labels: (1096, 24, 1)
    
    # 归一化
    data_norm, scaler = normalize_data(source_raw)
    
    # 打乱
    perm = np.random.RandomState(42).permutation(len(data_norm))
    data_norm = data_norm[perm]
    labels = labels[perm]
    source_raw_shuffled = source_raw[perm]
    
    print(f"  归一化后: {data_norm.shape}, 范围: [{data_norm.min():.3f}, {data_norm.max():.3f}]")
    
    # ===== Step 2: Vine-Copula =====
    vine = None
    if USE_VINE:
        print("\n" + "=" * 60)
        print("Step 2: Vine-Copula 五变量联合分布建模")
        print("=" * 60)
        vine = VineCopulaModel()
        vine.fit(source_raw.reshape(-1, 5))
        vine.validate(source_raw.reshape(-1, 5), var_names=SRC_COLS)
    
    # ===== Step 3: 训练 =====
    print("\n" + "=" * 60)
    print(f"Step 3: MC-TimeGAN 训练 (HIDDEN_DIM={HIDDEN_DIM}, 按天切割)")
    print("=" * 60)
    
    model = MCTimeGAN(
        input_features=data_norm.shape[-1],      # 5
        input_conditions=labels.shape[-1],        # 1
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, len(data_norm)),
        learning_rate=LR,
        lr_d=LR_D,
        d_threshold=D_THRESHOLD,
        use_lr_scheduler=USE_LR_SCHED,
        w_corr=W_CORR,                           # 相关性损失权重
        vine_model=vine,
    ).to(dev)
    
    model.checkpoint_dir = CKPT_DIR
    model.checkpoint_interval = CKPT_INTERVAL
    
    model.fit(data_norm, cond_labels=labels)
    
    # ===== Step 4: 评估所有checkpoint =====
    print("\n" + "=" * 60)
    print("Step 4: 评估所有checkpoint")
    print("=" * 60)
    
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pth")))
    print(f"  找到 {len(ckpt_files)} 个checkpoint")
    
    all_results = {}
    for ckpt_path in ckpt_files:
        ep = int(os.path.basename(ckpt_path).replace("epoch_", "").replace(".pth", ""))
        model.load_checkpoint(ckpt_path)
        metrics, gen_raw, score = evaluate_checkpoint(model, data_norm, labels, scaler, source_raw_shuffled)
        all_results[ep] = {'metrics': metrics, 'gen_raw': gen_raw, 'score': score}
        
        print(f"\n  Epoch {ep}:")
        for name in SRC_COLS:
            m = metrics[name]
            print(f"    {name:20s}: MAPE={m['MAPE']:5.2f}%, R²={m['R2']:.4f}, JS={m['JS']:.4f}")
        print(f"    综合得分: {score:.2f}")
    
    # ===== Step 5: 最优epoch =====
    best_ep = min(all_results, key=lambda e: all_results[e]['score'])
    best = all_results[best_ep]
    print(f"\n{'='*60}")
    print(f"  最优epoch: {best_ep}, 综合得分: {best['score']:.2f}")
    print(f"{'='*60}")
    
    # 保存最优结果
    np.save(os.path.join(OUTPUT_DIR, "source_generated.npy"), best['gen_raw'])
    np.save(os.path.join(OUTPUT_DIR, "source_original.npy"), source_raw_shuffled)
    np.save(os.path.join(OUTPUT_DIR, "source_labels.npy"), labels)
    
    # ===== Step 6: 输出对比表 =====
    print(f"\n{'='*100}")
    print(f"  指标对比 (最优epoch={best_ep} vs 参考论文)")
    print(f"{'='*100}")
    
    ref = {
        'Wind_production': {'MAE': 30.69, 'MAPE': 2.53, 'RMSE': 42.99, 'R2': 0.9932, 'FID': 34.24, 'JS': 0.0319},
        'PV_production':   {'MAE': 9.53,  'MAPE': 1.97, 'RMSE': 18.82, 'R2': 0.9934, 'FID': 105.09, 'JS': 0.0243},
    }
    
    print(f"{'变量':20s} | {'MAE':>8s} {'MAPE%':>8s} {'RMSE':>8s} {'R²':>8s} {'FID':>8s} {'JS':>8s}")
    print("-" * 100)
    for name in SRC_COLS:
        m = best['metrics'][name]
        line = f"{name:20s} | {m['MAE']:8.2f} {m['MAPE']:8.2f} {m['RMSE']:8.2f} {m['R2']:8.4f} {m['FID']:8.2f} {m['JS']:8.4f}"
        if name in ref:
            r = ref[name]
            line += f"  [论文: MAE={r['MAE']:.1f} MAPE={r['MAPE']:.1f}% R²={r['R2']:.4f}]"
        print(line)
    
    # ===== 图表生成 =====
    gen_raw = best['gen_raw']
    ori_raw = source_raw_shuffled
    hours = np.arange(24)
    
    # 图1: 逐小时均值曲线对比
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    for j in range(5):
        ax = axes.flatten()[j]
        ori_h = ori_raw[:, :, j].mean(axis=0)
        gen_h = gen_raw[:, :, j].mean(axis=0)
        ori_std = ori_raw[:, :, j].std(axis=0)
        gen_std = gen_raw[:, :, j].std(axis=0)
        
        ax.plot(hours, ori_h, 'b-', linewidth=2, label='Original Mean')
        ax.fill_between(hours, ori_h - ori_std, ori_h + ori_std, alpha=0.15, color='blue')
        ax.plot(hours, gen_h, 'r--', linewidth=2, label='Generated Mean')
        ax.fill_between(hours, gen_h - gen_std, gen_h + gen_std, alpha=0.15, color='red')
        
        m = best['metrics'][SRC_COLS[j]]
        ax.set_title(f"{SRC_COLS[j]} — R²={m['R2']:.4f}, MAPE={m['MAPE']:.1f}%", fontsize=10)
        ax.set_ylabel(f'{SRC_COLS[j]} ({VAR_UNITS[j]})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes.flatten()[5].axis('off')
    axes[-1, 0].set_xlabel('Hour of Day')
    plt.suptitle(f'Hourly Mean ± Std: Original vs Generated (Epoch={best_ep})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "source_comparison.png"), dpi=150)
    plt.close()
    
    # 图2: Checkpoint对比曲线
    eps = sorted(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for j, (name, label) in enumerate([('PV_production', 'PV'), ('Wind_production', 'Wind'), ('score', 'Overall')]):
        ax = axes[j]
        if name == 'score':
            vals = [all_results[e]['score'] for e in eps]
            ax.set_ylabel('Score (lower=better)')
        else:
            vals = [all_results[e]['metrics'][name]['R2'] for e in eps]
            ax.set_ylabel('R²')
            if name in ref:
                ax.axhline(ref[name]['R2'], color='green', linestyle=':', label=f"Paper ref={ref[name]['R2']}")
        ax.plot(eps, vals, 'bo-', markersize=5)
        ax.axvline(best_ep, color='r', linestyle='--', label=f'Best={best_ep}')
        ax.set_xlabel('Epoch')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Checkpoint Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "checkpoint_comparison.png"), dpi=150)
    plt.close()
    
    print(f"\n完成! 结果保存在 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
