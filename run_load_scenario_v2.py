"""
run_load_scenario_v2.py — 荷端场景生成 (改进版)
=================================================
改进点:
  1. 按天切割 (stride=24), 与源端一致
  2. 带checkpoint自动评估, 选最优epoch
  3. 标签使用负荷等级 (四分位)

用法: python run_load_scenario_v2.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper.mctimegan import MCTimeGAN

# ===================== 参数配置 =====================
DATABASE_PATH  = "data/Database.csv"
HORIZON        = 24
HIDDEN_DIM     = 48          # 荷端1变量, 24够了
NUM_LAYERS     = 3
EPOCHS         = 60000        # 荷端1变量训练快, 可以多跑
BATCH_SIZE     = 128
LR             = 5e-4
LR_D           = 2e-4
D_THRESHOLD    = 0.15
USE_LR_SCHED   = True
CKPT_INTERVAL  = 200
OUTPUT_DIR     = "helper/synthetic_data/load data"
CKPT_DIR       = "helper/checkpoints_load"
# ====================================================


def load_day_based_load_data(db_path):
    """按天切割荷端数据, 与源端方式一致"""
    print("[数据加载] 按天切割荷端数据...")
    
    df = pd.read_csv(db_path, index_col=0)
    df['Time'] = pd.to_datetime(df['Time'].str.replace('-T', ' '))
    df.set_index('Time', inplace=True)
    df_h = df.resample("1h").mean().dropna()
    
    load_col = 'Electric_demand'
    df_h['date'] = df_h.index.date
    
    load_days = []
    dates = sorted(df_h['date'].unique())
    for date in dates:
        day_data = df_h[df_h['date'] == date][[load_col]].values
        if len(day_data) == 24:
            load_days.append(day_data)
    
    load_data = np.array(load_days, dtype=np.float32)  # (N_days, 24, 1)
    
    # 生成负荷等级标签 (四分位)
    daily_means = load_data.mean(axis=1).flatten()  # 每天的均值
    labels_arr = np.zeros(len(load_data), dtype=int)
    p25, p50, p75 = np.percentile(daily_means, [25, 50, 75])
    labels_arr[daily_means > p25] = 1
    labels_arr[daily_means > p50] = 2
    labels_arr[daily_means > p75] = 3
    
    # 扩展为 (N, 24, 1)
    label_data = np.zeros((len(load_data), 24, 1), dtype=np.float32)
    for i in range(len(load_data)):
        label_data[i, :, 0] = labels_arr[i]
    
    print(f"  完整天数: {len(load_data)}")
    print(f"  形状: data={load_data.shape}, labels={label_data.shape}")
    print(f"  负荷范围: [{load_data.min():.0f}, {load_data.max():.0f}] kW")
    
    desc = ['低谷', '平时', '高峰', '尖峰']
    for lv in range(4):
        cnt = (labels_arr == lv).sum()
        print(f"  标签{lv}({desc[lv]}): {cnt}天 ({cnt/len(labels_arr)*100:.1f}%)")
    
    return load_data, label_data


def evaluate_load(ori_raw, gen_raw):
    """评估荷端生成质量"""
    ori_d = ori_raw[:, :, 0].mean(axis=1)
    gen_d = gen_raw[:, :, 0].mean(axis=1)
    o_s, g_s = np.sort(ori_d), np.sort(gen_d)
    
    mae = mean_absolute_error(o_s, g_s)
    rmse = np.sqrt(mean_squared_error(o_s, g_s))
    r2 = r2_score(o_s, g_s)
    
    th = max(np.percentile(np.abs(o_s), 10), 1.0)
    mask = np.abs(o_s) > th
    mape = np.mean(np.abs((o_s[mask] - g_s[mask]) / o_s[mask])) * 100 if mask.sum() > 0 else 0
    
    bins = 100
    rng = (min(o_s.min(), g_s.min()), max(o_s.max(), g_s.max()))
    h_o, _ = np.histogram(o_s, bins=bins, range=rng, density=True)
    h_g, _ = np.histogram(g_s, bins=bins, range=rng, density=True)
    h_o, h_g = h_o + 1e-10, h_g + 1e-10
    h_o, h_g = h_o / h_o.sum(), h_g / h_g.sum()
    m = 0.5 * (h_o + h_g)
    js = 0.5 * entropy(h_o, m) + 0.5 * entropy(h_g, m)
    
    fid = (ori_d.mean() - gen_d.mean())**2 + ori_d.var() + gen_d.var() - 2*np.sqrt(max(ori_d.var()*gen_d.var(), 0))
    
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2, 'JS': js, 'FID': fid}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(CKPT_DIR):
        import shutil; shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {dev}")
    
    # ===== Step 1: 加载数据 =====
    print("\n" + "=" * 60)
    print("Step 1: 按天加载荷端数据")
    print("=" * 60)
    load_raw, labels = load_day_based_load_data(DATABASE_PATH)
    
    # 归一化
    N, T, D = load_raw.shape
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(load_raw.reshape(-1, D)).reshape(N, T, D).astype(np.float32)
    
    # 打乱
    perm = np.random.RandomState(42).permutation(N)
    data_norm = data_norm[perm]
    labels = labels[perm]
    load_raw_shuffled = load_raw[perm]
    
    print(f"  归一化后: {data_norm.shape}")
    
    # ===== Step 2: 训练 =====
    print("\n" + "=" * 60)
    print("Step 2: MC-TimeGAN 训练 (荷端, 无Vine-Copula)")
    print("=" * 60)
    
    model = MCTimeGAN(
        input_features=D,                         # 1
        input_conditions=labels.shape[-1],        # 1
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, N),
        learning_rate=LR,
        lr_d=LR_D,
        d_threshold=D_THRESHOLD,
        use_lr_scheduler=USE_LR_SCHED,
        w_corr=0,                                 # 荷端1变量不需要相关性损失
        vine_model=None,
    ).to(dev)
    
    model.checkpoint_dir = CKPT_DIR
    model.checkpoint_interval = CKPT_INTERVAL
    
    model.fit(data_norm, cond_labels=labels)
    
    # ===== Step 3: 评估所有checkpoint =====
    print("\n" + "=" * 60)
    print("Step 3: 评估所有checkpoint")
    print("=" * 60)
    
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pth")))
    print(f"  找到 {len(ckpt_files)} 个checkpoint")
    
    all_results = {}
    for ckpt_path in ckpt_files:
        ep = int(os.path.basename(ckpt_path).replace("epoch_", "").replace(".pth", ""))
        model.load_checkpoint(ckpt_path)
        
        gen_norm = model.transform(data_norm.shape, cond=labels)
        gen_raw = scaler.inverse_transform(gen_norm.reshape(-1, D)).reshape(N, T, D)
        
        metrics = evaluate_load(load_raw_shuffled, gen_raw)
        all_results[ep] = {'metrics': metrics, 'gen_raw': gen_raw}
        
        print(f"  Epoch {ep:>6d}: MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}, "
              f"MAE={metrics['MAE']:.1f}, JS={metrics['JS']:.4f}")
    
    # ===== Step 4: 最优epoch =====
    best_ep = max(all_results, key=lambda e: all_results[e]['metrics']['R2'])
    best = all_results[best_ep]
    m = best['metrics']
    
    print(f"\n{'='*60}")
    print(f"  最优epoch: {best_ep}")
    print(f"  MAE={m['MAE']:.2f}, MAPE={m['MAPE']:.2f}%, RMSE={m['RMSE']:.2f}, "
          f"R²={m['R2']:.4f}, JS={m['JS']:.4f}")
    print(f"{'='*60}")
    
    # 保存最优结果
    np.save(os.path.join(OUTPUT_DIR, "load_generated.npy"), best['gen_raw'])
    np.save(os.path.join(OUTPUT_DIR, "load_original.npy"), load_raw_shuffled)
    np.save(os.path.join(OUTPUT_DIR, "load_labels.npy"), labels)
    
    # 论文对比
    print(f"\n  与参考论文对比:")
    print(f"    {'指标':10s} {'本文':>10s} {'论文Proposed':>15s}")
    print(f"    {'R²':10s} {m['R2']:>10.4f} {'0.9884':>15s}")
    print(f"    {'MAPE':10s} {m['MAPE']:>9.2f}% {'2.45%':>15s}")
    print(f"    {'MAE':10s} {m['MAE']:>10.2f} {'25.67':>15s}")
    print(f"    {'JS':10s} {m['JS']:>10.4f} {'0.0209':>15s}")
    
    # ===== 可视化 =====
    gen_raw = best['gen_raw']
    ori_raw = load_raw_shuffled
    hours = np.arange(24)
    
    # 图1: 逐小时均值±标准差
    fig, ax = plt.subplots(figsize=(14, 6))
    ori_h = ori_raw[:, :, 0].mean(axis=0)
    gen_h = gen_raw[:, :, 0].mean(axis=0)
    ori_std = ori_raw[:, :, 0].std(axis=0)
    gen_std = gen_raw[:, :, 0].std(axis=0)
    
    ax.plot(hours, ori_h, 'b-', linewidth=2, label='Original Mean')
    ax.fill_between(hours, ori_h - ori_std, ori_h + ori_std, alpha=0.15, color='blue')
    ax.plot(hours, gen_h, 'r--', linewidth=2, label='Generated Mean')
    ax.fill_between(hours, gen_h - gen_std, gen_h + gen_std, alpha=0.15, color='red')
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Electric Demand (kW)', fontsize=12)
    ax.set_title(f'Load Scenarios: Original vs Generated (Epoch={best_ep}, R²={m["R2"]:.4f})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "load_comparison.png"), dpi=150)
    plt.close()
    
    # 图2: Checkpoint R²曲线
    eps = sorted(all_results.keys())
    r2s = [all_results[e]['metrics']['R2'] for e in eps]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, r2s, 'bo-', markersize=4)
    ax.axvline(best_ep, color='r', linestyle='--', label=f'Best={best_ep}')
    ax.axhline(0.9884, color='green', linestyle=':', label='Paper ref=0.9884')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Load: R² by Checkpoint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "load_checkpoint.png"), dpi=150)
    plt.close()
    
    print(f"\n荷端场景生成完成! 结果保存在 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
