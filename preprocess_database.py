"""
preprocess_database.py — 数据预处理
====================================
极端天气识别方法参照:
  陈浩等, "考虑极端天气的先验知识引导风/光短期出力场景生成方法研究", 智慧电力, 2025

原文方法:
  "根据中央气象台提供的低温和大风标准，选取温度阈值、温度下降速率和风速阈值
   3个气象指标，构建了极端天气判定标准（表1）。
   若任意一项气象指标满足条件，即视为极端天气场景，
   并将对应的功率序列纳入极端天气数据集。"

  表1原始标准:
    (1) ΔT_24h: 24h内气温降速 ≥ 10°C  或 ΔT_48h ≥ 12°C
    (2) T_min:  1日内最低温 < 4°C
    (3) V_avg:  10min平均风速 ≥ 14 m/s

  判定逻辑: 任意一项满足 → 极端天气 (二分类: 0=正常, 1=极端)

本数据集适配:
  数据来自气候温和地区 (风速max=8.5m/s, 24h温降max=8.7°C)
  在保持论文"三指标+任一满足"框架不变的前提下, 根据本地区气候特征标定阈值
  同时补充高温极端指标 (参照苗谊凡等, 华北电力大学学报, 2025)

用法: python helper/preprocess_database.py
"""

import os
import numpy as np
import pandas as pd

DATABASE_PATH = "data/Database.csv"
OUTPUT_DATA_DIR = "data/raw"
OUTPUT_LABEL_DIR = "data/raw_labels"


def load_and_resample(csv_path):
    print(f"[1/5] 加载数据: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    df['Time'] = pd.to_datetime(df['Time'].str.replace('-T', ' '))
    df.set_index('Time', inplace=True)
    print(f"  原始: {len(df)} 条 (5min), {df.index[0]} ~ {df.index[-1]}")
    df_h = df.resample("1h").mean().dropna()
    print(f"  聚合: {len(df_h)} 条 (1h)")
    return df_h


def create_source_data(df_h):
    print("[2/5] 创建源端数据 (5变量)...")
    cols = ['Wind_speed', 'GHI', 'Temperature', 'PV_production', 'Wind_production']
    source = df_h[cols].copy()
    print(f"  形状: {source.shape}")
    for c in cols:
        print(f"    {c}: [{source[c].min():.1f}, {source[c].max():.1f}], 均值={source[c].mean():.1f}")
    return source


def create_load_data(df_h):
    print("[3/5] 创建荷端数据 (1变量)...")
    load = df_h[['Electric_demand']].copy()
    print(f"  形状: {load.shape}")
    return load


def identify_extreme_weather(df_h):
    """
    极端天气识别 — 严格遵循陈浩论文框架
    
    判定逻辑: "若任意一项气象指标满足条件, 即视为极端天气场景"
    → 二分类: 0 = 正常天气, 1 = 极端天气
    
    ==========================================================
    判别指标             陈浩论文原始阈值      本数据集标定阈值
    ----------------------------------------------------------
    (1) 24h降温速率      ΔT_24h ≤ -10°C      ΔT_24h ≤ -7°C
    (2) 48h降温速率      ΔT_48h ≤ -12°C      ΔT_48h ≤ -10°C
    (3) 最低温度         T_min < 4°C          T < 4°C (不变)
    (4) 风速             V_avg ≥ 14 m/s       V_1h ≥ 6 m/s
    (5) 高温 (补充)      —                    T > 35°C
    ==========================================================
    """
    print("[4/5] 极端天气识别 (陈浩论文: 任一满足即为极端)...")
    
    n = len(df_h)
    temp = df_h['Temperature'].values
    ws = df_h['Wind_speed'].values
    
    # 计算降温速率
    temp_s = df_h['Temperature']
    delta_24h = (temp_s - temp_s.shift(24)).values
    delta_48h = (temp_s - temp_s.shift(48)).values
    delta_24h = np.nan_to_num(delta_24h, nan=0.0)
    delta_48h = np.nan_to_num(delta_48h, nan=0.0)
    
    # 阈值 (论文框架 + 本地标定)
    TH_DELTA_24H = -7.0     # 论文: -10°C
    TH_DELTA_48H = -10.0    # 论文: -12°C
    TH_TEMP_LOW  = 4.0      # 论文: 4°C (不变)
    TH_WIND_HIGH = 6.0      # 论文: 14m/s (标定)
    TH_TEMP_HIGH = 35.0     # 补充: 高温
    
    print(f"  阈值设定 (论文框架 + 本地标定):")
    print(f"    ΔT_24h ≤ {TH_DELTA_24H}°C   [论文: -10°C]")
    print(f"    ΔT_48h ≤ {TH_DELTA_48H}°C   [论文: -12°C]")
    print(f"    T_min  <  {TH_TEMP_LOW}°C    [论文: 4°C, 不变]")
    print(f"    V_avg  ≥  {TH_WIND_HIGH} m/s [论文: 14m/s]")
    print(f"    T_max  >  {TH_TEMP_HIGH}°C   [补充: 高温极端]")
    
    # "若任意一项满足, 即视为极端天气场景"
    is_extreme = np.zeros(n, dtype=bool)
    
    cond_24h = delta_24h <= TH_DELTA_24H
    cond_48h = delta_48h <= TH_DELTA_48H
    cond_low = temp < TH_TEMP_LOW
    cond_wind = ws >= TH_WIND_HIGH
    cond_high = temp > TH_TEMP_HIGH
    
    is_extreme = cond_24h | cond_48h | cond_low | cond_wind | cond_high
    
    src_labels = is_extreme.astype(int)  # 0=正常, 1=极端
    
    # 统计
    n_extreme = is_extreme.sum()
    n_normal = n - n_extreme
    print(f"\n  各条件触发:")
    print(f"    24h降温: {cond_24h.sum()} 条 ({cond_24h.sum()/n*100:.2f}%)")
    print(f"    48h降温: {cond_48h.sum()} 条 ({cond_48h.sum()/n*100:.2f}%)")
    print(f"    低温:    {cond_low.sum()} 条 ({cond_low.sum()/n*100:.2f}%)")
    print(f"    大风:    {cond_wind.sum()} 条 ({cond_wind.sum()/n*100:.2f}%)")
    print(f"    高温:    {cond_high.sum()} 条 ({cond_high.sum()/n*100:.2f}%)")
    print(f"\n  源端标签 (二分类):")
    print(f"    0(正常天气): {n_normal} 条 ({n_normal/n*100:.1f}%)")
    print(f"    1(极端天气): {n_extreme} 条 ({n_extreme/n*100:.1f}%)")
    
    # 荷端标签: 负荷等级 (四分位)
    load = df_h['Electric_demand'].values
    load_labels = np.zeros(n, dtype=int)
    p25, p50, p75 = np.percentile(load, [25, 50, 75])
    load_labels[load > p25] = 1
    load_labels[load > p50] = 2
    load_labels[load > p75] = 3
    
    print(f"\n  荷端标签 (四分位):")
    for lv, desc in enumerate(['低谷', '平时', '高峰', '尖峰']):
        c = np.sum(load_labels == lv)
        print(f"    {lv}({desc}): {c} 条 ({c/n*100:.1f}%)")
    
    src_df = pd.DataFrame({'weather_label': src_labels}, index=df_h.index)
    load_df = pd.DataFrame({'load_label': load_labels}, index=df_h.index)
    return src_df, load_df


def save_all(source, load, src_labels, load_labels):
    print("[5/5] 保存文件...")
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    for df, path in [
        (source,      os.path.join(OUTPUT_DATA_DIR,  "source_data.csv")),
        (load,        os.path.join(OUTPUT_DATA_DIR,  "load_data.csv")),
        (src_labels,  os.path.join(OUTPUT_LABEL_DIR, "source_labels.csv")),
        (load_labels, os.path.join(OUTPUT_LABEL_DIR, "load_labels.csv")),
    ]:
        df.to_csv(path, index=False)
        print(f"  {path} ({df.shape})")


def main():
    print("=" * 60)
    print("  数据预处理 (极端天气识别: 陈浩论文框架)")
    print("  判定逻辑: 任意一项满足 → 极端天气 (二分类)")
    print("=" * 60)
    df_h = load_and_resample(DATABASE_PATH)
    source = create_source_data(df_h)
    load = create_load_data(df_h)
    src_labels, load_labels = identify_extreme_weather(df_h)
    save_all(source, load, src_labels, load_labels)
    print("\n完成! 下一步:")
    print("  源端: python run_source_scenario.py")
    print("  荷端: python run_load_scenario.py")


if __name__ == '__main__':
    main()
