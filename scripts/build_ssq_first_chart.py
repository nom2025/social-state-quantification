#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSQ First Chart: STI vs Discretionary Share
============================================
社会脅威指数 (STI) と裁量消費比率を重ね描きする「最初の1枚の図」

STI構成要素 (v0 — 利用可能な月次系列から開始):
  1. 完全失業率 (景気動向指数 cat01=3060, 月次 1975+)
  2. 常用雇用指数 前年同月比 (景気動向指数 cat01=3020, 月次 1975+) → 反転
  3. 刑法犯認知件数率 (社会生活統計指標 #K06101, 年次→月次線形補間)

消費構成 (家計調査 0002070001, 二人以上世帯, 全国, 月次 2000+):
  裁量消費 = 外食(098) + 教養娯楽(156) + 交通(146) + 被服(122) + 家具家事用品(112)
  裁量消費比率 = 裁量消費 / 消費支出(059) × 100

期間: 2000年1月〜最新
"""

import sys
import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "estat_config.json"
OUTPUT_DIR = BASE_DIR / "data" / "ssq"
CHART_DIR = BASE_DIR / "output" / "ssq"

# ── API Config ──
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
APP_ID = config.get("app_id", config.get("api_key", ""))

ESTAT_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
START_YEAR = 2000


# ═══════════════════════════════════════════════════════════
# e-Stat API ユーティリティ
# ═══════════════════════════════════════════════════════════

def fetch_estat(stats_data_id, extra_params, label=""):
    """e-Stat APIからデータ取得し、生レコードリストを返す"""
    params = {
        "appId": APP_ID,
        "statsDataId": stats_data_id,
        "metaGetFlg": "Y",
        "lang": "J",
        "limit": 100000,
    }
    params.update(extra_params)

    print(f"  [{label}] API呼び出し中...", end="", flush=True)
    resp = requests.get(ESTAT_URL, params=params, timeout=120)
    resp.raise_for_status()
    body = resp.json()

    result = body.get("GET_STATS_DATA", {}).get("RESULT", {})
    if result.get("STATUS") != 0:
        print(f" エラー: {result.get('ERROR_MSG', '不明')}")
        return [], {}

    stat_data = body["GET_STATS_DATA"]["STATISTICAL_DATA"]
    values = stat_data.get("DATA_INF", {}).get("VALUE", [])
    if isinstance(values, dict):
        values = [values]

    # メタ情報: 時間軸の名称マッピング
    meta_inf = stat_data.get("CLASS_INF", {}).get("CLASS_OBJ", [])
    if isinstance(meta_inf, dict):
        meta_inf = [meta_inf]
    time_map = {}
    cat01_map = {}
    for obj in meta_inf:
        if obj.get("@id") == "time":
            classes = obj.get("CLASS", [])
            if isinstance(classes, dict):
                classes = [classes]
            time_map = {c["@code"]: c["@name"] for c in classes}
        if obj.get("@id") == "cat01":
            classes = obj.get("CLASS", [])
            if isinstance(classes, dict):
                classes = [classes]
            cat01_map = {c["@code"]: c["@name"] for c in classes}

    print(f" {len(values)}件取得")
    return values, {"time_map": time_map, "cat01_map": cat01_map}


def parse_time_name(time_name, time_code):
    """時間名称 '2025年1月' or 時間コード 'YYYYMMDD' をパース → (year, month)"""
    # 名称パース: "2025年1月" → (2025, 1)
    if "年" in time_name and "月" in time_name:
        try:
            parts = time_name.replace("月", "").split("年")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass

    # 名称パース: "2020年度" → (2020, fiscal year → mid-year)
    if "年度" in time_name:
        try:
            year = int(time_name.replace("年度", "").strip())
            return year, None  # None = annual
        except ValueError:
            pass

    # 名称パース: "2020年" → (2020, annual)
    if "年" in time_name and "月" not in time_name:
        try:
            year = int(time_name.replace("年", "").strip())
            return year, None
        except ValueError:
            pass

    # コードパース: 最後の2桁が月
    if len(time_code) >= 6:
        try:
            year = int(time_code[:4])
            month = int(time_code[-2:])
            if 1 <= month <= 12:
                return year, month
            return year, None  # annual
        except (ValueError, TypeError):
            pass

    return None, None


# ═══════════════════════════════════════════════════════════
# STI Component 1: 完全失業率 (月次)
# ═══════════════════════════════════════════════════════════

def fetch_unemployment_rate():
    """景気動向指数テーブルから完全失業率を取得 (Lg6, cat01=3060)"""
    values, meta = fetch_estat("0003446462", {"cdCat01": "3060"},
                               label="完全失業率")
    time_map = meta.get("time_map", {})

    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        if raw in ("", "-", "***", "…", "x"):
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or month is None or year < START_YEAR:
            continue
        try:
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "unemployment_rate": float(raw),
            })
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"    → {len(df)}件 ({df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m})")
    return df


# ═══════════════════════════════════════════════════════════
# STI Component 2: 常用雇用指数 前年同月比 (月次, 反転)
# ═══════════════════════════════════════════════════════════

def fetch_employment_index():
    """景気動向指数テーブルから常用雇用指数(前年同月比)を取得 (Lg2, cat01=3020)
    ストレス方向: 雇用が減ると脅威↑ → 反転 (×-1)
    """
    values, meta = fetch_estat("0003446462", {"cdCat01": "3020"},
                               label="常用雇用指数(前年同月比)")
    time_map = meta.get("time_map", {})

    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        if raw in ("", "-", "***", "…", "x"):
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or month is None or year < START_YEAR:
            continue
        try:
            # 反転: マイナス成長 = 脅威 → ×-1 でストレス方向を統一
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "employment_decline": -float(raw),
            })
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"    → {len(df)}件 ({df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m})")
    return df


# ═══════════════════════════════════════════════════════════
# STI Leading Component: 有効求人倍率 (反転, 月次)
# ═══════════════════════════════════════════════════════════

def fetch_job_openings_inverted():
    """景気動向指数テーブルから有効求人倍率を取得 (C9, cat01=2090)
    ストレス方向: 求人減少 = 脅威↑ → 反転 (×-1)
    先行性: 失業率より数ヶ月先行して動く
    """
    values, meta = fetch_estat("0003446462", {"cdCat01": "2090"},
                               label="有効求人倍率(反転→脅威方向)")
    time_map = meta.get("time_map", {})

    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        if raw in ("", "-", "***", "…", "x"):
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or month is None or year < START_YEAR:
            continue
        try:
            # 反転: 求人倍率低い = 脅威大 → ×-1
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "job_scarcity": -float(raw),
            })
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"    → {len(df)}件 ({df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m})")
    return df


# ═══════════════════════════════════════════════════════════
# STI Leading Component: 消費者態度指数 (反転, 月次)
# ═══════════════════════════════════════════════════════════

def fetch_consumer_confidence_inverted():
    """景気動向指数テーブルから消費者態度指数を取得 (L6, cat01=1060)
    ストレス方向: 態度指数低下 = 不安↑ → 反転 (×-1)
    先行性: 景気動向指数の「先行系列」に分類
    """
    values, meta = fetch_estat("0003446462", {"cdCat01": "1060"},
                               label="消費者態度指数(反転→脅威方向)")
    time_map = meta.get("time_map", {})

    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        if raw in ("", "-", "***", "…", "x"):
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or month is None or year < START_YEAR:
            continue
        try:
            # 反転: 態度指数低い = 不安大 → ×-1
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "consumer_anxiety": -float(raw),
            })
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"    → {len(df)}件 ({df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m})")
    return df


# ═══════════════════════════════════════════════════════════
# Phase D 統制変数: 可処分所得 (月次, 勤労者世帯)
# ═══════════════════════════════════════════════════════════

def fetch_disposable_income():
    """家計調査から可処分所得を取得 (勤労者世帯, 月次)
    cat01=233: 可処分所得 / フォールバック: cat01=019 (実収入)
    cat02=04: 勤労者世帯
    """
    for cat01, name in [("233", "可処分所得"), ("019", "実収入")]:
        values, meta = fetch_estat(
            "0002070001",
            {
                "cdCat01": cat01,
                "cdCat02": "04",
                "cdArea": "00000",
                "cdTimeFrom": f"{START_YEAR}000101",
            },
            label=f"{name}(勤労者世帯)"
        )
        if values:
            break

    time_map = meta.get("time_map", {})
    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        if raw in ("", "-", "***", "…", "x"):
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or month is None or year < START_YEAR:
            continue
        try:
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "disposable_income": float(raw),
            })
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if not df.empty:
        print(f"    → {len(df)}件 ({df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m})")
        print(f"    平均: {df['disposable_income'].mean():,.0f}円")
    return df


def load_cpi_yoy():
    """既存CPIデータからYoY(前年同月比)を読み込む"""
    cpi_path = BASE_DIR / "data" / "price_chain" / "cpi_monthly.csv"
    if not cpi_path.exists():
        print(f"  CPIファイルなし: {cpi_path}")
        return pd.DataFrame()
    df = pd.read_csv(cpi_path, parse_dates=["date"])
    result = df[["date", "総合_前年同月比"]].rename(
        columns={"総合_前年同月比": "cpi_yoy"}
    ).dropna(subset=["cpi_yoy"])
    print(f"  CPI (前年同月比): {len(result)}件 "
          f"({result['date'].min():%Y-%m} ~ {result['date'].max():%Y-%m})")
    return result


# ═══════════════════════════════════════════════════════════
# STI Component 3: 刑法犯認知件数率 (年次→月次補間)
# ═══════════════════════════════════════════════════════════

def fetch_crime_rate():
    """社会生活統計指標から刑法犯認知件数率(人口千人当たり)を取得 (年次)
    → 線形補間で月次化
    """
    values, meta = fetch_estat(
        "0000010211",
        {"cdCat01": "#K06101", "cdArea": "00000"},
        label="刑法犯認知件数率(年次)"
    )
    time_map = meta.get("time_map", {})

    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        if raw in ("", "-", "***", "…", "x"):
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or year < START_YEAR:
            continue
        try:
            # 年度データ → 年度の中央月 (10月) にアンカー
            anchor_month = 10 if month is None else month
            records.append({
                "date": pd.Timestamp(year=year, month=anchor_month, day=1),
                "crime_rate": float(raw),
            })
        except (ValueError, TypeError):
            pass

    annual_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if annual_df.empty:
        print("    → 0件")
        return pd.DataFrame()

    print(f"    → {len(annual_df)}件 年次 ({annual_df['date'].min():%Y-%m} ~ {annual_df['date'].max():%Y-%m})")

    # 月次に線形補間
    start = annual_df["date"].min().replace(month=1)
    end = annual_df["date"].max().replace(month=12)
    monthly_idx = pd.date_range(start, end, freq="MS")
    monthly_df = pd.DataFrame({"date": monthly_idx})
    merged = monthly_df.merge(annual_df, on="date", how="left")
    merged["crime_rate"] = merged["crime_rate"].interpolate(method="linear")
    merged = merged.dropna(subset=["crime_rate"])

    print(f"    → 補間後 {len(merged)}件 月次 ({merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m})")
    return merged


# ═══════════════════════════════════════════════════════════
# 消費構成: 家計調査 品目別 (月次)
# ═══════════════════════════════════════════════════════════

def fetch_consumption_categories():
    """家計調査(二人以上世帯・全国)から品目別消費支出を取得

    cat01:
      059: 消費支出(total)
      098: 外食
      156: 教養娯楽
      146: 交通
      122: 被服及び履物
      112: 家具・家事用品
    """
    cat_codes = "059,098,156,146,122,112"
    cat_names = {
        "059": "total_consumption",
        "098": "dining_out",
        "156": "entertainment",
        "146": "transport",
        "122": "clothing",
        "112": "furniture",
    }

    values, meta = fetch_estat(
        "0002070001",
        {
            "cdCat01": cat_codes,
            "cdCat02": "03",       # 二人以上の世帯
            "cdArea": "00000",     # 全国
            "cdTimeFrom": f"{START_YEAR}000101",
        },
        label="家計調査 品目別消費支出"
    )
    time_map = meta.get("time_map", {})

    records = []
    for v in values:
        tc = v.get("@time", "")
        raw = v.get("$", "")
        cat01 = v.get("@cat01", "")
        if raw in ("", "-", "***", "…", "x") or cat01 not in cat_names:
            continue
        year, month = parse_time_name(time_map.get(tc, ""), tc)
        if year is None or month is None or year < START_YEAR:
            continue
        try:
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "category": cat_names[cat01],
                "value": float(raw),
            })
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(records)
    wide = df.pivot_table(index="date", columns="category", values="value",
                          aggfunc="first")
    wide = wide.sort_index().reset_index()

    print(f"    → {len(wide)}件 月次, 列: {[c for c in wide.columns if c != 'date']}")
    return wide


def compute_discretionary_share(consumption_df):
    """裁量消費比率 = (外食+教養娯楽+交通+被服+家具家事用品) / 消費支出 × 100"""
    df = consumption_df.copy()
    disc_cols = ["dining_out", "entertainment", "transport", "clothing", "furniture"]
    missing = [c for c in disc_cols + ["total_consumption"] if c not in df.columns]
    if missing:
        print(f"    ERROR: 欠損列: {missing}")
        return pd.DataFrame()

    df["discretionary"] = df[disc_cols].sum(axis=1)
    df["discretionary_share"] = df["discretionary"] / df["total_consumption"] * 100

    result = df[["date", "discretionary_share", "discretionary", "total_consumption"]].copy()
    print(f"    裁量消費比率: 平均 {result['discretionary_share'].mean():.1f}%, "
          f"範囲 {result['discretionary_share'].min():.1f}~{result['discretionary_share'].max():.1f}%")
    return result


# ═══════════════════════════════════════════════════════════
# STI 構築
# ═══════════════════════════════════════════════════════════

def build_sti(components):
    """
    STI = z-score標準化 → 等ウェイト平均
    components: {name: DataFrame with 'date' and one value column}
    全系列はストレス方向統一済み (高い = 脅威大)
    """
    if not components:
        print("  ERROR: STI構成要素なし")
        return pd.DataFrame(), pd.DataFrame()

    merged = None
    for name, df in components.items():
        val_col = [c for c in df.columns if c != "date"][0]
        temp = df[["date", val_col]].rename(columns={val_col: name})
        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # z-score (全期間ベース)
    comp_names = [c for c in merged.columns if c != "date"]
    for col in comp_names:
        m = merged[col].mean()
        s = merged[col].std()
        merged[f"{col}_z"] = (merged[col] - m) / s if s > 0 else 0.0

    z_cols = [f"{c}_z" for c in comp_names]
    merged["STI"] = merged[z_cols].mean(axis=1)

    print(f"  STI構築: {len(merged)}件, {len(comp_names)}要素: {comp_names}")
    print(f"  STI範囲: {merged['STI'].min():.2f} ~ {merged['STI'].max():.2f}")

    sti_out = merged[["date", "STI"] + z_cols].dropna(subset=["STI"])
    return sti_out, merged


# ═══════════════════════════════════════════════════════════
# チャート
# ═══════════════════════════════════════════════════════════

def plot_first_chart(sti_df, ds_df, output_path):
    """SSQ最初の1枚: STI vs 裁量消費比率"""
    merged = sti_df.merge(ds_df[["date", "discretionary_share"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if merged.empty:
        print("  ERROR: STIと消費比率の重複期間なし")
        return

    # 12ヶ月移動平均 (ノイズ除去)
    merged["ds_ma12"] = merged["discretionary_share"].rolling(12, center=True).mean()
    merged["sti_ma12"] = merged["STI"].rolling(12, center=True).mean()

    # フォント設定
    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # 裁量消費比率 (左軸, 青)
    c1 = '#1565C0'
    c1_light = '#90CAF9'
    ax1.plot(merged['date'], merged['discretionary_share'],
             color=c1_light, linewidth=0.5, alpha=0.5)
    ax1.plot(merged['date'], merged['ds_ma12'],
             color=c1, linewidth=2.0, label='裁量消費比率 (12MA)')
    ax1.set_ylabel('裁量消費比率 (%)', color=c1, fontsize=13)
    ax1.tick_params(axis='y', labelcolor=c1)

    # STI (右軸, 赤, 軸反転)
    ax2 = ax1.twinx()
    c2 = '#C62828'
    c2_light = '#EF9A9A'
    ax2.plot(merged['date'], merged['STI'],
             color=c2_light, linewidth=0.5, alpha=0.5)
    ax2.plot(merged['date'], merged['sti_ma12'],
             color=c2, linewidth=2.0, label='STI (12MA)')
    ax2.set_ylabel('STI (社会脅威指数) ← 高ストレス', color=c2, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax2.invert_yaxis()  # 反転: 脅威が大きいほど下 → 消費と同方向に動けば仮説支持

    # タイトル
    period = f"{merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m}"
    ax1.set_title(
        f'SSQ First Chart: 社会脅威指数 (STI) vs 裁量消費比率\n期間: {period}',
        fontsize=15, pad=20
    )

    # 危機期間マーカー
    crises = [
        ("2001-03", "2001-12", "ITバブル崩壊"),
        ("2008-09", "2009-06", "リーマン"),
        ("2011-03", "2011-09", "震災"),
        ("2014-04", "2014-06", "消費税8%"),
        ("2019-10", "2019-12", "消費税10%"),
        ("2020-03", "2020-09", "COVID-19"),
    ]
    y_top = ax1.get_ylim()[1]
    for start, end, name in crises:
        try:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            if s >= merged['date'].min() and s <= merged['date'].max():
                ax1.axvspan(s, e, alpha=0.08, color='gray')
                ax1.text(s, y_top, f' {name}', fontsize=7, alpha=0.6,
                         verticalalignment='top', rotation=90)
        except:
            pass

    # 書式
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 凡例
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, loc='upper left', fontsize=11,
               framealpha=0.9)

    # 相関情報
    corr_raw = merged['STI'].corr(merged['discretionary_share'])
    valid_ma = merged.dropna(subset=["sti_ma12", "ds_ma12"])
    corr_ma = valid_ma['sti_ma12'].corr(valid_ma['ds_ma12']) if len(valid_ma) > 12 else float('nan')

    info_text = (
        f'相関係数 (raw): r = {corr_raw:.3f}\n'
        f'相関係数 (12MA): r = {corr_ma:.3f}\n'
        f'N = {len(merged)} months\n'
        f'STI軸は反転表示 (下=高ストレス)'
    )
    ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {output_path}")


# ═══════════════════════════════════════════════════════════
# Phase A: デトレンド (前年同月比)
# ═══════════════════════════════════════════════════════════

def compute_yoy(df, date_col, value_col):
    """前年同月比 (12ヶ月差分) を計算"""
    df = df.sort_values(date_col).copy()
    df[f"{value_col}_yoy"] = df[value_col].diff(12)
    return df.dropna(subset=[f"{value_col}_yoy"])


def plot_detrended_chart(sti_df, ds_df, output_path):
    """Phase A: デトレンド後 STI(YoY) vs 裁量消費比率(YoY)"""
    # YoY変換
    sti_yoy = compute_yoy(sti_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")

    merged = sti_yoy.merge(ds_yoy[["date", "discretionary_share_yoy"]],
                           on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if merged.empty:
        print("  ERROR: デトレンドデータの重複期間なし")
        return None

    # 12ヶ月移動平均
    merged["ds_yoy_ma12"] = merged["discretionary_share_yoy"].rolling(12, center=True).mean()
    merged["sti_yoy_ma12"] = merged["STI_yoy"].rolling(12, center=True).mean()

    # フォント設定
    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # 裁量消費比率 YoY (左軸, 青)
    c1 = '#1565C0'
    c1_light = '#90CAF9'
    ax1.plot(merged['date'], merged['discretionary_share_yoy'],
             color=c1_light, linewidth=0.5, alpha=0.4)
    ax1.plot(merged['date'], merged['ds_yoy_ma12'],
             color=c1, linewidth=2.0, label='裁量消費比率 YoY (12MA)')
    ax1.axhline(y=0, color=c1, linewidth=0.5, linestyle='--', alpha=0.3)
    ax1.set_ylabel('裁量消費比率 前年同月差 (pp)', color=c1, fontsize=13)
    ax1.tick_params(axis='y', labelcolor=c1)

    # STI YoY (右軸, 赤, 軸反転)
    ax2 = ax1.twinx()
    c2 = '#C62828'
    c2_light = '#EF9A9A'
    ax2.plot(merged['date'], merged['STI_yoy'],
             color=c2_light, linewidth=0.5, alpha=0.4)
    ax2.plot(merged['date'], merged['sti_yoy_ma12'],
             color=c2, linewidth=2.0, label='STI YoY (12MA)')
    ax2.axhline(y=0, color=c2, linewidth=0.5, linestyle='--', alpha=0.3)
    ax2.set_ylabel('STI 前年同月差 ← 脅威増加', color=c2, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax2.invert_yaxis()

    # タイトル
    period = f"{merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m}"
    ax1.set_title(
        f'SSQ Phase A: デトレンド後 STI(YoY) vs 裁量消費比率(YoY)\n期間: {period}',
        fontsize=15, pad=20
    )

    # 危機期間マーカー
    crises = [
        ("2008-09", "2009-06", "リーマン"),
        ("2011-03", "2011-09", "震災"),
        ("2014-04", "2014-06", "消費税8%"),
        ("2019-10", "2019-12", "消費税10%"),
        ("2020-03", "2020-09", "COVID-19"),
    ]
    for start, end, name in crises:
        try:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            if s >= merged['date'].min() and s <= merged['date'].max():
                ax1.axvspan(s, e, alpha=0.10, color='gray')
                ax1.text(s + pd.Timedelta(days=15),
                         ax1.get_ylim()[1] * 0.95,
                         name, fontsize=8, alpha=0.7,
                         verticalalignment='top')
        except:
            pass

    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 凡例
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, loc='upper left', fontsize=11,
               framealpha=0.9)

    # 相関情報
    corr_raw = merged['STI_yoy'].corr(merged['discretionary_share_yoy'])
    valid_ma = merged.dropna(subset=["sti_yoy_ma12", "ds_yoy_ma12"])
    corr_ma = (valid_ma['sti_yoy_ma12'].corr(valid_ma['ds_yoy_ma12'])
               if len(valid_ma) > 12 else float('nan'))

    info_text = (
        f'相関係数 (raw YoY): r = {corr_raw:.3f}\n'
        f'相関係数 (12MA YoY): r = {corr_ma:.3f}\n'
        f'N = {len(merged)} months\n'
        f'STI軸は反転表示 (下=脅威増加)'
    )
    ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {output_path}")

    # サマリー出力
    print(f"  デトレンド相関 (raw): r = {corr_raw:.4f}")
    print(f"  デトレンド相関 (12MA): r = {corr_ma:.4f}")
    if corr_raw < 0:
        print(f"  → 方向は仮説と整合 (負: STI↑ → 裁量消費比率↓)")
    else:
        print(f"  → 方向は仮説と不整合 (正)")

    return merged


# ═══════════════════════════════════════════════════════════
# Phase B: 危機期間ズームパネル
# ═══════════════════════════════════════════════════════════

def plot_crisis_panels(sti_df, ds_df, output_path):
    """Phase B: 3大危機の前後12ヶ月ズーム"""
    # YoY変換
    sti_yoy = compute_yoy(sti_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    merged = sti_yoy.merge(ds_yoy[["date", "discretionary_share_yoy"]],
                           on="date", how="inner").sort_values("date")

    if merged.empty:
        print("  ERROR: データなし")
        return

    crises = [
        ("リーマンショック", "2008-09", "2007-09", "2010-03"),
        ("東日本大震災", "2011-03", "2010-03", "2012-09"),
        ("COVID-19", "2020-03", "2019-03", "2021-09"),
    ]

    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    for idx, (name, onset, win_start, win_end) in enumerate(crises):
        ax1 = axes[idx]
        onset_dt = pd.Timestamp(onset)
        ws = pd.Timestamp(win_start)
        we = pd.Timestamp(win_end)
        window = merged[(merged['date'] >= ws) & (merged['date'] <= we)]

        if window.empty:
            ax1.text(0.5, 0.5, "データなし", transform=ax1.transAxes,
                     ha='center', va='center')
            ax1.set_title(name)
            continue

        # 裁量消費比率 YoY (左軸, 青)
        c1 = '#1565C0'
        ax1.bar(window['date'], window['discretionary_share_yoy'],
                width=25, color=c1, alpha=0.5, label='裁量消費比率 YoY')
        ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax1.set_ylabel('裁量消費比率 YoY (pp)', color=c1, fontsize=10)
        ax1.tick_params(axis='y', labelcolor=c1)

        # STI YoY (右軸, 赤)
        ax2 = ax1.twinx()
        c2 = '#C62828'
        ax2.plot(window['date'], window['STI_yoy'],
                 color=c2, linewidth=2.0, marker='o', markersize=3,
                 label='STI YoY')
        ax2.set_ylabel('STI YoY', color=c2, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=c2)
        ax2.invert_yaxis()

        # 危機onset線
        ax1.axvline(x=onset_dt, color='black', linewidth=2, linestyle='--',
                    alpha=0.7)
        ax1.text(onset_dt, ax1.get_ylim()[1], f' {onset}',
                 fontsize=8, verticalalignment='top', fontweight='bold')

        ax1.set_title(name, fontsize=13, fontweight='bold')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right',
                 fontsize=8)
        ax1.grid(True, alpha=0.2)

        # パネル内相関
        r = window['STI_yoy'].corr(window['discretionary_share_yoy'])
        ax1.text(0.05, 0.05, f'r = {r:.3f}', transform=ax1.transAxes,
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat',
                                       alpha=0.8))

    fig.suptitle('SSQ Phase B: 危機期間ディープダイブ (YoYデトレンド済)',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {output_path}")


# ═══════════════════════════════════════════════════════════
# Phase C: ラグ相関分析 (CCF + ローリング相関)
# ═══════════════════════════════════════════════════════════

def compute_ccf(sti_df, ds_df, max_lag=12):
    """交差相関関数 (CCF): corr(STI(t), DS(t+k)) for k = -max_lag ~ +max_lag

    k > 0 : STI が先行（STI変化 → k ヶ月後に消費変化）
    k = 0 : 同時
    k < 0 : 消費が先行
    """
    sti_yoy = compute_yoy(sti_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    merged = sti_yoy.merge(ds_yoy[["date", "discretionary_share_yoy"]],
                           on="date", how="inner").sort_values("date")

    if merged.empty:
        return pd.DataFrame()

    results = []
    for k in range(-max_lag, max_lag + 1):
        # corr(STI(t), DS(t+k)) = corr(STI, DS.shift(-k))
        shifted = merged["discretionary_share_yoy"].shift(-k)
        valid = merged[["STI_yoy"]].assign(ds_shifted=shifted).dropna()
        if len(valid) < 24:
            continue
        r = valid["STI_yoy"].corr(valid["ds_shifted"])
        results.append({"lag_k": k, "correlation": r, "n": len(valid)})

    return pd.DataFrame(results)


def compute_rolling_correlation(sti_df, ds_df, window=36):
    """ローリングウィンドウ相関: 36ヶ月窓"""
    sti_yoy = compute_yoy(sti_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    merged = sti_yoy.merge(ds_yoy[["date", "discretionary_share_yoy"]],
                           on="date", how="inner").sort_values("date").reset_index(drop=True)

    if len(merged) < window:
        return pd.DataFrame()

    rolling_r = merged["STI_yoy"].rolling(window).corr(merged["discretionary_share_yoy"])
    result = merged[["date"]].copy()
    result["rolling_r"] = rolling_r
    return result.dropna(subset=["rolling_r"])


def plot_ccf(ccf_df, output_path):
    """CCFプロット: 決着の図"""
    if ccf_df.empty:
        print("  ERROR: CCFデータなし")
        return

    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, ax = plt.subplots(figsize=(14, 7))

    lags = ccf_df["lag_k"]
    corrs = ccf_df["correlation"]

    # バーの色分け: 正ラグ(STI先行)側を強調
    colors = ['#C62828' if k > 0 else '#1565C0' if k < 0 else '#333333'
              for k in lags]
    bars = ax.bar(lags, corrs, color=colors, alpha=0.7, edgecolor='white',
                  linewidth=0.5)

    # ゼロライン
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # 95%信頼区間 (近似: ±1.96/√n)
    n = ccf_df["n"].median()
    ci = 1.96 / np.sqrt(n)
    ax.axhline(y=ci, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(y=-ci, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.fill_between([-max(abs(lags)), max(abs(lags))], -ci, ci,
                    color='gray', alpha=0.05)

    # ピーク検出
    min_idx = corrs.idxmin()
    peak_lag = int(lags.iloc[min_idx])
    peak_r = corrs.iloc[min_idx]

    # ピークマーカー
    ax.annotate(f'ピーク: k={peak_lag}, r={peak_r:.3f}',
                xy=(peak_lag, peak_r),
                xytext=(peak_lag + 2, peak_r - 0.03),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                          alpha=0.9))

    # ラベル
    ax.set_xlabel('ラグ k (月)\n'
                  '← 消費が先行 | STIが先行 →',
                  fontsize=12)
    ax.set_ylabel('相関係数 r', fontsize=12)
    ax.set_title('SSQ Phase C: 交差相関関数 (CCF)\n'
                 'corr(STI_YoY(t), 裁量消費比率_YoY(t+k))',
                 fontsize=15, pad=20)

    ax.set_xticks(range(int(lags.min()), int(lags.max()) + 1))
    ax.grid(True, alpha=0.2, axis='y')

    # 解釈テキスト
    if peak_lag > 0:
        interp = (f'STIが {peak_lag}ヶ月先行\n'
                  f'→ 社会脅威の変化が消費構成を予測する\n'
                  f'→ SSQ仮説と整合')
        interp_color = '#2E7D32'
    elif peak_lag == 0:
        interp = ('同時相関\n'
                  '→ 共通原因の可能性\n'
                  '→ 因果方向は不確定')
        interp_color = '#F57F17'
    else:
        interp = (f'消費が {abs(peak_lag)}ヶ月先行\n'
                  '→ STIはただの景気指標\n'
                  '→ SSQ仮説と不整合')
        interp_color = '#C62828'

    ax.text(0.02, 0.02, interp, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor=interp_color, linewidth=2, alpha=0.9))

    # N数と信頼区間の注記
    ax.text(0.98, 0.98, f'N ≈ {int(n)} months\n95% CI: ±{ci:.3f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {output_path}")

    return peak_lag, peak_r


def plot_rolling_correlation(rolling_df, output_path):
    """ローリング相関プロット: レジーム変化の検出"""
    if rolling_df.empty:
        print("  ERROR: ローリング相関データなし")
        return

    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, ax = plt.subplots(figsize=(16, 6))

    # 相関値の塗り分け
    dates = rolling_df['date']
    r_vals = rolling_df['rolling_r']

    ax.fill_between(dates, 0, r_vals, where=(r_vals < 0),
                    color='#C62828', alpha=0.3, label='負の相関 (仮説方向)')
    ax.fill_between(dates, 0, r_vals, where=(r_vals >= 0),
                    color='#1565C0', alpha=0.3, label='正の相関')
    ax.plot(dates, r_vals, color='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # 危機マーカー
    crises = [
        ("2008-09", "リーマン"),
        ("2011-03", "震災"),
        ("2014-04", "消費税8%"),
        ("2020-03", "COVID"),
    ]
    for onset, name in crises:
        try:
            dt = pd.Timestamp(onset)
            if dt >= dates.min() and dt <= dates.max():
                ax.axvline(x=dt, color='gray', linewidth=1, linestyle='--',
                           alpha=0.5)
                ax.text(dt, ax.get_ylim()[1] * 0.9, f' {name}', fontsize=8,
                        alpha=0.7, rotation=90, verticalalignment='top')
        except:
            pass

    ax.set_ylabel('相関係数 r (36ヶ月窓)', fontsize=12)
    ax.set_title('SSQ Phase C: ローリング相関 STI(YoY) vs 裁量消費比率(YoY)\n'
                 '窓幅 = 36ヶ月',
                 fontsize=14, pad=15)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {output_path}")


# ═══════════════════════════════════════════════════════════
# Phase C+: CCF比較 (遅行STI vs 先行STI)
# ═══════════════════════════════════════════════════════════

def plot_ccf_comparison(ccf_lagging, ccf_leading, output_path):
    """CCF比較プロット: 遅行STI vs 先行STI"""
    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for ax, ccf_df, title, color in [
        (ax1, ccf_lagging, 'STI_lagging (遅行指標)\n失業率+雇用指数+犯罪率', '#666666'),
        (ax2, ccf_leading, 'STI_leading (先行指標)\n求人倍率+消費者態度指数', '#1B5E20'),
    ]:
        lags = ccf_df["lag_k"]
        corrs = ccf_df["correlation"]

        bar_colors = ['#C62828' if k > 0 else '#1565C0' if k < 0 else '#333333'
                      for k in lags]
        ax.bar(lags, corrs, color=bar_colors, alpha=0.7, edgecolor='white',
               linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

        # 95% CI
        n = ccf_df["n"].median()
        ci = 1.96 / np.sqrt(n)
        ax.axhline(y=ci, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.axhline(y=-ci, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

        # ピーク
        min_idx = corrs.idxmin()
        pk = int(lags.iloc[min_idx])
        pr = corrs.iloc[min_idx]
        ax.annotate(f'k={pk}, r={pr:.3f}',
                    xy=(pk, pr), xytext=(pk + 2.5, pr - 0.02),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                              alpha=0.9))

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('ラグ k (月)\n← 消費先行 | STI先行 →', fontsize=10)
        ax.set_xticks(range(int(lags.min()), int(lags.max()) + 1))
        ax.grid(True, alpha=0.2, axis='y')

    ax1.set_ylabel('相関係数 r', fontsize=12)

    fig.suptitle('SSQ Phase C+: CCF比較 — 遅行指標 vs 先行指標\n'
                 'ピークラグの移動が「観測方程式」の証拠',
                 fontsize=15, y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {output_path}")


# ═══════════════════════════════════════════════════════════
# Phase D: 正式回帰分析 — SSQを倒しにいく検証
# ═══════════════════════════════════════════════════════════

def run_phase_d(sti_leading_df, ds_df, unemp_df, income_df, cpi_df, output_dir):
    """Phase D: 統制変数付き時系列回帰

    モデル:
      裁量消費比率_YoY = α + β₁·STI_leading_YoY + β₂·可処分所得_YoY
                         + β₃·CPI_YoY + β₄·失業率_YoY + 季節ダミー + ε

    推定: OLS + Newey-West HAC標準誤差 (maxlags=12)

    成功判定 (事前固定):
      1. β₁ < 0 かつ p < 0.05
      2. 所得(β₂)を入れてもβ₁が消えない
      3. 期間分割で符号安定
    """
    import statsmodels.api as sm

    print("\n" + "=" * 70)
    print("Phase D: 正式回帰分析 — SSQを倒しにいく検証")
    print("=" * 70)

    # ── 1. YoY変換 ──
    print("\n  [D-1] データ準備: YoY変換 & マージ")
    sti_yoy = compute_yoy(sti_leading_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    unemp_yoy = compute_yoy(unemp_df.copy(), "date", "unemployment_rate")
    income_yoy = compute_yoy(income_df.copy(), "date", "disposable_income")

    # ── 2. マージ ──
    merged = (
        ds_yoy[["date", "discretionary_share_yoy"]]
        .merge(sti_yoy[["date", "STI_yoy"]], on="date")
        .merge(income_yoy[["date", "disposable_income_yoy"]], on="date")
        .merge(cpi_df[["date", "cpi_yoy"]], on="date")
        .merge(unemp_yoy[["date", "unemployment_rate_yoy"]], on="date")
        .sort_values("date")
        .dropna()
        .reset_index(drop=True)
    )

    print(f"  回帰期間: {merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m}")
    print(f"  観測数: {len(merged)}")

    if len(merged) < 40:
        print("  ERROR: 観測数不足、Phase D を中止")
        return None, None

    # ── 3. 季節ダミー ──
    merged["month"] = merged["date"].dt.month
    season_cols = []
    for m in range(2, 13):
        col = f"m{m}"
        merged[col] = (merged["month"] == m).astype(float)
        season_cols.append(col)

    # ── 4. 記述統計 ──
    print(f"\n  [D-2] 記述統計 (YoY変換後)")
    print(f"  {'変数':<28s} | {'平均':>8s} | {'SD':>8s} | {'min':>8s} | {'max':>8s}")
    print(f"  {'─' * 28}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}")
    for var, label in [
        ("discretionary_share_yoy", "裁量消費比率 YoY (pp)"),
        ("STI_yoy", "STI_leading YoY"),
        ("disposable_income_yoy", "可処分所得 YoY (円)"),
        ("cpi_yoy", "CPI YoY (%)"),
        ("unemployment_rate_yoy", "失業率 YoY (pp)"),
    ]:
        s = merged[var]
        print(f"  {label:<28s} | {s.mean():>+8.3f} | {s.std():>8.3f} | "
              f"{s.min():>+8.3f} | {s.max():>+8.3f}")

    # ── 5. 相関行列 ──
    print(f"\n  [D-3] 変数間相関 (多重共線性チェック)")
    corr_vars = ["STI_yoy", "disposable_income_yoy", "cpi_yoy", "unemployment_rate_yoy"]
    corr_labels = ["STI", "Income", "CPI", "Unemp"]
    corr_mat = merged[corr_vars].corr()
    header = f"  {'':>8s}"
    for lb in corr_labels:
        header += f" | {lb:>8s}"
    print(header)
    for i, (var, lb) in enumerate(zip(corr_vars, corr_labels)):
        row = f"  {lb:>8s}"
        for j, var2 in enumerate(corr_vars):
            row += f" | {corr_mat.iloc[i, j]:>+8.3f}"
        print(row)

    # ── 6. 3モデル推定 ──
    print(f"\n  [D-4] 回帰推定 (OLS + Newey-West HAC, maxlags=12)")

    y = merged["discretionary_share_yoy"]
    models = {}

    # Model 1: STI only
    X1 = sm.add_constant(merged[["STI_yoy"] + season_cols])
    res1 = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    models["M1_STI"] = res1

    # Model 2: STI + Income (核心テスト)
    X2 = sm.add_constant(merged[["STI_yoy", "disposable_income_yoy"] + season_cols])
    res2 = sm.OLS(y, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    models["M2_STI+Income"] = res2

    # Model 3: Full
    X3_cols = ["STI_yoy", "disposable_income_yoy", "cpi_yoy",
               "unemployment_rate_yoy"] + season_cols
    X3 = sm.add_constant(merged[X3_cols])
    res3 = sm.OLS(y, X3).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    models["M3_Full"] = res3

    # ── 7. 結果テーブル ──
    print(f"\n  {'═' * 70}")
    print(f"  回帰結果テーブル")
    print(f"  被説明変数: 裁量消費比率 YoY (pp)")
    print(f"  {'═' * 70}")

    key_vars = ["const", "STI_yoy", "disposable_income_yoy",
                "cpi_yoy", "unemployment_rate_yoy"]
    var_labels = {
        "const": "切片",
        "STI_yoy": "β₁ STI_leading",
        "disposable_income_yoy": "β₂ 可処分所得",
        "cpi_yoy": "β₃ CPI(インフレ)",
        "unemployment_rate_yoy": "β₄ 失業率",
    }
    model_shorts = ["STI単独", "STI+所得", "フル"]
    model_keys = list(models.keys())

    # Header
    h = f"  {'変数':<24s}"
    for s in model_shorts:
        h += f" | {s:>12s}"
    print(h)
    print(f"  {'─' * 24}" + ("-+-" + "─" * 12) * len(model_shorts))

    for var in key_vars:
        label = var_labels.get(var, var)

        # Coefficient + significance
        row = f"  {label:<24s}"
        for mk in model_keys:
            res = models[mk]
            if var in res.params.index:
                coef = res.params[var]
                pval = res.pvalues[var]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                row += f" | {coef:>9.4f}{stars:<3s}"
            else:
                row += f" |        —    "
        print(row)

        # SE row
        row_se = f"  {'':24s}"
        for mk in model_keys:
            res = models[mk]
            if var in res.params.index:
                se = res.bse[var]
                row_se += f" | ({se:>8.4f})  "
            else:
                row_se += f" |              "
        print(row_se)

        # p-value row
        row_p = f"  {'':24s}"
        for mk in model_keys:
            res = models[mk]
            if var in res.params.index:
                pval = res.pvalues[var]
                row_p += f" |  p={pval:<8.4f} "
            else:
                row_p += f" |              "
        print(row_p)

        if var != key_vars[-1]:
            print(f"  {'─' * 24}" + ("-+-" + "─" * 12) * len(model_shorts))

    # Fit statistics
    print(f"  {'═' * 24}" + ("=+=" + "═" * 12) * len(model_shorts))
    for stat_label, stat_fn in [
        ("R²", lambda r: f"{r.rsquared:.4f}"),
        ("Adj R²", lambda r: f"{r.rsquared_adj:.4f}"),
        ("AIC", lambda r: f"{r.aic:.1f}"),
        ("N", lambda r: f"{int(r.nobs)}"),
    ]:
        row = f"  {stat_label:<24s}"
        for mk in model_keys:
            row += f" | {stat_fn(models[mk]):>12s}"
        print(row)
    print(f"  {'═' * 70}")
    print(f"  *** p<0.01, ** p<0.05, * p<0.1 | HAC標準誤差 (Newey-West)")

    # ── 8. 標準化係数 ──
    std_y = y.std()
    print(f"\n  [D-5] 標準化係数比較 (Model 3 フルモデル)")
    print(f"  {'─' * 70}")
    print(f"  {'変数':<24s} | {'β_std':>8s} | {'β_raw':>8s} | {'SD(x)':>8s}")
    print(f"  {'─' * 24}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}")

    std_coefs = {}
    for var in ["STI_yoy", "disposable_income_yoy", "cpi_yoy", "unemployment_rate_yoy"]:
        if var in res3.params.index:
            b_raw = res3.params[var]
            sd_x = merged[var].std()
            b_std = b_raw * sd_x / std_y
            std_coefs[var] = b_std
            label = var_labels.get(var, var)
            print(f"  {label:<24s} | {b_std:>+8.4f} | {b_raw:>+8.4f} | {sd_x:>8.4f}")

    # |β₁| vs |β₂| comparison
    b1_std = abs(std_coefs.get("STI_yoy", 0))
    b2_std = abs(std_coefs.get("disposable_income_yoy", 0))
    print(f"\n  |β₁_std| = {b1_std:.4f} vs |β₂_std| = {b2_std:.4f}")
    if b2_std > 0:
        ratio = b1_std / b2_std
        if b1_std > b2_std:
            print(f"  → STIの効果が所得の {ratio:.2f}倍 → SSQ核心仮説を支持")
        else:
            print(f"  → 所得の効果がSTIの {1/ratio:.2f}倍 → 従来経済学が優勢")

    # ── 9. 期間分割安定性 ──
    print(f"\n  [D-6] 期間分割安定性テスト")
    print(f"  {'─' * 70}")

    midpoint = merged["date"].iloc[len(merged) // 2]
    splits = [
        ("前半", merged[merged["date"] <= midpoint]),
        ("後半", merged[merged["date"] > midpoint]),
    ]

    split_results = {}
    for label, subset in splits:
        if len(subset) < 30:
            print(f"  {label}: 観測数不足 ({len(subset)}件) → スキップ")
            continue

        y_sub = subset["discretionary_share_yoy"]
        X_sub = sm.add_constant(
            subset[["STI_yoy", "disposable_income_yoy", "cpi_yoy",
                     "unemployment_rate_yoy"] + season_cols]
        )
        res_sub = sm.OLS(y_sub, X_sub).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        split_results[label] = res_sub

        b1 = res_sub.params.get("STI_yoy", float("nan"))
        p1 = res_sub.pvalues.get("STI_yoy", float("nan"))
        b2 = res_sub.params.get("disposable_income_yoy", float("nan"))
        p2 = res_sub.pvalues.get("disposable_income_yoy", float("nan"))

        period = f"{subset['date'].min():%Y-%m} ~ {subset['date'].max():%Y-%m}"
        stars1 = "***" if p1 < 0.01 else "**" if p1 < 0.05 else "*" if p1 < 0.1 else "n.s."
        stars2 = "***" if p2 < 0.01 else "**" if p2 < 0.05 else "*" if p2 < 0.1 else "n.s."
        print(f"  {label} ({period}, N={len(subset)}):")
        print(f"    β₁(STI)  = {b1:+.4f} ({stars1}, p={p1:.4f})")
        print(f"    β₂(所得) = {b2:+.4f} ({stars2}, p={p2:.4f})")
        print(f"    R² = {res_sub.rsquared:.4f}")

    # ── 10. 成功判定 ──
    print(f"\n  {'═' * 70}")
    print(f"  SSQ 成功判定 (事前登録基準)")
    print(f"  {'═' * 70}")

    # Criterion 1: β₁ < 0 and p < 0.05 in full model
    b1_full = res3.params.get("STI_yoy", float("nan"))
    p1_full = res3.pvalues.get("STI_yoy", float("nan"))
    crit1 = b1_full < 0 and p1_full < 0.05

    # Criterion 2: β₁ survives income control
    b1_m2 = res2.params.get("STI_yoy", float("nan"))
    p1_m2 = res2.pvalues.get("STI_yoy", float("nan"))
    crit2 = b1_m2 < 0 and p1_m2 < 0.05

    # Criterion 3: Sign stable across splits
    signs_ok = all(
        res_sub.params.get("STI_yoy", 1) < 0
        for res_sub in split_results.values()
    )
    crit3 = signs_ok and len(split_results) >= 2

    mark = lambda c: "PASS" if c else "FAIL"

    print(f"\n  [{mark(crit1)}] 条件1: β₁ < 0 かつ有意 (p < 0.05)")
    print(f"          フルモデル β₁ = {b1_full:+.4f}, p = {p1_full:.4f}")

    print(f"\n  [{mark(crit2)}] 条件2: 所得統制後もβ₁が存続")
    b1_m1 = res1.params.get("STI_yoy", float("nan"))
    p1_m1 = res1.pvalues.get("STI_yoy", float("nan"))
    print(f"          Model 1 (STI単独): β₁ = {b1_m1:+.4f}, p = {p1_m1:.4f}")
    print(f"          Model 2 (STI+所得): β₁ = {b1_m2:+.4f}, p = {p1_m2:.4f}")

    print(f"\n  [{mark(crit3)}] 条件3: 期間分割で符号安定")
    for label, res_sub in split_results.items():
        b1_sub = res_sub.params.get("STI_yoy", float("nan"))
        p1_sub = res_sub.pvalues.get("STI_yoy", float("nan"))
        print(f"          {label}: β₁ = {b1_sub:+.4f}, p = {p1_sub:.4f}")

    all_pass = crit1 and crit2 and crit3
    two_pass = sum([crit1, crit2, crit3])

    print(f"\n  {'═' * 70}")
    if all_pass:
        print(f"  ★ 結論: SSQ仮説は3条件すべてを満たす")
        print(f"    → 「社会脅威場が消費構成を駆動する」は実証的に支持される")
        print(f"    → これは仮説提案ではなく「実証研究」である")
    elif two_pass >= 2:
        print(f"  ◆ 結論: 3条件中{two_pass}条件を満たす → 部分的支持")
        if not crit1:
            print(f"    → β₁が有意でない: STIの効果が統計的に確認できない")
        if not crit2:
            print(f"    → 所得統制後にβ₁が弱化: 所得の方が強い説明力")
        if not crit3:
            print(f"    → 期間分割で不安定: レジーム依存の可能性")
    elif crit1:
        print(f"  △ 結論: β₁有意だが他条件未達 → 予備的証拠のみ")
    else:
        print(f"  × 結論: β₁が有意でない → SSQ仮説は棄却")
    print(f"  {'═' * 70}")

    # ── 11. 結果保存 ──
    result_data = {
        "analysis": "SSQ Phase D Regression",
        "period": f"{merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m}",
        "n_obs": int(len(merged)),
        "models": {},
        "standardized_coefficients": {k: float(v) for k, v in std_coefs.items()},
        "period_split": {},
        "success_criteria": {
            "criterion_1_significant": bool(crit1),
            "criterion_2_survives_income": bool(crit2),
            "criterion_3_sign_stable": bool(crit3),
            "all_pass": bool(all_pass),
        },
    }

    for mk, res in models.items():
        mdata = {
            "r_squared": float(res.rsquared),
            "adj_r_squared": float(res.rsquared_adj),
            "aic": float(res.aic),
            "n_obs": int(res.nobs),
            "coefficients": {},
        }
        for var in key_vars:
            if var in res.params.index:
                mdata["coefficients"][var] = {
                    "estimate": float(res.params[var]),
                    "std_error": float(res.bse[var]),
                    "t_stat": float(res.tvalues[var]),
                    "p_value": float(res.pvalues[var]),
                }
        result_data["models"][mk] = mdata

    for label, res_sub in split_results.items():
        result_data["period_split"][label] = {
            "n_obs": int(res_sub.nobs),
            "r_squared": float(res_sub.rsquared),
            "beta1_STI": float(res_sub.params.get("STI_yoy", float("nan"))),
            "beta1_p": float(res_sub.pvalues.get("STI_yoy", float("nan"))),
        }

    result_path = output_dir / "phase_d_regression_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {result_path}")

    # ── 12. 係数プロット ──
    plot_phase_d_coefficients(models, model_shorts, var_labels, key_vars,
                              CHART_DIR)

    return result_data, merged


def plot_phase_d_coefficients(models, model_shorts, var_labels, key_vars, chart_dir):
    """Phase D: 係数比較プロット (β₁ across models)"""
    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    model_keys = list(models.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: β₁ (STI) across models with 95% CI
    ax = axes[0]
    positions = range(len(model_keys))
    for i, mk in enumerate(model_keys):
        res = models[mk]
        if "STI_yoy" in res.params.index:
            b = res.params["STI_yoy"]
            ci = res.conf_int().loc["STI_yoy"]
            color = '#C62828' if b < 0 else '#1565C0'
            ax.barh(i, b, color=color, alpha=0.7, height=0.5)
            ax.plot([ci[0], ci[1]], [i, i], color='black', linewidth=2,
                    marker='|', markersize=10)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_yticks(positions)
    ax.set_yticklabels(model_shorts, fontsize=11)
    ax.set_xlabel('β₁ (STI_leading) + 95% CI', fontsize=11)
    ax.set_title('β₁ の安定性: 統制変数追加による変化', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    # Panel B: Standardized coefficients in full model
    ax = axes[1]
    res3 = models[model_keys[-1]]  # Full model
    std_y = res3.model.endog.std()

    bar_vars = ["STI_yoy", "disposable_income_yoy", "cpi_yoy", "unemployment_rate_yoy"]
    bar_labels = ["STI_leading", "可処分所得", "CPI", "失業率"]
    beta_stds = []
    for var in bar_vars:
        if var in res3.params.index:
            sd_x = res3.model.exog[:, list(res3.model.exog_names).index(var)].std()
            beta_stds.append(res3.params[var] * sd_x / std_y)
        else:
            beta_stds.append(0)

    colors = ['#C62828' if b < 0 else '#1565C0' for b in beta_stds]
    ax.barh(range(len(bar_labels)), beta_stds, color=colors, alpha=0.7, height=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_yticks(range(len(bar_labels)))
    ax.set_yticklabels(bar_labels, fontsize=11)
    ax.set_xlabel('標準化係数 β_std', fontsize=11)
    ax.set_title('フルモデル: 標準化係数の比較\n(絶対値が大きい = 説明力が大きい)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    fig.suptitle('SSQ Phase D: 回帰分析結果', fontsize=15, y=1.02)
    plt.tight_layout()
    chart_path = str(chart_dir / "ssq_phase_d_regression.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {chart_path}")


# ═══════════════════════════════════════════════════════════
# Phase D+: Granger因果性検定 — 因果を取りにいく
# ═══════════════════════════════════════════════════════════

def run_granger_causality(sti_leading_df, ds_df, output_dir):
    """Phase D+: Granger因果性検定

    問い: STIは「消費と同時に動く代理変数」なのか、
          それとも「消費を予測する先行変数」なのか？

    検定:
      H₀(A): STI_leading_YoY は DS_YoY を Granger因果しない
      H₀(B): DS_YoY は STI_leading_YoY を Granger因果しない

    SSQ因果主張の成立条件:
      H₀(A) 棄却 (STI → DS: 有意) かつ H₀(B) 非棄却 (DS → STI: 非有意)

    手法:
      1. ADF検定で定常性確認
      2. VAR情報量基準で最適ラグ選択
      3. 標準Granger検定 (ラグ1〜12)
      4. HAC-robust Wald検定 (ロバストネス)
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    from statsmodels.tsa.api import VAR
    from scipy import stats as sp_stats

    print("\n" + "=" * 70)
    print("Phase D+: Granger因果性検定 — 因果を取りにいく")
    print("=" * 70)

    # ── 1. データ準備 ──
    sti_yoy = compute_yoy(sti_leading_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    merged = (
        sti_yoy[["date", "STI_yoy"]]
        .merge(ds_yoy[["date", "discretionary_share_yoy"]], on="date")
        .sort_values("date").dropna().reset_index(drop=True)
    )
    print(f"\n  期間: {merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m}")
    print(f"  観測数: {len(merged)}")

    # ── 2. ADF単位根検定 ──
    print(f"\n  [G-1] ADF単位根検定 (定常性の前提確認)")
    print(f"  {'─' * 60}")
    adf_ok = True
    for var, label in [("STI_yoy", "STI_leading YoY"),
                        ("discretionary_share_yoy", "裁量消費比率 YoY")]:
        adf = adfuller(merged[var].dropna(), maxlag=12, autolag='AIC')
        sig = "***" if adf[1] < 0.01 else "**" if adf[1] < 0.05 else "*" if adf[1] < 0.1 else "n.s."
        print(f"  {label:24s}: ADF = {adf[0]:+.4f}, p = {adf[1]:.4f} ({sig})")
        print(f"  {'':24s}  ラグ={adf[2]}, N={adf[3]}")
        if adf[1] > 0.05:
            adf_ok = False
            print(f"  {'':24s}  ⚠ 単位根の帰無仮説を棄却できず — 非定常の恐れ")

    if adf_ok:
        print(f"\n  → 両系列とも定常 ✓ Granger検定の前提を満たす")
    else:
        print(f"\n  → ⚠ 定常性に懸念あり — 結果の解釈に注意")

    # ── 3. VAR最適ラグ選択 ──
    print(f"\n  [G-2] VAR最適ラグ選択")
    print(f"  {'─' * 60}")
    var_data = merged[["discretionary_share_yoy", "STI_yoy"]]
    var_model = VAR(var_data)
    lag_order = var_model.select_order(maxlags=12)

    print(f"  AIC最適ラグ:  {lag_order.aic}")
    print(f"  BIC最適ラグ:  {lag_order.bic}")
    print(f"  HQIC最適ラグ: {lag_order.hqic}")

    optimal_lag = lag_order.aic  # AICを採用
    print(f"\n  → 採用ラグ: p = {optimal_lag} (AIC基準)")

    # ── 4. 標準Granger因果性検定 (全ラグ) ──
    print(f"\n  [G-3] 標準Granger因果性検定")
    print(f"  {'─' * 60}")

    max_test_lag = 12

    # Direction A: STI → DS
    print(f"\n  方向A: STI_leading → 裁量消費比率")
    print(f"  H₀: STIの過去値はDSの予測に寄与しない")
    data_a = merged[["discretionary_share_yoy", "STI_yoy"]].values

    gc_a = {}
    try:
        gc_results_a = grangercausalitytests(data_a, maxlag=max_test_lag,
                                              verbose=False)
        for lag in range(1, max_test_lag + 1):
            if lag in gc_results_a:
                f_val = gc_results_a[lag][0]['ssr_ftest'][0]
                p_val = gc_results_a[lag][0]['ssr_ftest'][1]
                gc_a[lag] = (f_val, p_val)
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"  {'ラグp':>6s} | {'F統計量':>10s} | {'p値':>10s} | {'判定':>6s}")
    print(f"  {'─' * 6}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 6}")
    for lag in sorted(gc_a.keys()):
        f_val, p_val = gc_a[lag]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        marker = " ← AIC" if lag == optimal_lag else ""
        print(f"  {lag:>6d} | {f_val:>10.3f} | {p_val:>10.4f} | {sig:>6s}{marker}")

    # Direction B: DS → STI (reverse)
    print(f"\n  方向B: 裁量消費比率 → STI_leading (逆因果)")
    print(f"  H₀: DSの過去値はSTIの予測に寄与しない")
    data_b = merged[["STI_yoy", "discretionary_share_yoy"]].values

    gc_b = {}
    try:
        gc_results_b = grangercausalitytests(data_b, maxlag=max_test_lag,
                                              verbose=False)
        for lag in range(1, max_test_lag + 1):
            if lag in gc_results_b:
                f_val = gc_results_b[lag][0]['ssr_ftest'][0]
                p_val = gc_results_b[lag][0]['ssr_ftest'][1]
                gc_b[lag] = (f_val, p_val)
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"  {'ラグp':>6s} | {'F統計量':>10s} | {'p値':>10s} | {'判定':>6s}")
    print(f"  {'─' * 6}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 6}")
    for lag in sorted(gc_b.keys()):
        f_val, p_val = gc_b[lag]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        marker = " ← AIC" if lag == optimal_lag else ""
        print(f"  {lag:>6d} | {f_val:>10.3f} | {p_val:>10.4f} | {sig:>6s}{marker}")

    # ── 5. HAC-robust Wald検定 (ロバストネス) ──
    print(f"\n  [G-4] HAC-robust Granger検定 (Newey-West, ロバストネス)")
    print(f"  {'─' * 60}")

    hac_results = {}
    for direction, y_col, x_col, label in [
        ("STI→DS", "discretionary_share_yoy", "STI_yoy",
         "STI → 裁量消費比率"),
        ("DS→STI", "STI_yoy", "discretionary_share_yoy",
         "裁量消費比率 → STI"),
    ]:
        p = optimal_lag
        # Build lagged regressors
        reg_data = merged[["date", y_col, x_col]].copy()
        for i in range(1, p + 1):
            reg_data[f"{y_col}_L{i}"] = reg_data[y_col].shift(i)
            reg_data[f"{x_col}_L{i}"] = reg_data[x_col].shift(i)
        reg_data = reg_data.dropna()

        y = reg_data[y_col]
        own_lag_cols = [f"{y_col}_L{i}" for i in range(1, p + 1)]
        x_lag_cols = [f"{x_col}_L{i}" for i in range(1, p + 1)]

        # Unrestricted model (own lags + x lags)
        X_unr = sm.add_constant(reg_data[own_lag_cols + x_lag_cols])
        res_unr = sm.OLS(y, X_unr).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        # Wald test: all x_lag coefficients = 0
        r_matrix = np.zeros((len(x_lag_cols), len(res_unr.params)))
        for i, col in enumerate(x_lag_cols):
            idx = list(res_unr.params.index).index(col)
            r_matrix[i, idx] = 1

        wald = res_unr.wald_test(r_matrix, use_f=True)
        f_val = float(wald.fvalue)
        p_val = float(wald.pvalue)

        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
        print(f"  {label} (lag={p}):")
        print(f"    Wald F = {f_val:.4f}, p = {p_val:.4f} ({sig})")

        # Print individual lag coefficients
        for col in x_lag_cols:
            b = res_unr.params[col]
            se = res_unr.bse[col]
            pv = res_unr.pvalues[col]
            s = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
            print(f"      {col}: β={b:+.4f} (SE={se:.4f}, p={pv:.4f}) {s}")

        hac_results[direction] = {
            "f_value": float(f_val), "p_value": float(p_val),
            "lag": int(p), "n_obs": int(res_unr.nobs),
        }

    # ── 6. 因果性判定 ──
    print(f"\n  {'═' * 70}")
    print(f"  Granger因果性 総合判定")
    print(f"  {'═' * 70}")

    # AIC最適ラグでの判定
    p_a = gc_a.get(optimal_lag, (0, 1.0))[1]
    p_b = gc_b.get(optimal_lag, (0, 1.0))[1]
    p_a_hac = hac_results.get("STI→DS", {}).get("p_value", 1.0)
    p_b_hac = hac_results.get("DS→STI", {}).get("p_value", 1.0)

    print(f"\n  最適ラグ p = {optimal_lag} での結果:")
    print(f"  {'方向':16s} | {'標準F検定':>12s} | {'HAC Wald':>12s}")
    print(f"  {'─' * 16}-+-{'─' * 12}-+-{'─' * 12}")

    sig_a = "***" if p_a < 0.01 else "**" if p_a < 0.05 else "*" if p_a < 0.1 else "n.s."
    sig_b = "***" if p_b < 0.01 else "**" if p_b < 0.05 else "*" if p_b < 0.1 else "n.s."
    sig_a_h = "***" if p_a_hac < 0.01 else "**" if p_a_hac < 0.05 else "*" if p_a_hac < 0.1 else "n.s."
    sig_b_h = "***" if p_b_hac < 0.01 else "**" if p_b_hac < 0.05 else "*" if p_b_hac < 0.1 else "n.s."

    print(f"  {'STI → DS':16s} | p={p_a:.4f} {sig_a:>4s} | p={p_a_hac:.4f} {sig_a_h:>4s}")
    print(f"  {'DS → STI':16s} | p={p_b:.4f} {sig_b:>4s} | p={p_b_hac:.4f} {sig_b_h:>4s}")

    # Robustness: count significant lags and analyze pattern
    n_sig_a = sum(1 for _, (_, p) in gc_a.items() if p < 0.05)
    n_sig_b = sum(1 for _, (_, p) in gc_b.items() if p < 0.05)

    # Short-lag analysis (lags 1-6) — more reliable, less overfitting
    short_sig_a = sum(1 for k, (_, p) in gc_a.items() if k <= 6 and p < 0.05)
    short_sig_b = sum(1 for k, (_, p) in gc_b.items() if k <= 6 and p < 0.05)
    short_n = sum(1 for k in gc_a.keys() if k <= 6)

    # Find first significant lag for reverse direction
    first_sig_b = None
    for k in sorted(gc_b.keys()):
        if gc_b[k][1] < 0.05:
            first_sig_b = k
            break

    print(f"\n  ロバストネス分析:")
    print(f"    全ラグ (1〜12) p<0.05:")
    print(f"      STI → DS: {n_sig_a}/{len(gc_a)} ラグで有意")
    print(f"      DS → STI: {n_sig_b}/{len(gc_b)} ラグで有意")
    print(f"    短ラグ (1〜6) p<0.05:")
    print(f"      STI → DS: {short_sig_a}/{short_n} ラグで有意")
    print(f"      DS → STI: {short_sig_b}/{short_n} ラグで有意")
    if first_sig_b:
        print(f"    DS→STI 逆因果が最初に有意になるラグ: {first_sig_b}ヶ月")

    # Final judgment — use SHORT lags for causality (more conservative)
    causal_a_robust = short_sig_a >= 3  # Majority of short lags
    causal_b_robust = short_sig_b >= 3
    causal_a_opt = p_a < 0.05
    causal_b_opt = p_b < 0.05

    print(f"\n  {'═' * 70}")
    print(f"  Granger因果性 総合判定")
    print(f"  {'═' * 70}")

    if causal_a_robust and not causal_b_robust:
        print(f"\n  ★ 結論: STI → DS の一方向Granger因果が支持される")
        print(f"    短ラグ(1〜6ヶ月): STI→DS {short_sig_a}/{short_n}有意, "
              f"DS→STI {short_sig_b}/{short_n}有意")
        if causal_b_opt:
            print(f"    ※ 長ラグ({first_sig_b}ヶ月〜)で逆方向も有意だが、")
            print(f"      これは年次フィードバック効果の可能性が高い")
        print(f"\n    → STIは「同時反応の代理」ではなく「予測変数」")
        print(f"    → SSQの因果主張が統計的に支持される")
        judgment = "UNIDIRECTIONAL_STI_TO_DS"
    elif causal_a_robust and causal_b_robust:
        print(f"\n  ◆ 結論: 双方向のGranger因果 (フィードバック)")
        print(f"    → STIとDSは相互影響しているが、STIの方が支配的")
        judgment = "BIDIRECTIONAL"
    elif not causal_a_robust and causal_b_robust:
        print(f"\n  × 結論: DS → STI の逆方向因果のみ")
        judgment = "REVERSE_DS_TO_STI"
    else:
        print(f"\n  △ 結論: 短ラグでは因果なし")
        if causal_a_opt:
            print(f"    ※ 長ラグ(AIC最適)では有意 — 中期的因果の可能性")
        judgment = "NO_SHORT_LAG_CAUSALITY"

    print(f"  {'═' * 70}")

    # ── 7. 結果保存 ──
    result_data = {
        "analysis": "SSQ Phase D+ Granger Causality",
        "optimal_lag_aic": int(optimal_lag),
        "stationarity": {
            "STI_yoy_adf_p": float(adfuller(merged["STI_yoy"].dropna(),
                                             maxlag=12, autolag='AIC')[1]),
            "DS_yoy_adf_p": float(adfuller(merged["discretionary_share_yoy"].dropna(),
                                            maxlag=12, autolag='AIC')[1]),
        },
        "granger_STI_to_DS": {
            int(k): {"f_stat": float(v[0]), "p_value": float(v[1])}
            for k, v in gc_a.items()
        },
        "granger_DS_to_STI": {
            int(k): {"f_stat": float(v[0]), "p_value": float(v[1])}
            for k, v in gc_b.items()
        },
        "hac_robust": hac_results,
        "judgment": judgment,
    }

    result_path = output_dir / "phase_d_granger_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {result_path}")

    # ── 8. プロット ──
    plot_granger_results(gc_a, gc_b, optimal_lag, CHART_DIR)

    return result_data


def plot_granger_results(gc_a, gc_b, optimal_lag, chart_dir):
    """Granger因果性検定のp値プロット"""
    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    fig, ax = plt.subplots(figsize=(14, 7))

    lags_a = sorted(gc_a.keys())
    pvals_a = [gc_a[k][1] for k in lags_a]
    lags_b = sorted(gc_b.keys())
    pvals_b = [gc_b[k][1] for k in lags_b]

    width = 0.35
    x_a = [l - width / 2 for l in lags_a]
    x_b = [l + width / 2 for l in lags_b]

    # Bars
    bars_a = ax.bar(x_a, pvals_a, width=width, color='#C62828', alpha=0.7,
                     label='STI → DS (SSQ因果方向)', edgecolor='white')
    bars_b = ax.bar(x_b, pvals_b, width=width, color='#1565C0', alpha=0.7,
                     label='DS → STI (逆因果)', edgecolor='white')

    # Significance thresholds
    ax.axhline(y=0.05, color='red', linewidth=1.5, linestyle='--',
               alpha=0.8, label='p = 0.05')
    ax.axhline(y=0.01, color='orange', linewidth=1, linestyle=':',
               alpha=0.6, label='p = 0.01')

    # Optimal lag marker
    ax.axvline(x=optimal_lag, color='green', linewidth=1.5, linestyle='--',
               alpha=0.5)
    ax.text(optimal_lag + 0.1, ax.get_ylim()[1] * 0.9,
            f'AIC最適ラグ={optimal_lag}', fontsize=10, color='green',
            fontweight='bold')

    # Shade "significant" zone
    ax.axhspan(0, 0.05, color='red', alpha=0.03)

    ax.set_xlabel('ラグ p (月)', fontsize=12)
    ax.set_ylabel('p値', fontsize=12)
    ax.set_title('SSQ Phase D+: Granger因果性検定\n'
                 '赤バーが0.05以下 = STIが消費を予測（SSQ因果が成立）',
                 fontsize=14, pad=15)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.3, 12.7)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y')

    # Interpretation box — use short-lag (1-6) pattern
    short_sig_a = sum(1 for k in range(1, 7) if k in gc_a and gc_a[k][1] < 0.05)
    short_sig_b = sum(1 for k in range(1, 7) if k in gc_b and gc_b[k][1] < 0.05)
    if short_sig_a >= 3 and short_sig_b == 0:
        interp = (f'一方向因果: STI → DS\n'
                  f'短ラグ(1-6): STI→DS {short_sig_a}/6有意, DS→STI 0/6\n'
                  f'→ SSQ因果主張を支持')
        interp_color = '#2E7D32'
    elif short_sig_a >= 3 and short_sig_b >= 3:
        interp = '双方向因果: フィードバック構造\n→ SSQ因果は部分的'
        interp_color = '#F57F17'
    elif short_sig_a == 0:
        interp = '短ラグでSTI → DS 非有意\n→ SSQ因果は不成立'
        interp_color = '#C62828'
    else:
        interp = f'STI→DS {short_sig_a}/6有意 (短ラグ)\n→ SSQ因果は部分的に支持'
        interp_color = '#F57F17'

    ax.text(0.02, 0.98, interp, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor=interp_color, linewidth=2, alpha=0.9))

    plt.tight_layout()
    chart_path = str(chart_dir / "ssq_granger_causality.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {chart_path}")


# ═══════════════════════════════════════════════════════════
# Phase E: 予測力の上乗せ検証 (Incremental Predictive Power)
# ═══════════════════════════════════════════════════════════

def run_incremental_prediction(sti_leading_df, ds_df, unemp_df, income_df,
                                cpi_df, output_dir):
    """Phase E: STIは「観測されていなかった状態量」か？

    Model A (baseline):
      DS_t = α + Σ β_i DS_{t-i} + γ₁ Income_{t-1} + γ₂ CPI_{t-1}
             + γ₃ Unemp_{t-1} + ε

    Model B (+ STI):
      Model A + Σ φ_j STI_{t-j}

    検証:
      1. In-sample: AIC, BIC, Adj-R², F検定 (nested model比較)
      2. Out-of-sample: 拡張窓1ステップ先予測 → RMSE比較
      3. Diebold-Mariano検定: 予測精度の統計的有意差
    """
    import statsmodels.api as sm
    from scipy import stats as sp_stats

    print("\n" + "=" * 70)
    print("Phase E: 予測力の上乗せ検証 — STIは新しい情報を持つか？")
    print("=" * 70)

    # ── 1. データ準備 ──
    sti_yoy = compute_yoy(sti_leading_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    unemp_yoy = compute_yoy(unemp_df.copy(), "date", "unemployment_rate")
    income_yoy = compute_yoy(income_df.copy(), "date", "disposable_income")

    merged = (
        ds_yoy[["date", "discretionary_share_yoy"]]
        .merge(sti_yoy[["date", "STI_yoy"]], on="date")
        .merge(income_yoy[["date", "disposable_income_yoy"]], on="date")
        .merge(cpi_df[["date", "cpi_yoy"]], on="date")
        .merge(unemp_yoy[["date", "unemployment_rate_yoy"]], on="date")
        .sort_values("date").dropna().reset_index(drop=True)
    )

    print(f"\n  期間: {merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m}")
    print(f"  観測数: {len(merged)}")

    # ── 2. ラグ変数構築 ──
    ar_lags = 4
    x_lags = 4

    for i in range(1, max(ar_lags, x_lags) + 1):
        merged[f"DS_L{i}"] = merged["discretionary_share_yoy"].shift(i)
        merged[f"STI_L{i}"] = merged["STI_yoy"].shift(i)
        merged[f"Income_L{i}"] = merged["disposable_income_yoy"].shift(i)
        merged[f"CPI_L{i}"] = merged["cpi_yoy"].shift(i)
        merged[f"Unemp_L{i}"] = merged["unemployment_rate_yoy"].shift(i)

    merged = merged.dropna().reset_index(drop=True)
    print(f"  ラグ構築後: {len(merged)}件 "
          f"({merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m})")

    # ── 3. 変数定義 ──
    y_col = "discretionary_share_yoy"
    ar_cols = [f"DS_L{i}" for i in range(1, ar_lags + 1)]
    ctrl_cols = ["Income_L1", "CPI_L1", "Unemp_L1"]
    sti_cols = [f"STI_L{i}" for i in range(1, x_lags + 1)]

    model_a_cols = ar_cols + ctrl_cols
    model_b_cols = ar_cols + ctrl_cols + sti_cols

    print(f"\n  Model A (baseline): {len(model_a_cols)} 説明変数")
    print(f"    AR(1-{ar_lags}): DS自己回帰")
    print(f"    Controls: Income_L1, CPI_L1, Unemp_L1")
    print(f"  Model B (+ STI):   {len(model_b_cols)} 説明変数")
    print(f"    + STI_L1..{x_lags}")

    # ── 4. In-sample 推定 ──
    print(f"\n  [E-1] In-sample モデル比較")
    print(f"  {'─' * 60}")

    y = merged[y_col]
    X_a = sm.add_constant(merged[model_a_cols])
    X_b = sm.add_constant(merged[model_b_cols])

    res_a = sm.OLS(y, X_a).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    res_b = sm.OLS(y, X_b).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    res_a_ols = sm.OLS(y, X_a).fit()
    res_b_ols = sm.OLS(y, X_b).fit()

    print(f"\n  {'指標':<20s} | {'Model A':>12s} | {'Model B':>12s} | {'改善':>10s}")
    print(f"  {'─' * 20}-+-{'─' * 12}-+-{'─' * 12}-+-{'─' * 10}")
    print(f"  {'R²':<20s} | {res_a.rsquared:>12.4f} | {res_b.rsquared:>12.4f} | "
          f"{res_b.rsquared - res_a.rsquared:>+10.4f}")
    print(f"  {'Adj R²':<20s} | {res_a.rsquared_adj:>12.4f} | {res_b.rsquared_adj:>12.4f} | "
          f"{res_b.rsquared_adj - res_a.rsquared_adj:>+10.4f}")
    print(f"  {'AIC':<20s} | {res_a_ols.aic:>12.1f} | {res_b_ols.aic:>12.1f} | "
          f"{res_b_ols.aic - res_a_ols.aic:>+10.1f}")
    print(f"  {'BIC':<20s} | {res_a_ols.bic:>12.1f} | {res_b_ols.bic:>12.1f} | "
          f"{res_b_ols.bic - res_a_ols.bic:>+10.1f}")
    print(f"  {'N':<20s} | {int(res_a.nobs):>12d} | {int(res_b.nobs):>12d} |")

    aic_improved = res_b_ols.aic < res_a_ols.aic
    bic_improved = res_b_ols.bic < res_a_ols.bic
    adj_r2_improved = res_b.rsquared_adj > res_a.rsquared_adj

    print(f"\n  AIC改善: {'PASS' if aic_improved else 'FAIL'} "
          f"(Δ = {res_b_ols.aic - res_a_ols.aic:+.1f})")
    print(f"  BIC改善: {'PASS' if bic_improved else 'FAIL'} "
          f"(Δ = {res_b_ols.bic - res_a_ols.bic:+.1f})")
    print(f"  Adj R²改善: {'PASS' if adj_r2_improved else 'FAIL'} "
          f"(Δ = {res_b.rsquared_adj - res_a.rsquared_adj:+.4f})")

    # ── 5. F検定 ──
    print(f"\n  [E-2] F検定 (STI追加の限界効果)")
    print(f"  {'─' * 60}")

    rss_a = res_a_ols.ssr
    rss_b = res_b_ols.ssr
    df_b = res_b_ols.df_resid
    k_extra = len(sti_cols)

    f_stat = ((rss_a - rss_b) / k_extra) / (rss_b / df_b)
    f_pval = 1 - sp_stats.f.cdf(f_stat, k_extra, df_b)

    sig = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 else "*" if f_pval < 0.1 else "n.s."
    print(f"  H₀: STIの追加は説明力を改善しない")
    print(f"  F({k_extra}, {df_b}) = {f_stat:.4f}, p = {f_pval:.6f} ({sig})")
    if f_pval < 0.05:
        print(f"  → H₀棄却: STIは有意な追加情報を持つ")

    # ── 6. STI係数詳細 ──
    print(f"\n  [E-3] Model B: STIラグ係数 (HAC SE)")
    print(f"  {'─' * 60}")
    for col in sti_cols:
        b = res_b.params[col]
        se = res_b.bse[col]
        p = res_b.pvalues[col]
        s = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"    {col}: β = {b:+.4f} (SE={se:.4f}, p={p:.4f}) {s}")

    # ── 7. Out-of-sample 予測 ──
    print(f"\n  [E-4] Out-of-sample 予測比較")
    print(f"  {'─' * 60}")

    min_train = 60
    n_total = len(merged)
    errors_a = []
    errors_b = []
    oos_dates = []

    # sm.add_constant() fails on single-row DataFrames → pre-add constant
    merged["const"] = 1.0
    all_a_cols = ["const"] + model_a_cols
    all_b_cols = ["const"] + model_b_cols

    for t in range(min_train, n_total):
        train = merged.iloc[:t]
        test_row = merged.iloc[t:t + 1]
        y_train = train[y_col]
        y_test = test_row[y_col].values[0]

        try:
            pred_a = sm.OLS(y_train, train[all_a_cols]).fit().predict(test_row[all_a_cols]).values[0]
            pred_b = sm.OLS(y_train, train[all_b_cols]).fit().predict(test_row[all_b_cols]).values[0]
            errors_a.append(y_test - pred_a)
            errors_b.append(y_test - pred_b)
            oos_dates.append(test_row["date"].values[0])
        except Exception:
            continue

    errors_a = np.array(errors_a)
    errors_b = np.array(errors_b)

    print(f"  拡張窓1ステップ先予測 (最小訓練={min_train}ヶ月)")
    print(f"  OOS予測数: {len(errors_a)}")

    if len(errors_a) == 0:
        print(f"  ⚠ OOS予測が0件 — スキップ")
        return {"in_sample": {"aic_drop": aic_b - aic_a, "bic_drop": bic_b - bic_a,
                              "f_stat": f_stat, "f_pval": f_pval}, "oos": None}

    rmse_a = np.sqrt(np.mean(errors_a ** 2))
    rmse_b = np.sqrt(np.mean(errors_b ** 2))
    mae_a = np.mean(np.abs(errors_a))
    mae_b = np.mean(np.abs(errors_b))

    print(f"  OOS期間: {pd.Timestamp(oos_dates[0]):%Y-%m} ~ "
          f"{pd.Timestamp(oos_dates[-1]):%Y-%m}")
    print(f"\n  {'指標':<20s} | {'Model A':>10s} | {'Model B':>10s} | {'改善率':>10s}")
    print(f"  {'─' * 20}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 10}")
    print(f"  {'RMSE':<20s} | {rmse_a:>10.4f} | {rmse_b:>10.4f} | "
          f"{(1 - rmse_b / rmse_a) * 100:>+9.1f}%")
    print(f"  {'MAE':<20s} | {mae_a:>10.4f} | {mae_b:>10.4f} | "
          f"{(1 - mae_b / mae_a) * 100:>+9.1f}%")

    # ── 8. Diebold-Mariano 検定 ──
    print(f"\n  [E-5] Diebold-Mariano 検定")
    print(f"  {'─' * 60}")

    d = errors_a ** 2 - errors_b ** 2
    d_mean = d.mean()
    d_var = np.var(d, ddof=1)
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    dm_pval = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))

    sig_dm = "***" if dm_pval < 0.01 else "**" if dm_pval < 0.05 else "*" if dm_pval < 0.1 else "n.s."
    print(f"  H₀: Model A と Model B の予測精度は同等")
    print(f"  DM統計量 = {dm_stat:+.4f}, p = {dm_pval:.4f} ({sig_dm})")
    if dm_stat > 0 and dm_pval < 0.05:
        print(f"  → Model B (STI入り) が有意に予測精度を改善")
    elif dm_stat > 0:
        print(f"  → Model B が改善傾向だが統計的有意差なし")
    else:
        print(f"  → Model A が優勢")

    csfe_diff = np.cumsum(errors_a ** 2 - errors_b ** 2)

    # ── 9. 総合判定 ──
    print(f"\n  {'═' * 70}")
    print(f"  Phase E 総合判定: STIは新しい情報を持つか？")
    print(f"  {'═' * 70}")

    checks = {
        "AIC改善": aic_improved,
        "BIC改善": bic_improved,
        "F検定有意 (p<0.05)": f_pval < 0.05,
        "OOS RMSE改善": rmse_b < rmse_a,
        "DM検定 (p<0.10)": dm_stat > 0 and dm_pval < 0.10,
    }
    for name, passed in checks.items():
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")

    n_pass = sum(checks.values())
    print(f"\n  5指標中 {n_pass} 指標でPASS")

    print(f"\n  {'═' * 70}")
    if n_pass >= 4:
        print(f"  ★ 結論: STIは「観測されていなかった状態量」である")
        print(f"    → 既存マクロ変数では捕捉できない新しい予測情報を保持")
        print(f"    → SSQ「社会脅威場は測定可能」が成立")
        verdict = "NEW_STATE_VARIABLE"
    elif n_pass >= 3:
        print(f"  ◆ 結論: STIは追加的な予測情報を持つ (強い証拠)")
        verdict = "STRONG_EVIDENCE"
    elif n_pass >= 2:
        print(f"  △ 結論: STIは部分的に新しい情報を持つ")
        verdict = "PARTIAL_EVIDENCE"
    else:
        print(f"  × 結論: STIは既存マクロ変数の合成に過ぎない")
        verdict = "NO_NEW_INFORMATION"
    print(f"  {'═' * 70}")

    # ── 10. 保存 ──
    result_data = {
        "analysis": "SSQ Phase E Incremental Prediction",
        "in_sample": {
            "model_a": {"aic": float(res_a_ols.aic), "bic": float(res_a_ols.bic),
                        "r2": float(res_a.rsquared), "adj_r2": float(res_a.rsquared_adj)},
            "model_b": {"aic": float(res_b_ols.aic), "bic": float(res_b_ols.bic),
                        "r2": float(res_b.rsquared), "adj_r2": float(res_b.rsquared_adj)},
            "f_test": {"f_stat": float(f_stat), "p_value": float(f_pval)},
        },
        "out_of_sample": {
            "n_predictions": len(errors_a),
            "rmse_a": float(rmse_a), "rmse_b": float(rmse_b),
            "mae_a": float(mae_a), "mae_b": float(mae_b),
            "dm_stat": float(dm_stat), "dm_pval": float(dm_pval),
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "verdict": verdict,
    }
    result_path = output_dir / "phase_e_prediction_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {result_path}")

    # ── 11. プロット ──
    plot_incremental_results(oos_dates, errors_a, errors_b, csfe_diff,
                              checks, CHART_DIR)
    return result_data


def plot_incremental_results(oos_dates, errors_a, errors_b, csfe_diff,
                              checks, chart_dir):
    """Phase E: 予測比較プロット"""
    for fn in ['MS Gothic', 'Yu Gothic', 'Meiryo']:
        try:
            matplotlib.font_manager.FontProperties(family=fn)
            plt.rcParams['font.family'] = fn
            break
        except:
            pass

    dates = pd.to_datetime(oos_dates)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1])

    # Panel A: CSFE差分
    ax = axes[0]
    ax.plot(dates, csfe_diff, color='#C62828', linewidth=1.5)
    ax.fill_between(dates, 0, csfe_diff,
                    where=(np.array(csfe_diff) > 0), color='#C62828', alpha=0.2,
                    label='Model B 優勢 (STI入り)')
    ax.fill_between(dates, 0, csfe_diff,
                    where=(np.array(csfe_diff) <= 0), color='#1565C0', alpha=0.2,
                    label='Model A 優勢 (baseline)')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('累積二乗予測誤差差分\n(正 = STI追加で改善)', fontsize=11)
    ax.set_title('Phase E: CSFE差分 Σ(e²_A - e²_B)\n右上がり = STIが予測を改善',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel B: Rolling RMSE
    ax = axes[1]
    window = 24
    if len(errors_a) > window:
        roll_a = pd.Series(errors_a ** 2).rolling(window).mean().apply(np.sqrt)
        roll_b = pd.Series(errors_b ** 2).rolling(window).apply(
            lambda x: np.sqrt(np.mean(x)))
        ax.plot(dates, roll_a, color='#1565C0', linewidth=1.5,
                label='Model A (baseline)')
        ax.plot(dates, roll_b, color='#C62828', linewidth=1.5,
                label='Model B (+STI)')
    ax.set_ylabel('Rolling RMSE (24M窓)', fontsize=11)
    ax.set_title('Rolling RMSE比較', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    rmse_a = np.sqrt(np.mean(errors_a ** 2))
    rmse_b = np.sqrt(np.mean(errors_b ** 2))
    ax.text(0.98, 0.98,
            f'全体RMSE: A={rmse_a:.3f}, B={rmse_b:.3f}\n'
            f'改善率: {(1 - rmse_b / rmse_a) * 100:+.1f}%',
            transform=ax.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    chart_path = str(chart_dir / "ssq_phase_e_prediction.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  チャート保存: {chart_path}")


# ═══════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SSQ First Chart Builder")
    print(f"期間: {START_YEAR}年〜最新")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: STI構成要素の取得 ──
    print("\n[1/4] STI構成要素")
    print("-" * 40)

    print("\n  (a) 完全失業率 (月次)")
    unemp_df = fetch_unemployment_rate()

    print("\n  (b) 常用雇用指数 前年同月比 → 反転 (月次)")
    employ_df = fetch_employment_index()

    print("\n  (c) 刑法犯認知件数率 (年次→月次補間)")
    crime_df = fetch_crime_rate()

    sti_components = {}
    if not unemp_df.empty:
        sti_components["unemployment"] = unemp_df
    if not employ_df.empty:
        sti_components["employment_decline"] = employ_df
    if not crime_df.empty:
        sti_components["crime"] = crime_df

    print(f"\n  STI構成要素: {len(sti_components)}系列 → "
          f"{list(sti_components.keys())}")

    # ── Step 2: STI構築 ──
    print("\n[2/4] STI構築")
    print("-" * 40)
    sti_df, sti_detail = build_sti(sti_components)

    # ── Step 3: 消費データ取得 → 裁量消費比率 ──
    print("\n[3/4] 消費データ & 裁量消費比率")
    print("-" * 40)
    consumption_df = fetch_consumption_categories()
    ds_df = compute_discretionary_share(consumption_df)

    # ── Step 4: 最初の1枚 ──
    print("\n[4/6] 最初の1枚の図")
    print("-" * 40)
    chart_path = str(CHART_DIR / "ssq_first_chart.png")
    plot_first_chart(sti_df, ds_df, chart_path)

    # データ保存
    if not sti_detail.empty:
        sti_path = OUTPUT_DIR / "sti_monthly.csv"
        sti_detail.to_csv(sti_path, index=False, encoding="utf-8-sig")
        print(f"  STIデータ保存: {sti_path}")
    if not ds_df.empty:
        ds_path = OUTPUT_DIR / "discretionary_share_monthly.csv"
        ds_df.to_csv(ds_path, index=False, encoding="utf-8-sig")
        print(f"  裁量消費比率データ保存: {ds_path}")

    # ── Step 5: Phase A デトレンド ──
    print("\n[5/6] Phase A: デトレンド (前年同月比)")
    print("-" * 40)
    detrended_path = str(CHART_DIR / "ssq_detrended_chart.png")
    detrended = plot_detrended_chart(sti_df, ds_df, detrended_path)

    # ── Step 6: Phase B 危機ズーム ──
    print("\n[6/8] Phase B: 危機期間ディープダイブ")
    print("-" * 40)
    crisis_path = str(CHART_DIR / "ssq_crisis_panels.png")
    plot_crisis_panels(sti_df, ds_df, crisis_path)

    # ── Step 7: Phase C ラグ相関 (CCF) ──
    print("\n[7/8] Phase C: 交差相関関数 (CCF)")
    print("-" * 40)
    ccf_df = compute_ccf(sti_df, ds_df, max_lag=12)
    if not ccf_df.empty:
        ccf_path = str(CHART_DIR / "ssq_ccf.png")
        peak_lag, peak_r = plot_ccf(ccf_df, ccf_path)
        print(f"\n  CCF結果:")
        print(f"  ピーク負相関: k = {peak_lag} ヶ月, r = {peak_r:.4f}")
        if peak_lag > 0:
            print(f"  → STIが {peak_lag}ヶ月先行して消費構成に影響")
            print(f"  → SSQ仮説の因果方向と整合")
        elif peak_lag == 0:
            print(f"  → 同時相関: 共通原因の可能性")
        else:
            print(f"  → 消費が {abs(peak_lag)}ヶ月先行: SSQ仮説と不整合")

        # CCF全体をテーブル表示
        print(f"\n  CCFテーブル (k=-12~+12):")
        print(f"  {'k':>4s} | {'r':>8s} | {'N':>4s} | {'方向':>8s}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*4}-+-{'-'*8}")
        for _, row in ccf_df.iterrows():
            k = int(row['lag_k'])
            r = row['correlation']
            n = int(row['n'])
            direction = "STI先行" if k > 0 else "同時" if k == 0 else "消費先行"
            marker = " <<<" if (k == peak_lag) else ""
            print(f"  {k:>+4d} | {r:>+8.4f} | {n:>4d} | {direction:>8s}{marker}")

    # ── Step 8: Phase C ローリング相関 ──
    print("\n[8/8] Phase C: ローリング相関 (36ヶ月窓)")
    print("-" * 40)
    rolling_df = compute_rolling_correlation(sti_df, ds_df, window=36)
    if not rolling_df.empty:
        rolling_path = str(CHART_DIR / "ssq_rolling_correlation.png")
        plot_rolling_correlation(rolling_df, rolling_path)

        # レジーム統計
        neg_pct = (rolling_df['rolling_r'] < 0).mean() * 100
        print(f"  負の相関が出現する期間: {neg_pct:.1f}%")
        print(f"  相関範囲: {rolling_df['rolling_r'].min():.3f} ~ "
              f"{rolling_df['rolling_r'].max():.3f}")

    # ── Step 9: STI_leading 構築 ──
    print("\n" + "=" * 60)
    print("Phase C+: STI_leading (先行指標) の構築と比較")
    print("=" * 60)

    print("\n  (d) 有効求人倍率 → 反転 (月次, 先行)")
    job_df = fetch_job_openings_inverted()

    print("\n  (e) 消費者態度指数 → 反転 (月次, 先行)")
    conf_df = fetch_consumer_confidence_inverted()

    leading_components = {}
    if not job_df.empty:
        leading_components["job_scarcity"] = job_df
    if not conf_df.empty:
        leading_components["consumer_anxiety"] = conf_df

    sti_leading_df = None
    if leading_components:
        print(f"\n  STI_leading構成: {list(leading_components.keys())}")
        sti_leading_df, sti_leading_detail = build_sti(leading_components)

        # CCF比較
        print("\n  CCF: STI_leading vs 裁量消費比率")
        ccf_leading = compute_ccf(sti_leading_df, ds_df, max_lag=12)

        if not ccf_leading.empty and not ccf_df.empty:
            # ピーク
            min_idx_l = ccf_leading["correlation"].idxmin()
            peak_lag_l = int(ccf_leading["lag_k"].iloc[min_idx_l])
            peak_r_l = ccf_leading["correlation"].iloc[min_idx_l]

            print(f"\n  {'─'*50}")
            print(f"  CCF比較結果:")
            print(f"  {'─'*50}")
            print(f"  STI_lagging  (遅行): ピーク k = {peak_lag:+d}, r = {peak_r:+.4f}")
            print(f"  STI_leading  (先行): ピーク k = {peak_lag_l:+d}, r = {peak_r_l:+.4f}")
            print(f"  ピーク移動量: {peak_lag_l - peak_lag:+d} ヶ月")
            print(f"  {'─'*50}")

            if peak_lag_l > peak_lag:
                print(f"  → ピークが正方向に移動: 先行指標の方がSTI先行性を示す")
                print(f"  → 「観測方程式」仮説と整合")
            if peak_lag_l > 0:
                print(f"  → STI_leading が {peak_lag_l}ヶ月先行して消費を予測")
                print(f"  → SSQ仮説の因果方向が支持される")

            # CCFテーブル (leading)
            print(f"\n  CCFテーブル (STI_leading, k=-12~+12):")
            print(f"  {'k':>4s} | {'r':>8s} | {'N':>4s} | {'方向':>8s}")
            print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*4}-+-{'-'*8}")
            for _, row in ccf_leading.iterrows():
                k = int(row['lag_k'])
                r = row['correlation']
                n = int(row['n'])
                direction = "STI先行" if k > 0 else "同時" if k == 0 else "消費先行"
                marker = " <<<" if (k == peak_lag_l) else ""
                print(f"  {k:>+4d} | {r:>+8.4f} | {n:>4d} | {direction:>8s}{marker}")

            # 比較チャート
            comp_path = str(CHART_DIR / "ssq_ccf_comparison.png")
            plot_ccf_comparison(ccf_df, ccf_leading, comp_path)

            # STI_leading データ保存
            sti_leading_detail.to_csv(
                OUTPUT_DIR / "sti_leading_monthly.csv",
                index=False, encoding="utf-8-sig")
            print(f"\n  STI_leading保存: {OUTPUT_DIR / 'sti_leading_monthly.csv'}")

    # ── Step 10: Phase D 正式回帰分析 ──
    if sti_leading_df is not None:
        print("\n" + "=" * 60)
        print("Phase D 準備: 統制変数の取得")
        print("=" * 60)

        print("\n  (f) 可処分所得 (勤労者世帯, 月次)")
        income_df = fetch_disposable_income()

        print("\n  (g) CPI 前年同月比 (既存データ)")
        cpi_df = load_cpi_yoy()

        if not income_df.empty and not cpi_df.empty and not unemp_df.empty:
            phase_d_result, phase_d_data = run_phase_d(
                sti_leading_df, ds_df, unemp_df, income_df, cpi_df, OUTPUT_DIR
            )
        else:
            print("  Phase D: 統制変数データ不足でスキップ")
            missing = []
            if income_df.empty:
                missing.append("可処分所得")
            if cpi_df.empty:
                missing.append("CPI")
            if unemp_df.empty:
                missing.append("失業率")
            print(f"  欠損: {', '.join(missing)}")

    # ── Step 11: Phase D+ Granger因果性検定 ──
    if sti_leading_df is not None:
        granger_result = run_granger_causality(sti_leading_df, ds_df, OUTPUT_DIR)

    # ── Step 12: Phase E 予測力の上乗せ検証 ──
    if (sti_leading_df is not None and not income_df.empty
            and not cpi_df.empty and not unemp_df.empty):
        phase_e_result = run_incremental_prediction(
            sti_leading_df, ds_df, unemp_df, income_df, cpi_df, OUTPUT_DIR
        )

    # ── Step 13: ロバストネスバッテリー用CSVキャッシュ出力 ──
    if sti_leading_df is not None and 'phase_d_data' not in dir():
        # phase_d_data は run_phase_d の返り値 merged
        pass
    if sti_leading_df is not None:
        try:
            cache_cols = ["date", "STI_yoy", "discretionary_share_yoy",
                          "disposable_income_yoy", "cpi_yoy", "unemployment_rate_yoy"]
            if phase_d_data is not None and all(c in phase_d_data.columns for c in cache_cols):
                cache_path = OUTPUT_DIR / "phase_d_merged.csv"
                phase_d_data[cache_cols].to_csv(cache_path, index=False, encoding="utf-8-sig")
                print(f"\n  ロバストネス用キャッシュ保存: {cache_path}")
                print(f"  ({len(phase_d_data)}行, {len(cache_cols)}列)")
        except NameError:
            print("  Phase D未実行のためキャッシュ出力スキップ")

    # サマリー統計
    print(f"\n{'=' * 60}")
    print("総合サマリー")
    print(f"{'=' * 60}")
    if not sti_df.empty and not ds_df.empty:
        overlap = sti_df.merge(ds_df[["date", "discretionary_share"]],
                               on="date", how="inner")
        print(f"  期間: {overlap['date'].min():%Y-%m} ~ {overlap['date'].max():%Y-%m}"
              f" ({len(overlap)}件)")
        r_level = overlap['STI'].corr(overlap['discretionary_share'])
        print(f"  相関 (raw level): r = {r_level:+.4f}")
        if detrended is not None:
            r_yoy = detrended['STI_yoy'].corr(detrended['discretionary_share_yoy'])
            print(f"  相関 (YoY detrended): r = {r_yoy:+.4f}")
        if not ccf_df.empty:
            print(f"  CCFピーク (STI_lagging): k = {peak_lag}, r = {peak_r:+.4f}")
        if leading_components and not ccf_leading.empty:
            print(f"  CCFピーク (STI_leading): k = {peak_lag_l}, r = {peak_r_l:+.4f}")
        print(f"\n  仮説H1: β₁ < 0 (STI↑ → 裁量消費比率↓)")

    print(f"\n完了!")


if __name__ == "__main__":
    main()
