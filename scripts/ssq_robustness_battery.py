#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSQ Robustness Battery
======================
SSQ論文 Phase 1: 査読耐性を上げるためのロバストネス検証バッテリー

7つのサブテスト:
  1. 定常性バッテリー (ADF/KPSS/PP/ZA + 共和分)
  2. Toda-Yamamoto Granger因果検定
  3. 残差診断 (BP/White/ARCH-LM/LB/JB)
  4. 構造変化検定 (Chow/Bai-Perron/CUSUM)
  5. Clark-West検定
  6. プラセボテスト (シャッフル/ランダム/非裁量/未来リーク)
  7. 代替STI仕様テスト

前提: build_ssq_first_chart.py を事前に実行し、
  data/ssq/phase_d_merged.csv が存在すること
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_ljungbox, acorr_breusch_godfrey,
)
from statsmodels.stats.stattools import jarque_bera
from scipy import stats as sp_stats

from arch.unitroot import PhillipsPerron, ZivotAndrews
import ruptures

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ssq"
CHART_DIR = BASE_DIR / "output" / "ssq"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# ユーティリティ
# ═══════════════════════════════════════════════════════════

def compute_yoy(df, date_col, value_col):
    """12ヶ月差分（build_ssq_first_chart.py と同一ロジック）"""
    df = df.sort_values(date_col).copy()
    df[f"{value_col}_yoy"] = df[value_col].diff(12)
    return df.dropna(subset=[f"{value_col}_yoy"])


def save_json(data, path):
    """JSON保存ユーティリティ"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  保存: {path}")


def refit_m3(merged):
    """M3フルモデルを再推定（残差取得用）"""
    season_cols = [f"m{m}" for m in range(2, 13)]
    for m in range(2, 13):
        col = f"m{m}"
        if col not in merged.columns:
            merged[col] = (merged["date"].dt.month == m).astype(float)

    y = merged["discretionary_share_yoy"]
    x_cols = ["STI_yoy", "disposable_income_yoy", "cpi_yoy",
              "unemployment_rate_yoy"] + season_cols
    X = sm.add_constant(merged[x_cols])
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    return res, X, y


# ═══════════════════════════════════════════════════════════
# データローダー
# ═══════════════════════════════════════════════════════════

def load_data():
    """全データをロード"""
    print("=" * 60)
    print("データロード")
    print("=" * 60)

    # Phase D merged (YoY変換済み)
    merged_path = DATA_DIR / "phase_d_merged.csv"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"{merged_path} が見つかりません。\n"
            "先に build_ssq_first_chart.py を実行してください。"
        )
    merged = pd.read_csv(merged_path, parse_dates=["date"])
    print(f"  Phase D merged: {len(merged)}行 ({merged['date'].min():%Y-%m} ~ {merged['date'].max():%Y-%m})")

    # STI level (個別成分含む)
    sti_df = pd.read_csv(DATA_DIR / "sti_leading_monthly.csv", parse_dates=["date"])
    print(f"  STI leading: {len(sti_df)}行")

    # DS level
    ds_df = pd.read_csv(DATA_DIR / "discretionary_share_monthly.csv", parse_dates=["date"])
    print(f"  DS: {len(ds_df)}行")

    # 既存結果
    with open(DATA_DIR / "phase_d_regression_results.json", "r", encoding="utf-8") as f:
        phase_d_results = json.load(f)
    with open(DATA_DIR / "phase_e_prediction_results.json", "r", encoding="utf-8") as f:
        phase_e_results = json.load(f)

    return {
        "merged": merged,
        "sti_df": sti_df,
        "ds_df": ds_df,
        "phase_d_results": phase_d_results,
        "phase_e_results": phase_e_results,
    }


# ═══════════════════════════════════════════════════════════
# Step 2: 定常性バッテリー (§4.1)
# ═══════════════════════════════════════════════════════════

def _run_unit_root_tests(series, name):
    """1つの系列に対して全定常性テストを実行"""
    results = {}
    arr = series.dropna().values

    # ADF (3 autolag variants)
    for autolag in ["AIC", "BIC", "t-stat"]:
        try:
            stat, pval, used_lag, nobs, crit, icbest = adfuller(arr, autolag=autolag)
            results[f"ADF_{autolag}"] = {
                "statistic": float(stat), "p_value": float(pval),
                "used_lag": int(used_lag), "nobs": int(nobs),
                "reject_5pct": pval < 0.05,
            }
        except Exception as e:
            results[f"ADF_{autolag}"] = {"error": str(e)}

    # KPSS (null = stationary)
    try:
        stat, pval, used_lag, crit = kpss(arr, regression="c", nlags="auto")
        results["KPSS"] = {
            "statistic": float(stat), "p_value": float(pval),
            "used_lag": int(used_lag),
            "fail_to_reject_5pct": pval > 0.05,  # True = supports stationarity
        }
    except Exception as e:
        results["KPSS"] = {"error": str(e)}

    # Phillips-Perron
    try:
        pp = PhillipsPerron(arr)
        results["Phillips_Perron"] = {
            "statistic": float(pp.stat), "p_value": float(pp.pvalue),
            "used_lag": int(pp.lags),
            "reject_5pct": pp.pvalue < 0.05,
        }
    except Exception as e:
        results["Phillips_Perron"] = {"error": str(e)}

    # Zivot-Andrews (structural break unit root)
    try:
        za = ZivotAndrews(arr)
        results["Zivot_Andrews"] = {
            "statistic": float(za.stat), "p_value": float(za.pvalue),
            "breakpoint_index": int(za.breakpoint) if hasattr(za, 'breakpoint') else None,
            "reject_5pct": za.pvalue < 0.05,
        }
    except Exception as e:
        results["Zivot_Andrews"] = {"error": str(e)}

    # Summary judgment
    adf_reject_count = sum(
        1 for k in ["ADF_AIC", "ADF_BIC", "ADF_t-stat"]
        if results.get(k, {}).get("reject_5pct", False)
    )
    kpss_supports = results.get("KPSS", {}).get("fail_to_reject_5pct", False)
    pp_rejects = results.get("Phillips_Perron", {}).get("reject_5pct", False)

    # I(0) if: (ADF majority rejects) AND (KPSS does not reject)
    i0_evidence = adf_reject_count >= 2 and kpss_supports
    results["_summary"] = {
        "adf_reject_count": adf_reject_count,
        "kpss_supports_stationarity": kpss_supports,
        "pp_rejects_unit_root": pp_rejects,
        "judgment": "I(0)" if i0_evidence else "ambiguous_or_I(1)",
    }

    print(f"\n  {name}:")
    print(f"    ADF棄却数: {adf_reject_count}/3, KPSS定常支持: {kpss_supports}, PP棄却: {pp_rejects}")
    print(f"    判定: {results['_summary']['judgment']}")

    return results


def run_stationarity_battery(data):
    """定常性テストバッテリー"""
    print("\n" + "=" * 60)
    print("§4.1 定常性バッテリー")
    print("=" * 60)

    merged = data["merged"]
    sti_yoy = merged["STI_yoy"].dropna()
    ds_yoy = merged["discretionary_share_yoy"].dropna()

    results = {
        "STI_yoy": _run_unit_root_tests(sti_yoy, "STI_yoy"),
        "DS_yoy": _run_unit_root_tests(ds_yoy, "DS_yoy"),
    }

    # If STI_yoy is I(1) or ambiguous → run cointegration
    sti_judgment = results["STI_yoy"]["_summary"]["judgment"]
    if sti_judgment != "I(0)":
        print("\n  STI_yoyがI(0)未確定 → 共和分検定を実施")
        results["cointegration"] = _run_cointegration(data)
    else:
        results["cointegration"] = {"skipped": True, "reason": "STI_yoy is I(0)"}

    save_json(results, DATA_DIR / "robustness_stationarity.json")
    return results


def _run_cointegration(data):
    """Engle-Granger + Johansen共和分検定"""
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    sti_df = data["sti_df"]
    ds_df = data["ds_df"]

    # Level series (compute YoY for alignment, then use levels)
    sti_yoy = compute_yoy(sti_df.copy(), "date", "STI")
    ds_yoy = compute_yoy(ds_df.copy(), "date", "discretionary_share")
    m = sti_yoy[["date", "STI"]].merge(ds_yoy[["date", "discretionary_share"]], on="date").dropna()

    results = {}

    # Engle-Granger
    try:
        stat, pval, crit = coint(m["STI"].values, m["discretionary_share"].values)
        results["engle_granger"] = {
            "statistic": float(stat), "p_value": float(pval),
            "critical_values": {k: float(v) for k, v in zip(["1%", "5%", "10%"], crit)},
            "cointegrated_5pct": pval < 0.05,
        }
        print(f"    Engle-Granger: stat={stat:.3f}, p={pval:.4f}, 共和分={'あり' if pval < 0.05 else 'なし'}")
    except Exception as e:
        results["engle_granger"] = {"error": str(e)}

    # Johansen
    try:
        joh = coint_johansen(m[["STI", "discretionary_share"]].values, det_order=0, k_ar_diff=4)
        trace_stats = joh.lr1.tolist()
        trace_crit_5pct = joh.cvt[:, 1].tolist()
        results["johansen"] = {
            "trace_statistics": trace_stats,
            "trace_critical_5pct": trace_crit_5pct,
            "n_cointegrating_relations": int(sum(
                1 for t, c in zip(trace_stats, trace_crit_5pct) if t > c
            )),
        }
        print(f"    Johansen: trace={trace_stats[0]:.3f} vs cv5%={trace_crit_5pct[0]:.3f}")
    except Exception as e:
        results["johansen"] = {"error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════
# Step 3: Toda-Yamamoto Granger (§4.2)
# ═══════════════════════════════════════════════════════════

def run_toda_yamamoto(data, stationarity_results):
    """Toda-Yamamoto法によるGranger因果検定"""
    print("\n" + "=" * 60)
    print("§4.2 Toda-Yamamoto Granger因果検定")
    print("=" * 60)

    merged = data["merged"]
    sti = merged["STI_yoy"].values
    ds = merged["discretionary_share_yoy"].values
    endog = np.column_stack([sti, ds])

    # d_max: 最大積分次数
    sti_j = stationarity_results.get("STI_yoy", {}).get("_summary", {}).get("judgment", "I(0)")
    ds_j = stationarity_results.get("DS_yoy", {}).get("_summary", {}).get("judgment", "I(0)")
    d_max = 1 if ("I(1)" in sti_j or "ambiguous" in sti_j or "I(1)" in ds_j or "ambiguous" in ds_j) else 0
    # Conservative: always use d_max=1 for safety
    d_max = max(d_max, 1)
    print(f"  d_max = {d_max}")

    # Optimal lag selection via VAR
    var_model = VAR(endog)
    lag_results = {}
    for ic in ["aic", "bic", "hqic"]:
        try:
            selected = var_model.select_order(maxlags=15)
            lag_results[ic] = int(getattr(selected, ic))
        except Exception:
            lag_results[ic] = 6  # fallback
    p_opt = lag_results.get("aic", 6)
    print(f"  最適ラグ: AIC={lag_results.get('aic')}, BIC={lag_results.get('bic')}, HQIC={lag_results.get('hqic')}")
    print(f"  使用ラグ p = {p_opt}")

    # Fit VAR(p + d_max)
    p_aug = p_opt + d_max
    var_fit = var_model.fit(p_aug)

    results = {
        "d_max": d_max,
        "p_optimal": p_opt,
        "p_augmented": p_aug,
        "lag_selection": lag_results,
        "directions": {},
    }

    # Wald test for each direction
    for direction, col_cause, col_effect, label in [
        ("STI_to_DS", 0, 1, "STI→DS"),
        ("DS_to_STI", 1, 0, "DS→STI"),
    ]:
        # The coefficient matrix: var_fit.coefs shape = (p_aug, n_vars, n_vars)
        # For equation of col_effect, test that lags 1..p_opt of col_cause are jointly zero
        # (lags p_opt+1..p_aug are left unrestricted)

        # Use the params from the fitted model
        coefs = var_fit.coefs  # shape: (p_aug, 2, 2)
        # Coefficients of col_cause in equation for col_effect, lags 1..p_opt
        restricted_coefs = np.array([coefs[lag][col_effect, col_cause] for lag in range(p_opt)])

        # Get the variance-covariance of the restricted parameters
        # var_fit.params is (1+2*p_aug, 2) — intercept + lagged vars for each equation
        # We need to extract the relevant covariance submatrix

        # Alternative: manual Wald test using OLS on the effect equation
        T = len(endog)
        # Build lagged matrix
        max_lag = p_aug
        Y_effect = endog[max_lag:, col_effect]
        X_list = [np.ones(T - max_lag)]  # intercept
        for lag in range(1, max_lag + 1):
            X_list.append(endog[max_lag - lag:T - lag, 0])  # STI lags
            X_list.append(endog[max_lag - lag:T - lag, 1])  # DS lags
        X = np.column_stack(X_list)

        # OLS fit
        ols_res = sm.OLS(Y_effect, X).fit()

        # Indices of col_cause coefficients for lags 1..p_opt
        # Layout: const, STI_L1, DS_L1, STI_L2, DS_L2, ...
        if col_cause == 0:  # STI
            restrict_indices = [1 + 2 * (lag - 1) for lag in range(1, p_opt + 1)]
        else:  # DS
            restrict_indices = [2 + 2 * (lag - 1) for lag in range(1, p_opt + 1)]

        R = np.zeros((p_opt, X.shape[1]))
        for i, idx in enumerate(restrict_indices):
            R[i, idx] = 1.0

        # Wald statistic: (Rβ)' (R V R')^{-1} (Rβ) where V = Cov(β)
        beta = ols_res.params
        V = ols_res.cov_params()
        Rb = R @ beta
        RVR = R @ V @ R.T
        try:
            wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
            p_value = float(1 - sp_stats.chi2.cdf(wald_stat, df=p_opt))
        except np.linalg.LinAlgError:
            wald_stat = float("nan")
            p_value = float("nan")

        results["directions"][direction] = {
            "wald_statistic": wald_stat,
            "df": p_opt,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05 if not np.isnan(p_value) else False,
        }
        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
        print(f"  {label}: Wald={wald_stat:.3f}, df={p_opt}, p={p_value:.4f} {sig}")

    # Lag-by-lag analysis (standard F-test for comparison)
    print("\n  ラグ別Granger F検定（参考: 標準法）:")
    for p_test in [2, 4, 6, 8, 10, 12]:
        if p_test > len(endog) // 3:
            break
        try:
            gc = grangercausalitytests(
                np.column_stack([ds, sti]), maxlag=p_test, verbose=False)
            f_stat = gc[p_test][0]['ssr_ftest'][0]
            f_pval = gc[p_test][0]['ssr_ftest'][1]
            print(f"    lag={p_test:2d}: STI→DS F={f_stat:.3f} p={f_pval:.4f}")
        except Exception:
            pass

    save_json(results, DATA_DIR / "robustness_toda_yamamoto.json")
    return results


# ═══════════════════════════════════════════════════════════
# Step 4: 残差診断 (§4.4)
# ═══════════════════════════════════════════════════════════

def run_residual_diagnostics(data):
    """M3フルモデルの残差診断"""
    print("\n" + "=" * 60)
    print("§4.4 残差診断")
    print("=" * 60)

    merged = data["merged"].copy()
    res, X, y = refit_m3(merged)
    residuals = res.resid
    fitted = res.fittedvalues

    results = {}

    # Breusch-Pagan
    try:
        bp_stat, bp_pval, bp_fstat, bp_fpval = het_breuschpagan(residuals, X)
        results["breusch_pagan"] = {
            "lm_stat": float(bp_stat), "lm_pval": float(bp_pval),
            "f_stat": float(bp_fstat), "f_pval": float(bp_fpval),
            "heteroscedastic_5pct": bp_pval < 0.05,
        }
        print(f"  Breusch-Pagan: LM={bp_stat:.3f}, p={bp_pval:.4f}")
    except Exception as e:
        results["breusch_pagan"] = {"error": str(e)}

    # White test
    try:
        w_stat, w_pval, w_fstat, w_fpval = het_white(residuals, X)
        results["white"] = {
            "lm_stat": float(w_stat), "lm_pval": float(w_pval),
            "f_stat": float(w_fstat), "f_pval": float(w_fpval),
            "heteroscedastic_5pct": w_pval < 0.05,
        }
        print(f"  White: LM={w_stat:.3f}, p={w_pval:.4f}")
    except Exception as e:
        results["white"] = {"error": str(e)}

    # ARCH-LM test
    try:
        from statsmodels.stats.diagnostic import het_arch
        arch_stat, arch_pval, arch_fstat, arch_fpval = het_arch(residuals, nlags=4)
        results["arch_lm"] = {
            "lm_stat": float(arch_stat), "lm_pval": float(arch_pval),
            "f_stat": float(arch_fstat), "f_pval": float(arch_fpval),
            "arch_effects_5pct": arch_pval < 0.05,
        }
        print(f"  ARCH-LM(4): LM={arch_stat:.3f}, p={arch_pval:.4f}")
    except Exception as e:
        results["arch_lm"] = {"error": str(e)}

    # Ljung-Box
    try:
        lb = acorr_ljungbox(residuals, lags=[6, 12, 18], return_df=True)
        results["ljung_box"] = {}
        for lag in lb.index:
            results["ljung_box"][f"lag_{lag}"] = {
                "statistic": float(lb.loc[lag, "lb_stat"]),
                "p_value": float(lb.loc[lag, "lb_pvalue"]),
                "serial_correlation_5pct": lb.loc[lag, "lb_pvalue"] < 0.05,
            }
        print(f"  Ljung-Box(12): Q={lb.loc[12, 'lb_stat']:.3f}, p={lb.loc[12, 'lb_pvalue']:.4f}")
    except Exception as e:
        results["ljung_box"] = {"error": str(e)}

    # Breusch-Godfrey
    try:
        bg_stat, bg_pval, bg_fstat, bg_fpval = acorr_breusch_godfrey(res, nlags=12)
        results["breusch_godfrey"] = {
            "lm_stat": float(bg_stat), "lm_pval": float(bg_pval),
            "f_stat": float(bg_fstat), "f_pval": float(bg_fpval),
            "serial_correlation_5pct": bg_pval < 0.05,
        }
        print(f"  Breusch-Godfrey(12): LM={bg_stat:.3f}, p={bg_pval:.4f}")
    except Exception as e:
        results["breusch_godfrey"] = {"error": str(e)}

    # Jarque-Bera
    try:
        jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)
        results["jarque_bera"] = {
            "statistic": float(jb_stat), "p_value": float(jb_pval),
            "skewness": float(skew), "kurtosis": float(kurtosis),
            "non_normal_5pct": jb_pval < 0.05,
        }
        print(f"  Jarque-Bera: JB={jb_stat:.3f}, p={jb_pval:.4f}, skew={skew:.3f}, kurt={kurtosis:.3f}")
    except Exception as e:
        results["jarque_bera"] = {"error": str(e)}

    # ── Diagnostic plots ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.5, s=15)
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # Q-Q plot
    ax = axes[0, 1]
    sm.qqplot(residuals, line='45', ax=ax, alpha=0.5, markersize=4)
    ax.set_title("Normal Q-Q Plot")

    # Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=25, density=True, alpha=0.7, edgecolor='black')
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_range, sp_stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
            'r-', linewidth=2)
    ax.set_xlabel("Residuals")
    ax.set_title("Residual Distribution")

    # ACF of residuals
    ax = axes[1, 1]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=ax, alpha=0.05)
    ax.set_title("Residual ACF")

    plt.suptitle("SSQ M3 Residual Diagnostics", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = CHART_DIR / "robustness_residual_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  プロット保存: {plot_path}")

    save_json(results, DATA_DIR / "robustness_residual_diagnostics.json")
    return results


# ═══════════════════════════════════════════════════════════
# Step 5: 構造変化検定 (§4.3)
# ═══════════════════════════════════════════════════════════

def run_structural_break_tests(data):
    """構造変化検定"""
    print("\n" + "=" * 60)
    print("§4.3 構造変化検定")
    print("=" * 60)

    merged = data["merged"].copy()
    res, X, y = refit_m3(merged)
    residuals = res.resid.values
    dates = merged["date"].values

    results = {}

    # ── Chow test at 2018-01 ──
    print("\n  Chow検定 (break = 2018-01):")
    break_date = pd.Timestamp("2018-01-01")
    mask_pre = merged["date"] < break_date
    mask_post = merged["date"] >= break_date
    n1, n2, k = mask_pre.sum(), mask_post.sum(), X.shape[1]

    if n1 > k and n2 > k:
        res_full = sm.OLS(y, X).fit()
        res_pre = sm.OLS(y[mask_pre], X[mask_pre]).fit()
        res_post = sm.OLS(y[mask_post], X[mask_post]).fit()

        ssr_full = res_full.ssr
        ssr_pre = res_pre.ssr
        ssr_post = res_post.ssr

        chow_f = ((ssr_full - ssr_pre - ssr_post) / k) / ((ssr_pre + ssr_post) / (n1 + n2 - 2 * k))
        chow_p = 1 - sp_stats.f.cdf(chow_f, k, n1 + n2 - 2 * k)
        results["chow"] = {
            "break_date": "2018-01",
            "f_statistic": float(chow_f), "p_value": float(chow_p),
            "structural_break_5pct": chow_p < 0.05,
            "n_pre": int(n1), "n_post": int(n2),
        }
        print(f"    F={chow_f:.3f}, p={chow_p:.4f}, break={'あり' if chow_p < 0.05 else 'なし'}")
    else:
        results["chow"] = {"error": "Insufficient observations in sub-samples"}

    # ── Bai-Perron (ruptures) ──
    print("\n  Bai-Perron ブレーク検出:")
    try:
        # Use residuals from full model for break detection
        algo = ruptures.Binseg(model="l2", min_size=24).fit(residuals)

        # Test 1-3 breakpoints with BIC penalty
        n = len(residuals)
        bp_results = {}
        for n_bkps in [1, 2, 3]:
            bkps = algo.predict(n_bkps=n_bkps)
            # Convert indices to dates
            bkp_dates = [str(pd.Timestamp(dates[min(b - 1, len(dates) - 1)]).strftime("%Y-%m")) for b in bkps[:-1]]
            bp_results[f"{n_bkps}_break"] = {
                "breakpoint_indices": [int(b) for b in bkps[:-1]],
                "breakpoint_dates": bkp_dates,
            }
            print(f"    {n_bkps}ブレーク: {bkp_dates}")

        # Penalty-based automatic detection
        pen_value = np.log(n) * residuals.var() * 2  # BIC-like
        auto_bkps = algo.predict(pen=pen_value)
        auto_dates = [str(pd.Timestamp(dates[min(b - 1, len(dates) - 1)]).strftime("%Y-%m")) for b in auto_bkps[:-1]]
        bp_results["auto_bic"] = {
            "breakpoint_indices": [int(b) for b in auto_bkps[:-1]],
            "breakpoint_dates": auto_dates,
            "penalty": float(pen_value),
        }
        print(f"    自動(BIC): {auto_dates}")

        # Check if any break is near 2017-2018
        target_range = (pd.Timestamp("2017-01-01"), pd.Timestamp("2019-01-01"))
        near_target = False
        for b in bp_results.get("1_break", {}).get("breakpoint_indices", []):
            if b < len(dates):
                d = pd.Timestamp(dates[b])
                if target_range[0] <= d <= target_range[1]:
                    near_target = True
        bp_results["near_2017_2018"] = near_target

        results["bai_perron"] = bp_results
    except Exception as e:
        results["bai_perron"] = {"error": str(e)}

    # ── CUSUM ──
    print("\n  CUSUM検定:")
    try:
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        cusum_stat, cusum_pval = breaks_cusumolsresid(res.resid)
        results["cusum"] = {
            "statistic": float(cusum_stat),
            "p_value": float(cusum_pval),
            "structural_instability_5pct": cusum_pval < 0.05,
        }
        print(f"    CUSUM: stat={cusum_stat:.3f}, p={cusum_pval:.4f}")
    except Exception as e:
        results["cusum"] = {"error": str(e)}

    # ── CUSUM plot ──
    try:
        cum_resid = np.cumsum(residuals) / (np.std(residuals) * np.sqrt(len(residuals)))
        n = len(residuals)
        t_range = np.arange(n)
        # Brown-Durbin-Evans bounds (approximate)
        a = 0.948  # 5% significance
        upper = a * np.sqrt(1 + 2 * t_range / n)
        lower = -upper

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates[:n], cum_resid, 'b-', linewidth=1.5, label='CUSUM')
        ax.plot(dates[:n], upper, 'r--', linewidth=1, label='5% bounds')
        ax.plot(dates[:n], lower, 'r--', linewidth=1)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.axvline(pd.Timestamp("2018-01-01"), color='orange', linestyle=':', label='2018-01')
        ax.set_title("CUSUM of Recursive Residuals")
        ax.legend()
        ax.set_xlabel("Date")
        plt.tight_layout()
        cusum_path = CHART_DIR / "robustness_cusum.png"
        plt.savefig(cusum_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  CUSUMプロット保存: {cusum_path}")
    except Exception as e:
        print(f"  CUSUMプロットエラー: {e}")

    # ── Breakpoints visualization ──
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates[:len(residuals)], residuals, 'b-', alpha=0.6, linewidth=0.8)
        ax.axhline(0, color='grey', linewidth=0.5)
        colors = ['red', 'green', 'purple']
        for n_bkps, color in zip([1, 2, 3], colors):
            bkps = bp_results.get(f"{n_bkps}_break", {}).get("breakpoint_indices", [])
            for b in bkps:
                if b < len(dates):
                    ax.axvline(pd.Timestamp(dates[b]), color=color, linestyle='--',
                              alpha=0.7, label=f'{n_bkps}-break: {pd.Timestamp(dates[b]):%Y-%m}')
        ax.axvline(pd.Timestamp("2018-01-01"), color='orange', linestyle=':',
                  linewidth=2, label='2018-01 (hypothesized)')
        ax.set_title("Bai-Perron Breakpoint Detection (M3 residuals)")
        ax.legend(fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual")
        plt.tight_layout()
        bp_path = CHART_DIR / "robustness_breakpoints.png"
        plt.savefig(bp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ブレークポイントプロット保存: {bp_path}")
    except Exception as e:
        print(f"  ブレークポイントプロットエラー: {e}")

    save_json(results, DATA_DIR / "robustness_structural_breaks.json")
    return results


# ═══════════════════════════════════════════════════════════
# Step 6: Clark-West検定 (§4.6)
# ═══════════════════════════════════════════════════════════

def run_clark_west_test(data):
    """Clark-West (2006) 検定"""
    print("\n" + "=" * 60)
    print("§4.6 Clark-West検定")
    print("=" * 60)

    merged = data["merged"].copy()
    season_cols = [f"m{m}" for m in range(2, 13)]
    for m in range(2, 13):
        col = f"m{m}"
        if col not in merged.columns:
            merged[col] = (merged["date"].dt.month == m).astype(float)

    y = merged["discretionary_share_yoy"].values
    dates = merged["date"].values
    n = len(y)

    # Rolling window OOS prediction (matching Phase E design)
    train_min = 64  # minimum training window
    sti_lags = [1, 2, 3, 4]  # AR(4) + STI lags

    errors_a = []  # Model A: baseline (AR + controls + seasonals)
    errors_b = []  # Model B: + STI
    yhat_a_list = []
    yhat_b_list = []
    oos_dates = []

    x_base_cols = ["disposable_income_yoy", "cpi_yoy", "unemployment_rate_yoy"] + season_cols
    x_sti_cols = ["STI_yoy"] + x_base_cols

    n_errors = 0
    for t in range(train_min, n):
        train = merged.iloc[:t].copy()
        test_row = merged.iloc[[t]].copy()

        y_train = train["discretionary_share_yoy"]
        y_test = test_row["discretionary_share_yoy"].values[0]

        # Model A (no STI)
        try:
            Xa_train = sm.add_constant(train[x_base_cols], has_constant='add')
            Xa_test = sm.add_constant(test_row[x_base_cols], has_constant='add')
            res_a = sm.OLS(y_train, Xa_train).fit()
            yhat_a = float(res_a.predict(Xa_test).iloc[0])
        except Exception as e:
            if n_errors < 3:
                print(f"    [debug] t={t} Model A error: {e}")
            n_errors += 1
            continue

        # Model B (with STI)
        try:
            Xb_train = sm.add_constant(train[x_sti_cols], has_constant='add')
            Xb_test = sm.add_constant(test_row[x_sti_cols], has_constant='add')
            res_b = sm.OLS(y_train, Xb_train).fit()
            yhat_b = float(res_b.predict(Xb_test).iloc[0])
        except Exception as e:
            if n_errors < 3:
                print(f"    [debug] t={t} Model B error: {e}")
            n_errors += 1
            continue

        e_a = y_test - yhat_a
        e_b = y_test - yhat_b

        errors_a.append(e_a)
        errors_b.append(e_b)
        yhat_a_list.append(yhat_a)
        yhat_b_list.append(yhat_b)
        oos_dates.append(dates[t])

    errors_a = np.array(errors_a)
    errors_b = np.array(errors_b)
    yhat_a_arr = np.array(yhat_a_list)
    yhat_b_arr = np.array(yhat_b_list)

    # Clark-West adjustment: f_t = e1t² - (e2t² - (ŷ1t - ŷ2t)²)
    f_t = errors_a**2 - (errors_b**2 - (yhat_a_arr - yhat_b_arr)**2)
    cw_mean = f_t.mean()
    cw_se = f_t.std(ddof=1) / np.sqrt(len(f_t))
    cw_tstat = cw_mean / cw_se
    cw_pval = 1 - sp_stats.norm.cdf(cw_tstat)  # one-sided

    # Diebold-Mariano for comparison
    d_t = errors_a**2 - errors_b**2
    dm_mean = d_t.mean()
    dm_se = d_t.std(ddof=1) / np.sqrt(len(d_t))
    dm_tstat = dm_mean / dm_se
    dm_pval = 1 - sp_stats.norm.cdf(dm_tstat)  # one-sided

    rmse_a = np.sqrt((errors_a**2).mean())
    rmse_b = np.sqrt((errors_b**2).mean())

    results = {
        "n_predictions": len(errors_a),
        "rmse_model_a": float(rmse_a),
        "rmse_model_b": float(rmse_b),
        "rmse_improvement_pct": float((rmse_a - rmse_b) / rmse_a * 100),
        "clark_west": {
            "t_statistic": float(cw_tstat),
            "p_value": float(cw_pval),
            "significant_10pct": cw_pval < 0.10,
            "significant_5pct": cw_pval < 0.05,
        },
        "diebold_mariano": {
            "t_statistic": float(dm_tstat),
            "p_value": float(dm_pval),
            "significant_10pct": dm_pval < 0.10,
        },
    }

    print(f"  OOS予測数: {len(errors_a)}")
    print(f"  RMSE改善: {(rmse_a - rmse_b)/rmse_a*100:+.1f}% (A={rmse_a:.4f}, B={rmse_b:.4f})")
    print(f"  Clark-West: t={cw_tstat:.3f}, p={cw_pval:.4f} {'***' if cw_pval < 0.01 else '**' if cw_pval < 0.05 else '*' if cw_pval < 0.10 else ''}")
    print(f"  Diebold-Mariano: t={dm_tstat:.3f}, p={dm_pval:.4f}")

    save_json(results, DATA_DIR / "robustness_clark_west.json")
    return results


# ═══════════════════════════════════════════════════════════
# Step 7: プラセボテスト (§4.7)
# ═══════════════════════════════════════════════════════════

def run_placebo_tests(data):
    """プラセボ・反証テスト"""
    print("\n" + "=" * 60)
    print("§4.7 プラセボテスト")
    print("=" * 60)

    merged = data["merged"].copy()
    ds_df = data["ds_df"].copy()
    season_cols = [f"m{m}" for m in range(2, 13)]
    for m in range(2, 13):
        col = f"m{m}"
        if col not in merged.columns:
            merged[col] = (merged["date"].dt.month == m).astype(float)

    results = {}
    np.random.seed(42)

    # ── 7.1 シャッフルSTI ──
    print("\n  [7.1] シャッフルSTI (1000回):")
    y = merged["discretionary_share_yoy"]
    x_cols = ["STI_yoy", "disposable_income_yoy", "cpi_yoy",
              "unemployment_rate_yoy"] + season_cols
    X = sm.add_constant(merged[x_cols])
    real_res = sm.OLS(y, X).fit()
    real_beta = float(real_res.params["STI_yoy"])

    n_shuffle = 1000
    shuffle_betas = []
    for i in range(n_shuffle):
        shuffled = merged.copy()
        shuffled["STI_yoy"] = np.random.permutation(shuffled["STI_yoy"].values)
        X_s = sm.add_constant(shuffled[x_cols])
        res_s = sm.OLS(y, X_s).fit()
        shuffle_betas.append(float(res_s.params["STI_yoy"]))

    shuffle_betas = np.array(shuffle_betas)
    percentile = float(np.mean(shuffle_betas <= real_beta) * 100)  # lower tail (β is negative)

    results["shuffle_sti"] = {
        "real_beta": real_beta,
        "shuffle_mean": float(shuffle_betas.mean()),
        "shuffle_std": float(shuffle_betas.std()),
        "percentile": percentile,
        "in_lower_5pct": percentile < 5.0,
        "in_lower_1pct": percentile < 1.0,
    }
    print(f"    実β = {real_beta:.4f}")
    print(f"    シャッフル: mean={shuffle_betas.mean():.4f}, std={shuffle_betas.std():.4f}")
    print(f"    パーセンタイル: {percentile:.1f}% (下側5%{'以内 ✓' if percentile < 5 else '外'})")

    # ── 7.2 ランダム指標プラセボ ──
    print("\n  [7.2] ランダム指標プラセボ (1000回):")
    n_random = 1000
    random_betas = []
    for i in range(n_random):
        rand_sti = np.random.randn(len(merged))
        rand_merged = merged.copy()
        rand_merged["STI_yoy"] = rand_sti
        X_r = sm.add_constant(rand_merged[x_cols])
        res_r = sm.OLS(y, X_r).fit()
        random_betas.append(float(res_r.params["STI_yoy"]))

    random_betas = np.array(random_betas)
    rand_percentile = float(np.mean(random_betas <= real_beta) * 100)

    results["random_indicator"] = {
        "real_beta": real_beta,
        "random_mean": float(random_betas.mean()),
        "random_std": float(random_betas.std()),
        "percentile": rand_percentile,
        "in_lower_5pct": rand_percentile < 5.0,
    }
    print(f"    ランダム: mean={random_betas.mean():.4f}, std={random_betas.std():.4f}")
    print(f"    パーセンタイル: {rand_percentile:.1f}%")

    # ── 7.3 非裁量消費ターゲット ──
    print("\n  [7.3] 非裁量消費ターゲット:")
    try:
        # non-discretionary share = 100 - discretionary_share
        nd_df = ds_df[["date", "discretionary_share", "total_consumption", "discretionary"]].copy()
        nd_df["nondiscretionary_share"] = 100.0 - nd_df["discretionary_share"]
        nd_yoy = compute_yoy(nd_df, "date", "nondiscretionary_share")

        nd_merged = (
            nd_yoy[["date", "nondiscretionary_share_yoy"]]
            .merge(merged[["date", "STI_yoy", "disposable_income_yoy", "cpi_yoy",
                          "unemployment_rate_yoy"] + season_cols], on="date")
            .dropna()
        )

        y_nd = nd_merged["nondiscretionary_share_yoy"]
        x_nd_cols = ["STI_yoy", "disposable_income_yoy", "cpi_yoy",
                     "unemployment_rate_yoy"] + season_cols
        X_nd = sm.add_constant(nd_merged[x_nd_cols])
        res_nd = sm.OLS(y_nd, X_nd).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        nd_beta = float(res_nd.params["STI_yoy"])
        nd_pval = float(res_nd.pvalues["STI_yoy"])

        results["nondiscretionary_target"] = {
            "beta_STI": nd_beta,
            "p_value": nd_pval,
            "significant_5pct": nd_pval < 0.05,
            "note": "Non-discretionary = 100 - discretionary. β should have opposite sign.",
        }
        print(f"    非裁量消費 β(STI) = {nd_beta:+.4f}, p = {nd_pval:.4f}")
        print(f"    (裁量消費のβ={real_beta:+.4f}と符号反転していれば整合的)")
    except Exception as e:
        results["nondiscretionary_target"] = {"error": str(e)}

    # ── 7.4 未来リーク検定 ──
    print("\n  [7.4] 未来リーク検定:")
    leak_results = {}
    for k in [1, 3, 6, 12]:
        try:
            leak_merged = merged.copy()
            leak_merged["STI_future"] = leak_merged["STI_yoy"].shift(-k)
            leak_clean = leak_merged.dropna(subset=["STI_future"])

            y_lk = leak_clean["discretionary_share_yoy"]
            x_lk_cols = ["STI_future", "disposable_income_yoy", "cpi_yoy",
                         "unemployment_rate_yoy"] + season_cols
            X_lk = sm.add_constant(leak_clean[x_lk_cols])
            res_lk = sm.OLS(y_lk, X_lk).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

            lk_beta = float(res_lk.params["STI_future"])
            lk_pval = float(res_lk.pvalues["STI_future"])
            leak_results[f"k={k}"] = {
                "beta": lk_beta, "p_value": lk_pval,
                "significant_5pct": lk_pval < 0.05,
            }
            sig = "***" if lk_pval < 0.01 else "**" if lk_pval < 0.05 else "*" if lk_pval < 0.10 else ""
            print(f"    STI(t+{k:2d}): β={lk_beta:+.4f}, p={lk_pval:.4f} {sig}")
        except Exception as e:
            leak_results[f"k={k}"] = {"error": str(e)}

    results["future_leak"] = leak_results

    # ── Placebo distribution plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(shuffle_betas, bins=40, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Shuffled STI')
    ax.axvline(real_beta, color='red', linewidth=2, linestyle='--',
              label=f'Real β = {real_beta:.3f}\n({percentile:.1f}th percentile)')
    ax.axvline(np.percentile(shuffle_betas, 5), color='orange', linewidth=1,
              linestyle=':', label='5th percentile')
    ax.set_xlabel("β coefficient (STI_yoy)")
    ax.set_title("Shuffle Test: STI β Distribution (N=1000)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(random_betas, bins=40, density=True, alpha=0.7, color='lightgreen',
            edgecolor='black', label='Random indicator')
    ax.axvline(real_beta, color='red', linewidth=2, linestyle='--',
              label=f'Real β = {real_beta:.3f}\n({rand_percentile:.1f}th percentile)')
    ax.set_xlabel("β coefficient")
    ax.set_title("Random Indicator Placebo (N=1000)")
    ax.legend(fontsize=9)

    plt.suptitle("SSQ Placebo Tests", fontsize=14)
    plt.tight_layout()
    plot_path = CHART_DIR / "robustness_placebo_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  プロット保存: {plot_path}")

    save_json(results, DATA_DIR / "robustness_placebo.json")
    return results


# ═══════════════════════════════════════════════════════════
# Step 8: 代替STI仕様 (§4.5)
# ═══════════════════════════════════════════════════════════

def run_alternative_sti_specs(data):
    """代替STI仕様でのβ安定性テスト"""
    print("\n" + "=" * 60)
    print("§4.5 代替STI仕様テスト")
    print("=" * 60)

    merged = data["merged"].copy()
    sti_df = data["sti_df"].copy()
    ds_df = data["ds_df"].copy()

    season_cols = [f"m{m}" for m in range(2, 13)]
    for m in range(2, 13):
        col = f"m{m}"
        if col not in merged.columns:
            merged[col] = (merged["date"].dt.month == m).astype(float)

    results = {}

    # ── 仕様1: 求人倍率反転のみ ──
    print("\n  [1] 求人倍率反転のみ (job_scarcity_z):")
    try:
        sti_1 = sti_df[["date", "job_scarcity_z"]].copy()
        sti_1.rename(columns={"job_scarcity_z": "alt_STI"}, inplace=True)
        r1 = _run_alt_regression(sti_1, merged, season_cols, "job_scarcity_only")
        results["job_scarcity_only"] = r1
    except Exception as e:
        results["job_scarcity_only"] = {"error": str(e)}

    # ── 仕様2: 消費者態度指数反転のみ ──
    print("\n  [2] 消費者態度指数反転のみ (consumer_anxiety_z):")
    try:
        sti_2 = sti_df[["date", "consumer_anxiety_z"]].copy()
        sti_2.rename(columns={"consumer_anxiety_z": "alt_STI"}, inplace=True)
        r2 = _run_alt_regression(sti_2, merged, season_cols, "consumer_anxiety_only")
        results["consumer_anxiety_only"] = r2
    except Exception as e:
        results["consumer_anxiety_only"] = {"error": str(e)}

    # ── 仕様3: 3成分 (+ 失業率) ──
    print("\n  [3] 3成分 (job_scarcity + consumer_anxiety + unemployment):")
    try:
        # unemployment rate z-score using 60-month window (same as STI construction)
        unemp_col = merged[["date", "unemployment_rate_yoy"]].copy()
        # Use the existing unemployment_rate_yoy as a proxy component
        sti_3 = sti_df[["date", "job_scarcity_z", "consumer_anxiety_z"]].copy()
        # For 3-component: need to add unemployment z-score to STI level before YoY
        # Simpler: recompute at YoY level
        sti_3["alt_STI"] = (sti_3["job_scarcity_z"] + sti_3["consumer_anxiety_z"]) / 2
        # We need unemployment z at the level, then take YoY
        # Approximate: use the original 2-component STI and add unemployment via merged
        # Better: compute 3-component z-score from merged
        sti_3_merged = merged.copy()
        # Standardize unemployment_rate_yoy to same scale as STI_yoy
        u_std = sti_3_merged["unemployment_rate_yoy"].std()
        s_std = sti_3_merged["STI_yoy"].std()
        sti_3_merged["alt_STI_yoy"] = (
            sti_3_merged["STI_yoy"] * 2 / 3 +
            sti_3_merged["unemployment_rate_yoy"] / u_std * s_std / 3
        )
        r3 = _run_alt_regression_yoy(sti_3_merged, season_cols, "3_component")
        results["3_component"] = r3
    except Exception as e:
        results["3_component"] = {"error": str(e)}

    # ── 仕様4: PCA重み付け ──
    print("\n  [4] PCA重み付け:")
    try:
        pca_data = sti_df[["job_scarcity_z", "consumer_anxiety_z"]].dropna()
        # Manual PCA (no sklearn needed)
        cov_mat = np.cov(pca_data.values.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        # PC1 (largest eigenvalue)
        pc1_weights = eigenvectors[:, -1]  # last column = largest eigenvalue
        # Ensure positive direction (larger value = more threat)
        if pc1_weights.sum() < 0:
            pc1_weights = -pc1_weights

        explained_var = eigenvalues[-1] / eigenvalues.sum()

        sti_pca = sti_df[["date", "job_scarcity_z", "consumer_anxiety_z"]].dropna().copy()
        sti_pca["alt_STI"] = (
            sti_pca["job_scarcity_z"] * pc1_weights[0] +
            sti_pca["consumer_anxiety_z"] * pc1_weights[1]
        )
        r4 = _run_alt_regression(sti_pca[["date", "alt_STI"]], merged, season_cols, "PCA_weighted")
        r4["pca_weights"] = {"job_scarcity": float(pc1_weights[0]),
                             "consumer_anxiety": float(pc1_weights[1])}
        r4["explained_variance_ratio"] = float(explained_var)
        results["PCA_weighted"] = r4
        print(f"    PCA weights: job_scarcity={pc1_weights[0]:.3f}, consumer_anxiety={pc1_weights[1]:.3f}")
        print(f"    分散説明率: {explained_var:.1%}")
    except Exception as e:
        results["PCA_weighted"] = {"error": str(e)}

    # ── Summary table ──
    print(f"\n  {'仕様':<25s} | {'β(STI)':>10s} | {'p値':>10s} | {'R²':>8s}")
    print(f"  {'─' * 25}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 8}")
    # Original
    orig = data["phase_d_results"]["models"]["M3_Full"]
    orig_beta = orig["coefficients"]["STI_yoy"]["estimate"]
    orig_p = orig["coefficients"]["STI_yoy"]["p_value"]
    orig_r2 = orig["r_squared"]
    print(f"  {'Original (2-comp equal)':25s} | {orig_beta:>+10.4f} | {orig_p:>10.4f} | {orig_r2:>8.3f}")

    for name, r in results.items():
        if isinstance(r, dict) and "beta_alt_STI" in r:
            print(f"  {name:25s} | {r['beta_alt_STI']:>+10.4f} | {r['p_value']:>10.4f} | {r['r_squared']:>8.3f}")

    save_json(results, DATA_DIR / "robustness_alternative_sti.json")
    return results


def _run_alt_regression(alt_sti_df, merged, season_cols, label):
    """代替STI (level) でYoY変換 → M3回帰"""
    alt_yoy = compute_yoy(alt_sti_df, "date", "alt_STI")
    alt_merged = (
        merged[["date", "discretionary_share_yoy", "disposable_income_yoy",
                "cpi_yoy", "unemployment_rate_yoy"] + season_cols]
        .merge(alt_yoy[["date", "alt_STI_yoy"]], on="date")
        .dropna()
    )

    y = alt_merged["discretionary_share_yoy"]
    x_cols = ["alt_STI_yoy", "disposable_income_yoy", "cpi_yoy",
              "unemployment_rate_yoy"] + season_cols
    X = sm.add_constant(alt_merged[x_cols])
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    beta = float(res.params["alt_STI_yoy"])
    pval = float(res.pvalues["alt_STI_yoy"])
    r2 = float(res.rsquared)

    print(f"    β = {beta:+.4f}, p = {pval:.4f}, R² = {r2:.3f}")
    return {"beta_alt_STI": beta, "p_value": pval, "r_squared": r2, "n_obs": int(res.nobs)}


def _run_alt_regression_yoy(alt_merged, season_cols, label):
    """代替STI (既にYoY) でM3回帰"""
    y = alt_merged["discretionary_share_yoy"]
    x_cols = ["alt_STI_yoy", "disposable_income_yoy", "cpi_yoy",
              "unemployment_rate_yoy"] + season_cols
    X = sm.add_constant(alt_merged[x_cols])
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    beta = float(res.params["alt_STI_yoy"])
    pval = float(res.pvalues["alt_STI_yoy"])
    r2 = float(res.rsquared)

    print(f"    β = {beta:+.4f}, p = {pval:.4f}, R² = {r2:.3f}")
    return {"beta_alt_STI": beta, "p_value": pval, "r_squared": r2, "n_obs": int(res.nobs)}


# ═══════════════════════════════════════════════════════════
# Step 9: Go/No-Go判定
# ═══════════════════════════════════════════════════════════

def evaluate_go_nogo(stationarity, structural, clark_west, placebo):
    """Go/No-Go 4基準を自動評価"""
    print("\n" + "=" * 60)
    print("Go/No-Go 判定")
    print("=" * 60)

    criteria = {}

    # 1. 定常性: STI_yoy が 2/3テストでI(0)、またはI(1)+共和分あり
    sti_sum = stationarity.get("STI_yoy", {}).get("_summary", {})
    is_i0 = sti_sum.get("judgment") == "I(0)"
    coint = stationarity.get("cointegration", {})
    has_coint = (
        coint.get("engle_granger", {}).get("cointegrated_5pct", False) or
        coint.get("johansen", {}).get("n_cointegrating_relations", 0) > 0
    )
    stat_pass = is_i0 or has_coint
    criteria["stationarity"] = {
        "pass": stat_pass,
        "detail": f"I(0)={is_i0}, cointegrated={has_coint}",
    }
    print(f"  1. 定常性: {'PASS ✓' if stat_pass else 'FAIL ✗'} — I(0)={is_i0}, 共和分={has_coint}")

    # 2. Bai-Perron: 2017-2018付近にブレーク検出
    bp = structural.get("bai_perron", {})
    bp_pass = bp.get("near_2017_2018", False)
    bp_dates = bp.get("1_break", {}).get("breakpoint_dates", [])
    criteria["structural_break"] = {
        "pass": bp_pass,
        "detail": f"1-break dates: {bp_dates}",
    }
    print(f"  2. 構造変化: {'PASS ✓' if bp_pass else 'FAIL ✗'} — 1-break: {bp_dates}")

    # 3. Clark-West: p < 0.10
    cw = clark_west.get("clark_west", {})
    cw_pass = cw.get("significant_10pct", False)
    cw_p = cw.get("p_value", 1.0)
    criteria["clark_west"] = {
        "pass": cw_pass,
        "detail": f"p={cw_p:.4f}",
    }
    print(f"  3. Clark-West: {'PASS ✓' if cw_pass else 'FAIL ✗'} — p={cw_p:.4f}")

    # 4. シャッフルプラセボ: 実βが95%タイル以上 (= lower 5%)
    shuffle = placebo.get("shuffle_sti", {})
    shuffle_pass = shuffle.get("in_lower_5pct", False)
    pctile = shuffle.get("percentile", 100)
    criteria["shuffle_placebo"] = {
        "pass": shuffle_pass,
        "detail": f"percentile={pctile:.1f}%",
    }
    print(f"  4. シャッフル: {'PASS ✓' if shuffle_pass else 'FAIL ✗'} — {pctile:.1f}パーセンタイル")

    # Overall
    all_pass = all(c["pass"] for c in criteria.values())
    n_pass = sum(1 for c in criteria.values() if c["pass"])
    verdict = "GO" if all_pass else "CONDITIONAL_GO" if n_pass >= 3 else "REVIEW_NEEDED"

    result = {
        "criteria": criteria,
        "n_pass": n_pass,
        "n_total": 4,
        "verdict": verdict,
    }

    print(f"\n  総合判定: {verdict} ({n_pass}/4 基準クリア)")
    if verdict == "REVIEW_NEEDED":
        failed = [k for k, v in criteria.items() if not v["pass"]]
        print(f"  要検討: {', '.join(failed)}")

    save_json(result, DATA_DIR / "robustness_go_nogo.json")
    return result


# ═══════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SSQ Robustness Battery — Phase 1")
    print("=" * 60)

    # ── Load data ──
    data = load_data()

    # ── Step 2: Stationarity (highest priority) ──
    stationarity = run_stationarity_battery(data)

    # ── Step 3: Toda-Yamamoto (highest priority) ──
    toda_yamamoto = run_toda_yamamoto(data, stationarity)

    # ── Step 4: Residual diagnostics ──
    residual_diag = run_residual_diagnostics(data)

    # ── Step 5: Structural breaks ──
    structural = run_structural_break_tests(data)

    # ── Step 6: Clark-West ──
    clark_west = run_clark_west_test(data)

    # ── Step 7: Placebo tests ──
    placebo = run_placebo_tests(data)

    # ── Step 8: Alternative STI specifications ──
    alt_sti = run_alternative_sti_specs(data)

    # ── Step 9: Go/No-Go ──
    go_nogo = evaluate_go_nogo(stationarity, structural, clark_west, placebo)

    print(f"\n{'=' * 60}")
    print("ロバストネスバッテリー完了!")
    print(f"{'=' * 60}")
    print(f"  結果JSON: {DATA_DIR}/robustness_*.json")
    print(f"  プロット: {CHART_DIR}/robustness_*.png")
    print(f"  Go/No-Go: {go_nogo['verdict']}")


if __name__ == "__main__":
    main()
