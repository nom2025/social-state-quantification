# Social State Quantification

*A New Framework for Real-Time Societal Observation Using Public Statistics*

> This repository provides the reference implementation and reproducible pipeline accompanying the research papers on Social State Quantification.

Author: **Tomohiro Nomura (Independent Researcher)**

---

![KSI Time Series](figures/ksi_timeseries.png)

---

## Motivation

Existing socioeconomic indicators each capture only a fragment of societal conditions:

- **GDP** is published quarterly with significant lag — it cannot detect real-time pressure.
- **CPI** measures price levels only — it says nothing about wages, employment, or household capacity.
- **Unemployment rate** reflects labor market slack — but misses underemployment, overwork, and financial stress.

There is no single indicator that captures **household-level, multi-dimensional socioeconomic pressure** in real time. The Kernel Stress Index (KSI) fills this gap by integrating 10 public statistics into a unified state quantity, enabling continuous observation of societal stress as it accumulates.

---

## Overview

**Social State Quantification (SSQ)** is a new methodological framework for observing, measuring, and interpreting the real-time condition of a society using publicly available statistics.

This repository contains:

- Reference implementation of the **Kernel Stress Index (KSI)**
- **SSQ empirical pipeline**: Social Threat Index (STI) construction, discretionary share analysis, and robustness tests
- Sample datasets and reproducible results

---

## Key Concepts

### 1. Kernel Stress Index (KSI)

A unified state quantity that integrates heterogeneous public statistics into a single continuous indicator of household-level socioeconomic pressure.

**Algorithm:**
1. Convert each indicator into a z-score based on a 60-month moving window
2. Align indicator polarity toward the "stress-increasing" direction
3. Apply LOCF (Last Observation Carried Forward) to harmonize monthly data
4. Aggregate standardized indicators via vector summation

### 2. Social Threat Index (STI)

A composite index measuring year-over-year changes in societal threat perception, constructed from unemployment rate, CPI, and consumer confidence.

### 3. Discretionary Share

The ratio of discretionary consumption to total household expenditure, capturing structural shifts in household spending behavior.

---

## Repository Structure

```
social-state-quantification/
├── paper/              - Research paper (PDF)
├── scripts/            - SSQ empirical analysis pipelines
│   ├── build_ssq_first_chart.py    - Full SSQ pipeline (STI, DS, regression, Granger, OOS)
│   └── ssq_robustness_battery.py   - Robustness tests (Toda-Yamamoto, Clark-West, placebo, etc.)
├── src/                - KSI reference implementation
│   ├── ksi.py          - Kernel Stress Index calculation
│   ├── utils.py        - Utility functions (z-score, LOCF, polarity)
│   └── examples/
│       └── demo.py     - Usage example
├── data/
│   ├── sample_data.csv - KSI sample data
│   └── ssq/            - SSQ analysis outputs
│       ├── sti_monthly.csv                - Social Threat Index time series
│       ├── discretionary_share_monthly.csv - Discretionary share time series
│       ├── phase_d_merged.csv             - Merged analysis dataset
│       ├── phase_d_regression_results.json - OLS/HAC regression results
│       ├── phase_d_granger_results.json   - Granger causality results
│       ├── phase_e_prediction_results.json - Out-of-sample prediction results
│       └── robustness_*.json              - Robustness test results
├── figures/            - Charts and diagrams
├── LICENSE
├── README.md
└── requirements.txt
```

---

## SSQ Empirical Paper

The main empirical analysis is in `scripts/build_ssq_first_chart.py`, which implements the full pipeline:

1. **Data Acquisition**: Fetches monthly data from e-Stat API (household survey, CPI, unemployment, consumer confidence)
2. **Index Construction**: Builds STI (Social Threat Index) and discretionary expenditure share
3. **Regression Analysis**: OLS with HAC standard errors (Newey-West)
4. **Granger Causality**: Tests directional causality between STI and discretionary share
5. **Out-of-Sample Prediction**: Expanding-window forecasts with Diebold-Mariano test
6. **Visualization**: Generates time series charts (Figure 1)

### Key Results

- STI strongly predicts changes in discretionary share (beta = -1.185, p = 0.0004)
- Granger causality runs from STI to discretionary share (not reverse)
- Out-of-sample RMSE improvement: +6.9% over baseline

### Robustness Tests (`scripts/ssq_robustness_battery.py`)

| Test | Result |
|------|--------|
| Toda-Yamamoto | STI -> DS: p=0.0001; DS -> STI: p=0.27 |
| Clark-West | t=3.738, p=0.0001 |
| Shuffle Placebo | 0.0 percentile (not spurious) |
| Bai-Perron | Structural break at 2018-09 |
| Alternative STI | All 4 specifications: p < 0.005 |

---

## Quick Start

```bash
git clone https://github.com/nom2025/social-state-quantification.git
cd social-state-quantification
pip install -r requirements.txt

# KSI demo
python src/ksi.py --input data/sample_data.csv

# SSQ pipeline (requires e-Stat API key in config/estat_config.json)
python scripts/build_ssq_first_chart.py
```

> **Note**: The SSQ pipeline requires an e-Stat API key. Register at https://www.e-stat.go.jp/ and place your key in `config/estat_config.json`:
> ```json
> {"api_key": "YOUR_API_KEY_HERE"}
> ```

---

## Citation

If you use this framework in your research, please cite:

```
Nomura, T. (2026). "Social Threat Perception and Changes in Household
Consumption Composition: An Empirical Analysis Using Japanese Monthly Data."
Under review.
```

---

## Updates

| Date | Change |
|------|--------|
| 2026-02-25 | Added SSQ empirical pipeline, robustness tests, and analysis data |
| 2026-02-22 | Initial public release: paper, KSI reference implementation, sample data |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
