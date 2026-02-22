# Data

## sample_data.csv

Synthetic sample data for demonstrating the KSI calculation pipeline.

**Columns:**
| Column | Description | Polarity |
|--------|-------------|----------|
| date | Monthly date (YYYY-MM-DD) | - |
| cpi | Consumer Price Index | Higher = more stress |
| total_cash_earnings | Total cash earnings (JPY) | Higher = less stress |
| unemployment_rate | Unemployment rate (%) | Higher = more stress |
| household_expenditure | Household expenditure (JPY) | Higher = less stress |
| food_engel_coefficient | Engel coefficient (%) | Higher = more stress |
| working_hours | Monthly working hours | Higher = more stress |
| part_time_ratio | Part-time worker ratio (%) | Higher = more stress |
| consumer_confidence | Consumer confidence index | Higher = less stress |
| savings_rate | Household savings rate (%) | Higher = less stress |
| debt_ratio | Household debt ratio (%) | Higher = more stress |

**Note:** This is synthetic data created to demonstrate the methodology.
For real analysis, use actual public statistics from sources such as
Japan's e-Stat (https://www.e-stat.go.jp/).
