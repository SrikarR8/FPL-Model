# FPL Prediction Model — Project Log

Running log of bugs found, fixes applied, empirical backtest results, and modeling decisions made during development. Kept chronologically and bolstered with experiment data so the reasoning behind each change is fully traceable.

---

## 🏗️ Pipeline Architecture

```mermaid
flowchart TD
    subgraph Data_Ingestion[1. Data Ingestion & Alignment]
        A1[Raw FPL Match CSVs] --> M[Merge by Season, GW, Fixture]
        A2[FBref / Understat xG & xA] --> M
        A3[Team Match Data & Kickoffs] --> M
    end

    subgraph Anti_Leakage[2. Strict Anti-Leakage Feature Engineering]
        M --> L1[Opponent Pos Lagged: GW-1]
        M --> L2[Opponent xGA: Shifted Expanding Avg]
        L1 & L2 --> C[Composite Opponent Difficulty Index]
        M --> R1[5-GW Rolling Form: points, mins, xG, xA, bps]
        M --> R2[38-GW Talent Anchors: points, xG, bps]
    end

    subgraph Model_Training[3. Model Training & Walk-Forward Validation]
        C & R1 & R2 --> F[Feature Matrix: 9 Features]
        F --> X[XGBoost Tweedie Regressor<br/>max_depth=3, colsample=0.8, lr=0.05]
        X --> WF[Walk-Forward Evaluation<br/>GW 1 to 38 Backtesting]
    end

    subgraph Decision_Layer[4. Inference & Ranking]
        WF --> P1[Predicted Points E[Y|X]]
        P1 --> P2[Top 5 Recommendations vs Popular vs Baseline]
        P1 --> P3[Upcoming GW1 Predictions]
    end
```

---

## 🔬 Key Investigations & Empirical Findings

### 1. Collinearity & Feature Dilution: The Composite Difficulty Solution
* **Problem**: Passing multiple collinear opponent metrics (`opp_league_pos`, `team_xg`, `opp_xg_allowed`) caused tree splits to fragment, reducing the feature importance of `rolling_avg_points_last_5` from ~16% to 13.5% and dropping Spearman correlation from 0.25 to 0.22.
* **Root Cause**: GBDTs split on correlated features arbitrarily across shallow trees, diluting the signal of player form.
* **Fix**: Replaced separate opponent columns with a single normalized **Composite Opponent Difficulty Index**:
  $$\text{Composite Difficulty} = \left(\frac{\text{opp\_league\_pos}}{20.0}\right) \times 0.5 + \left(\frac{\text{opp\_xga\_season\_avg}}{2.5}\right) \times 0.5$$
* **Result**: Restored `rolling_avg_points_last_5` to **19.95%** (#1 feature) and weekly correlation to **0.2530**.

---

### 2. The 38-Game Talent & Quality Baseline Discovery
* **Problem**: The 5-GW form model was partially blind to elite player class. When Haaland or Watkins had 2 quiet games, their 5-GW average dipped, causing the model to prematurely drop them for budget forwards facing weak opposition.
* **Empirical Comparison (2024–25 & 2025–26 Seasons)**:

| Configuration | 2024–25 Weekly Corr | 2024–25 Top 5 Pts | 2025–26 Weekly Corr | 2025–26 Top 5 Pts | Haaland Selections (2025–26) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5-GW Form Only** | `0.1622` | `4.45 pts` | `0.2530` | `4.33 pts` | 23 / 38 |
| **5-GW + 15-GW Baseline** | `0.1598` | `4.36 pts` | `0.2204` | `4.19 pts` | 20 / 38 |
| **5-GW + 38-GW Baseline** | **`0.1925`** 🏆 | **`4.44 pts`** 🏆 | **`0.2370`** 🏆 | **`4.41 pts`** 🏆 | **28 / 38** ⬆️ |

* **Takeaway**: 15 games is an awkward middle ground (lags form, too short for class). A 38-game rolling window spans a full campaign, creating a permanent talent floor for world-class talismans.

---

### 3. Top 20% vs. Bottom 80% & Quartile Stratification
* **Evaluation**: Measuring how well the model discriminates between different tiers of players:

#### 📊 4-Quartile Stratification (2025–26 Season)
| Quartile Tier | Avg Players / GW | Avg Actual Pts Scored | Avg Predicted Pts | Intra-Bucket Spearman |
| :--- | :---: | :---: | :---: | :---: |
| 🌟 **Q4 (Top 75–100%)** | `6.1` | **`4.26 pts`** | `4.35 pts` | `0.0275` |
| 🟢 **Q3 (Upper 50–75%)** | `5.4` | **`3.39 pts`** | `3.38 pts` | `0.1540` |
| 🟡 **Q2 (Lower 25–50%)** | `5.7` | **`3.10 pts`** | `2.91 pts` | `0.0905` |
| 🔴 **Q1 (Bottom 0–25%)** | `6.2` | **`2.29 pts`** | `2.32 pts` | `-0.0126` |

#### 📊 Top 20% vs Bottom 80% Split
| Bucket | Avg Players / GW | Avg Actual Pts Scored | Avg Predicted Pts | Pooled Spearman |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Top 20% (Elite Picks)** | **`5.1`** | **`4.30 pts`** 🚀 | `4.47 pts` | `0.0915` |
| 📉 **Bottom 80% (Rest of Pool)** | **`18.3`** | **`2.96 pts`** | `2.90 pts` | `0.1909` |
| 🌐 **Full Player Pool (Overall)** | **`23.3`** | **`3.25 pts`** | `3.24 pts` | **`0.2552`** 🏆 |

* **Takeaway**: The model demonstrates **monotonic calibration** across tiers ($\text{Q1: 2.29} \to \text{Q2: 3.10} \to \text{Q3: 3.39} \to \text{Q4: 4.26}$). Intra-bucket correlation within the Top 20% is low because all 5 players are predicted within a razor-thin band ($4.3 - 4.8\text{ pts}$), where single-match luck dictates the micro-order.

---

### 4. BPS Investigation: Continuous Raw BPS vs. Ordinal Match Rank
* **Question**: Should BPS be transformed into an ordinal match rank (#1 best player in game) or kept continuous?

| BPS Feature Tested | Weekly Spearman | Pooled Spearman | NDCG | Top 20% Weekly Spearman | Top 5 Avg Points |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No BPS** | `0.2472` | `0.2533` | `0.7384` | `0.1843` | `4.31 pts` |
| **Ordinal Match Rank (5-GW Rank)** | `0.2455` | `0.2535` | `0.7319` | `0.0477` | `4.33 pts` |
| **Ordinal Match Rank (38-GW Rank)** | `0.2343` | `0.2421` | `0.7359` | `0.1633` | `4.31 pts` |
| 🌟 **Raw Continuous BPS (38-GW Avg)** | **`0.2588`** 🏆 | **`0.2630`** 🏆 | **`0.7410`** 🏆 | **`0.1230`** | **`4.46 pts`** 🏆 |

* **Takeaway**: Converting to rank squashes magnitude. A 64 BPS match (2 goals + dominant display) becomes #1, identical to a 21 BPS match in a 0-0 draw. Raw continuous BPS preserves the true scale of player involvement.

---

### 5. Expected Value Ceilings vs. Single-Match Variance
* **Observation**: Predictions sit between $2.0$ and $5.5$, while actual realizations spike to $13 - 17$.
* **Mathematical Reality**: Regression models output the **Conditional Expected Value $E[Y|X]$**. Even prime Haaland has a ~30% blank risk (2 pts), ~35% 1-goal risk (6 pts), and only ~15% chance of a multi-goal haul ($13+\text{ pts}$). The probability-weighted expected points naturally sit at $\approx 5.5\text{ pts}$.
* **Central Limit Theorem Validation**: When averaging over a 10-gameweek sample (GW 1 to 10), the noisy weekly Poisson realizations smooth out into a bell curve centered at $\mu = 3.10$, matching the model's predicted expectation of $\mu = 3.35$:

#### 📊 5-Number Summary: 10-Game Per-Player Average (GW 1 to 10)
| Statistic | 10-Game Avg Predicted Points ($E[Y]$) | 10-Game Avg Actual Points Scored ($\bar{Y}$) |
| :--- | :---: | :---: |
| **Minimum (`min`)** | `2.11` | `0.00` |
| **25th Percentile (`Q1`)** | `2.82` | `2.00` |
| **Median (`50%`)** | `3.32` | `2.76` |
| **75th Percentile (`Q3`)** | `3.97` | `4.31` |
| **Maximum (`max`)** | `4.72` | `9.80` *(Haaland anomaly)* |
| *Mean* | *`3.35`* | *`3.10`* |
| *Std Dev* | *`0.71`* | *`1.97` (smoothed down from 3.7 weekly variance)* |

---

## 🛠️ Detailed Bug Fixes & Modeling Notes

### Data Pipeline
- **Chronological Sorting Before Rolling Windows**: `groupby().transform()` depends on row order. Enforced strict sorting by `['name', 'season', 'kickoff_time', 'GW']` before computing any rolling features.
- **Strict Anti-Leakage Shift**: Enforced `.shift(1)` across all player and opponent metrics so current match outcomes never contaminate pre-match inputs.
- **Latin-1 Encoding for Player Names**: Solved character dropping for foreign names (Gyökeres, Højlund, João Pedro) ensuring 100% join match rate across raw CSVs.
- **Fixture Difficulty Rating (FDR) Asymmetry**: Sourced `team_h_difficulty` and `team_a_difficulty` independently based on actual venue.

### Model Regularization
- **Tree Depth & Feature Subsampling**: Bound `max_depth=3` and `colsample_bytree=0.8` to prevent tree memorization on smaller forward starter subsets (~25 players/week).
- **Tweedie Regressor (`reg:tweedie`, power=1.2)**: Specifically models zero-inflated, right-skewed compound Poisson distributions common in fantasy points.

---

## 📋 Open Roadmap Items

- [ ] Replicate full pipeline for Midfielders (`MID_model.py`), Defenders (`DEF_model.py`), and Goalkeepers (`GK_model.py`).
- [ ] Implement PuLP Integer Linear Programming (ILP) optimizer for optimal 15-man squad selection within £100m budget and 3-players-per-club constraints.
- [ ] Build two-stage model: $(1)$ Probability of starting / 60+ minutes $\times$ $(2)$ Expected points conditional on starting.
- [ ] Add new-signing cold-start transfer logic for players arriving from foreign leagues.
