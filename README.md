# Fantasy Premier League (FPL) ML Prediction Model

An end-to-end Machine Learning model and backtesting framework designed to project Fantasy Premier League (FPL) player points, evaluate fixture difficulties, and rank top picks for upcoming gameweeks.

---

## Project Overview

At the moment this project is an experiment, created to iterate upon the lessons learned so far from building and backtesting predictive football models. 

Currently, the model does an excellent job "separating" good players from great players across the league (delivering strong monotonic calibration from quartile to quartile). So far, the model is performing on par with the average FPL player
> **More thoughts, design decisions, and deep-dive experiment notes can be found in [`PROJECT_LOG.md`](PROJECT_LOG.md).**

---

## Overview & Methodology

The model employs regularized **XGBoost Regressors with a Tweedie distribution objective** (`reg:tweedie`, variance power = 1.2), designed specifically for zero-inflated, right-skewed point distributions common in fantasy football.

- **Strict Anti-Leakage Rules**: Lagged league standings ($GW - 1$), shifted expanding xGA averages, and shifted rolling player metrics prevent any look-ahead bias.
- **Walk-Forward Historical Backtesting**: Simulates actual in-season weekly decision-making (GW 1 to 38) to evaluate true out-of-sample predictive power.
- **Two-Stage Architecture (In Development)**: Separately predicting $(1)$ probability of playing 60+ minutes and $(2)$ expected points conditional on starting.

---

## Feature Engineering

### Current Active Features
The model currently uses the following feature set across training and inference:

1. **`rolling_avg_points_last_5`**: Short-term player form (5-game rolling average FPL points, shifted by 1).
2. **`rolling_avg_points_last_38`**: Long-term player quality anchor (38-game rolling average FPL points).
3. **`rolling_avg_mins_last_5`**: Starter security metric (5-game rolling average minutes played).
4. **`rolling_avg_xG_last_5`**: Short-term underlying goal threat (5-game rolling average expected goals).
5. **`rolling_avg_xA_last_5`**: Short-term underlying assist potential (5-game rolling average expected assists).
6. **`rolling_avg_xG_last_38`**: Long-term underlying quality anchor (38-game rolling average xG).
7. **`rolling_avg_bps_last_38`**: Long-term bonus points system anchor (38-game rolling average raw Opta BPS).
8. **`composite_opp_difficulty`**: Blended opponent index $(0 \text{ to } 1)$ combining prior-gameweek lagged league position ($GW - 1$) and cumulative season-long expected goals allowed ($xGA$).
9. **`was_home`**: Binary indicator for home vs. away venue advantage.

### Brainstormed / Potential Future Features
Additional features evaluated or planned for future iterations:

- **`team_xg_season_avg`**: Cumulative attacking power and chance-creation volume of the player's team.
- **`xg_per_90_last_38`**: Normalized per-90 rate of underlying expected goals to account for rotation and substitutions.
- **`cost_last_1` / Price Tier**: Prior-week FPL market price as an external consensus proxy for baseline player talent.
- **`rolling_avg_bps_last_5`**: Short-term match involvement and bonus magnet velocity.
- **Set Piece & Penalty Duty Flags**: Indicator for primary penalty and corner/free-kick takers.

---

## Repository Structure

```text
├── Models/
│   ├── FWD_model.py          # Active forward point prediction & walk-forward backtest
│   ├── Test_FWD_model.py     # Experimental forward model testbed
│   └── ModelData/            # Aggregated, processed player and team match data
│       ├── playerData.csv    # Processed multi-season player statistics
│       └── teamData.csv      # Match fixtures, results, standings, and xG/xGA
├── scripts/
│   ├── build_player_data.py  # Data extraction & aggregation pipeline
│   ├── build_team_data.py    # Team match stats and standings processor
│   ├── scrape_fbref.py       # Scraper for advanced underlying statistics
│   └── fpl_api.py            # FPL API integration
├── requirements.txt          # Project dependencies
├── .gitignore                # Git ignore configuration
├── PROJECT_LOG.md            # Comprehensive project log & experiment notes
└── README.md                 # Project documentation
```

---

## Data Sources & Acknowledgments

This project utilizes historical Fantasy Premier League and Premier League underlying match data:

- **Primary FPL Data**: Sourced from [Vaastav Anand's Fantasy-Premier-League repository](https://github.com/vaastav/Fantasy-Premier-League), which compiles detailed matchday and gameweek CSVs for every Premier League season.
- **Underlying Advanced Metrics**: xG, xA, and team metrics sourced from [FBref](https://fbref.com/) via Understat and FPL API endpoints.

> *Note: Raw match CSVs (`RawData/`) are excluded from version control to maintain a lightweight repository. You can generate or refresh the dataset using the scripts in the `scripts/` directory.*

---

## Getting Started

### 1. Installation
Clone the repository and install required dependencies:

```bash
git clone https://github.com/<your-username>/FPL-Model.git
cd FPL-Model
pip install -r requirements.txt
```

### 2. Run Forward Prediction & Backtest
Execute the walk-forward backtesting pipeline and generate Gameweek 1 predictions:

```bash
cd Models
python3 FWD_model.py
```

---

## Evaluation Metrics

I am actively testing and comparing several evaluation metrics to measure real-world performance:

- **Spearman Rank Correlation**: Evaluates relative player ranking accuracy across all active starters.
- **NDCG & NDCG@5**: Measures the ranking quality of the highest projected recommendations.
- **⭐ Top 5 Picks Average Points**: Direct comparison of actual weekly points scored by Model selections vs. 5-GW Form Baseline vs. Most Popular FPL consensus.
- **Active Metrics Research**: Actively testing Top-K precision, intra-bucket elite discrimination (Top 20% vs Bottom 80%), and full integer-programming squad point simulations.
