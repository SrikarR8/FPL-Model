# ⚽ Fantasy Premier League (FPL) ML Prediction Model

An end-to-end Machine Learning model and backtesting framework designed to project Fantasy Premier League (FPL) player points, evaluate fixture difficulties, and rank top picks for upcoming gameweeks.

---

## 📊 Overview & Methodology

The model employs regularized **XGBoost Regressors with a Tweedie distribution objective** (`reg:tweedie`), designed specifically for zero-inflated, right-skewed point distributions common in fantasy football.

### Key Pipeline Features
- **38-Game Talent & Quality Anchors**: Full-season rolling baselines (`rolling_avg_points_last_38`, `rolling_avg_xG_last_38`, `rolling_avg_bps_last_38`) to capture true player class and prevent dropping elite talismans during short cold streaks.
- **Short-Term Form Modulation**: 5-game rolling averages (`points`, `mins`, `xG`, `xA`, `bps`) to capture current form and starting security.
- **Composite Opponent Difficulty Index**: Blends prior gameweek lagged league position ($GW - 1$) and cumulative season-long expected goals allowed ($xGA$) without look-ahead bias.
- **Walk-Forward Historical Backtesting**: Strict temporal walk-forward evaluation (simulating actual weekly decisions across GW 1 to 38) comparing Model Picks vs. 5-GW Form Baselines vs. Real Most Popular FPL crowd selections.

---

## 📁 Repository Structure

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
└── README.md                 # Project documentation
```

---

## 🌐 Data Sources & Acknowledgments

This project utilizes historical Fantasy Premier League and Premier League underlying match data.

- **Primary FPL Data**: Sourced from [Vaastav Anand's Fantasy-Premier-League repository](https://github.com/vaastav/Fantasy-Premier-League), which compiles detailed matchday and gameweek CSVs for every Premier League season.
- **Underlying Advanced Metrics**: xG, xA, and team metrics sourced from [FBref](https://fbref.com/) via Understat and FPL API endpoints.

> *Note: Raw match CSVs (`RawData/`) are excluded from version control to maintain a lightweight repository. You can generate or refresh the dataset using the scripts in the `scripts/` directory.*

---

## 🚀 Getting Started

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

## 📈 Evaluation Metrics

- **Spearman Rank Correlation**: Evaluates relative player ranking accuracy across all active starters.
- **NDCG & NDCG@5**: Measures quality of top recommendations.
- **Top 5 Picks Average Points**: Direct comparison of actual weekly points scored by Model selections vs. Form Baseline vs. Most Popular FPL consensus.
