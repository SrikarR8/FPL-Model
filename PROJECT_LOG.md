# FPL Prediction Model — Project Log

Running log of bugs found, fixes applied, and modeling decisions made during development.
Kept chronologically so the reasoning behind each change is traceable.

---

## Data Pipeline

### Bug: Missing chronological sort before rolling-window calculations
`groupby().transform()` with `.shift()`/`.rolling()` depends entirely on row order.
`df_combined` was built via `pd.concat([df_prev, df_current])` without a `sort_values()`
call afterward — rolling averages were being computed over whatever order rows happened
to land in after concat, not true chronological order (season → gameweek).
**Fix:** `df_combined.sort_values(['name'/'element', 'season_id', 'GW'])` before any
rolling feature calculation.

### Bug: Player identity keyed on `name` instead of a stable ID
Grouping/merging on the `name` column is fragile — formatting can differ subtly across
season files (whitespace, accents, structure), which silently splits a single player
into two groups or fails a merge across seasons.
**Fix / ongoing:** prefer a stable identifier (`code` in `players_raw.csv`, which persists
across seasons) over `element`/`id` (season-specific) or `name` (inconsistent formatting)
wherever joining across seasons.

### Bug: `xP` column contains potential lookahead bias
`xP` in `merged_gw.csv` is sourced from FPL's `ep_this` field via a post-gameweek scraper.
Empirical check showed scraped `xP` correlates unusually strongly (~0.40) with same-gameweek
actual points — too strong for a genuinely pre-match prediction — suggesting the scraped
value sometimes reflects a post-match-updated `ep_this`, not the pre-deadline value.
**Fix:** dropped `xP` from training features entirely rather than trying to `shift(1)` an
ambiguously-timed column. Live `ep_next` (pulled directly pre-deadline during the season)
used instead as a comparison baseline, not as a training feature.

### Bug/gap: `xP` missing for ~25/38 gameweeks for players with clean data otherwise (e.g. Saka)
Traced to the field's availability differing across seasons (older season files predate
consistent `ep_this` scraping). Confirms `xP` isn't reliable enough to keep even after
addressing the leakage concern above — reinforces the decision to drop it.

### Bug: Fixture Difficulty Rating (FDR) of 1 missing/miscoded in some historical seasons
~2,000 rows in 2024-25 and 2025-26 season files correctly show FDR=1, but earlier
combined-season data showed zero instances of FDR=1 at all — a distributional
inconsistency across seasons (likely a scraping/merge artifact or scale recalibration),
not a real absence of easy fixtures in earlier years.
**Fix:** corrected FDR merge/encoding across all seasons. Result: `fixture_difficulty`
importance jumped from mid-pack to 3rd-highest feature, and this was a major contributor
to the Spearman correlation improvement (~0.166 → ~0.19).

### Data gap: `chance_of_playing_next_round` and `can_select` not available historically
These are live/current-snapshot fields in `players_raw.csv` / `bootstrap-static`, not
stored per-gameweek historically.
**Decision:** defaulted to `chance_of_playing_next_round = 100` and `can_select = True`
for all historical training rows (reasonable — most player-gameweeks are full-fitness).
Live values pulled fresh from the API at actual in-season prediction time, where the
feature has real value (downgrading doubtful/injured players).

---

## Feature Engineering

### `fixture_difficulty` sourced correctly (home vs away split)
`fixtures.csv` has independent `team_h_difficulty` / `team_a_difficulty` per fixture
(asymmetric — same match can be "easy" for one side, "hard" for the other). Verified
correct side is picked based on whether the player's team was home or away (confirmed
manually against Saka's Arsenal fixtures: Wolves (H) → 1, Man City → 5).

### Rolling averages built manually — not present in source data
`merged_gw.csv` has raw per-gameweek stats only. Rolling features (`rolling_avg_points_last_5`,
`rolling_avg_mins_last_5`, `rolling_avg_xG_last_5`, `rolling_avg_xA_last_5`) computed via
`groupby(player_id)[stat].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())`.
`.shift(1)` is critical — omitting it leaks the current gameweek's own result into its
own "rolling average" feature.

---

## Model Evaluation

### Bug: RMSE identified as a poor primary metric for this problem
FPL points have high inherent randomness (deflections, red cards, missed penalties).
RMSE's squared-error penalty over-punishes these largely unpredictable events and can
push a model toward overly conservative, mean-reverting predictions.
**Decision:** primary metrics are MAE (vs. rolling-average baseline) and Spearman rank
correlation, since squad selection depends on relative ranking more than exact point
values. Top-K precision and backtested squad points (actual points scored by the
optimizer's chosen squad) identified as the most direct measures of real-world value,
to be added once the optimizer stage is built.

### Bug: Model appeared to underperform a simple rolling-average baseline (0.166 vs 0.219)
Initial Spearman comparison seemed to show XGBoost losing to the naive baseline.
**Root cause:** the two numbers weren't computed on the same data — baseline was
evaluated across all combined seasons, model was evaluated only on the held-out walk-forward
test set. Not a fair comparison.
**Fix:** recomputed both on the identical held-out test rows. Corrected result:
model 0.193 vs baseline 0.172 — model genuinely ahead, as expected.
**Lesson:** always confirm both sides of a comparison are computed on exactly the same
rows before drawing conclusions from a metric gap.

### Diagnosed: raw feature importance was misleading for FWD position
`rolling_avg_mins_last_5` dominated overall gain (~726 vs next-highest ~108) when
importance was computed across all FWD rows, including low-minute bench players.
xG/xA appeared to contribute almost nothing.
**Root cause:** for players who barely play, `total_points` is near-zero almost
regardless of underlying quality — minutes alone nearly determines the outcome for
that subset, swamping the importance of skill-based features.
**Fix:** filtered to `minutes >= 40` (meaningful-minutes rows only) before evaluating
feature importance. Result: importance rebalanced substantially — xG (0.102) and
xA (0.135) became comparable in magnitude to points/minutes, confirming they were
being diluted by bench-player noise, not genuinely weak signals.
**Follow-up idea (not yet built):** two-stage model — (1) predict probability of
meaningful minutes, (2) predict points conditional on playing — to avoid conflating
these two different questions in one set of trees. Likely most valuable for FWD/GK
(boom-bust positions); may matter less for DEF/MID.

### Investigated: `new_cost` SHAP pattern (dense cluster near zero + far outlying high-impact points)
Distribution matches FPL's real price skew (most players clustered £4-7.5m, small
number of premium players £9m+). Confirmed via row count: 17 distinct players above
£9.0m in the FWD dataset. Tree model has learned a threshold-like "elite tier" effect
rather than a smooth linear relationship — plausible given real football (elite players
are a genuinely distinct tier), but noted as a limitation: model may extrapolate
imperfectly for new players entering that price tier without matching historical
performance (cold-start-adjacent risk).
**Decision:** kept `new_cost` as-is (no log-transform/binning) — judged the underlying
"price reflects quality" relationship as real signal, not an artifact worth correcting
for MVP.

### Fixed: overfitting after initial full-feature model underperformed baseline
Suspected default hyperparameters (unconstrained depth, no regularization) were
overfitting on the position-filtered (smaller) FWD training set.
**Fix:** reduced `max_depth`, added `subsample`/`colsample_bytree`/`min_child_weight`
regularization, used early stopping. Combined with the FDR bug fix above, contributed
to the 0.166 → 0.19 improvement.

### Hyperparameter search: 96-combination grid search, run on corrected (post-FDR-fix) data
Result: Spearman correlation improved 0.193 → 0.215 (walk-forward, FWD, GW20-38 of
2025-26, held-out test set, same rows used for both model and baseline comparison).
Confirmed this gain holds on the corrected dataset (not just compensating for the
now-fixed FDR bug).

---

## Validation Methodology

### Walk-forward backtest implemented (not a single static train/test split)
For each gameweek 20-38 of the target season: train on all prior seasons + current
season up to `gw - 3`, use `gw-3` to `gw` as early-stopping validation, predict on `gw`.
Mirrors actual deployment (predicting one gameweek ahead using only information
available at that point) far more closely than a single fixed split.

### Confirmed no train/val row overlap
`X_train.index.intersection(X_val.index)` checked empty before trusting any
train-vs-validation comparison — guards against silently evaluating on rows the
model already trained on.

---

## Open Items / Not Yet Done

- [ ] Confirm hyperparameter search used walk-forward structure (not a single split) —
      risk of having tuned to one validation window rather than generalizable params
- [ ] Run same importance/SHAP diagnostic process on GK, DEF, MID (currently FWD-only)
- [ ] Decide whether two-stage (minutes-probability + conditional-points) model is
      worth building generally, or only for FWD/GK
- [ ] Build PuLP integer linear program for squad selection (budget, position counts,
      max-3-per-club constraints)
- [ ] Backtest actual squad points from model + optimizer combined, vs. baseline squad
      — the real end-to-end validation metric
- [ ] Handle new-signing cold start (no FPL history) — planned fallback: flag low-history
      players, use price-based heuristic or cross-league stats as substitute features
- [ ] Weekly transfer-decision layer (in-season, once initial squad + optimizer work)
