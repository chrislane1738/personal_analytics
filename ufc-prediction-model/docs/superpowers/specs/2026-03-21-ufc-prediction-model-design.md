# UFC Fight Prediction Model — Design Specification

**Date:** 2026-03-21
**Status:** Approved
**Goal:** Build a machine learning model that predicts UFC fight outcomes with high accuracy, focused on explainability, with a polished Next.js dashboard for visualization.

---

## 1. Objectives & Success Criteria

**Primary:** Prediction accuracy — maximize correct winner predictions across all UFC fights, with particular strength on prelim/early-prelim fights where fundamental stats are more predictive than "name value."

**Secondary:** Fight analysis / explainability — the model must explain *why* it predicts a winner (style matchups, reach advantage, momentum, etc.), not just output a probability.

**Tertiary:** Value betting — naturally emerges when accuracy + explainability are strong. The model-without-odds variant enables comparison against market consensus.

### Non-Goals (Explicit)
- Method of victory prediction (KO/Sub/Decision) — future enhancement, not v1.
- Round-by-round simulation — out of scope.
- Live in-fight prediction — out of scope.

---

## 2. Data Strategy

### 2.1 Hybrid Approach (Three Layers)

| Layer | Source | Purpose | Update Cadence |
|-------|--------|---------|----------------|
| **Training data** | Kaggle CSVs (UFC 2025 Dataset, MMA Dataset 2023, UFC Rankings) | Baseline model training, 7,000+ fights, 95+ columns | Manual refresh quarterly |
| **Data refresh** | `ufcscraper` PyPI package (v1.1.0) | Scrapes ufcstats.com + bestfightodds.com, keeps training data current | Weekly or before events |
| **Upcoming fights** | `ufcapi.aristotle.me` REST API (100 req/day free) | Fighter lookup, upcoming cards, head-to-head data | On-demand before events |

### 2.2 Primary Kaggle Datasets

1. **UFC 2025 Dataset** (aminealibi) — Most current (Mar 2026), includes fighting style membership scores. CC BY 4.0.
2. **MMA Dataset 2023** (remypereira) — Relational structure (events → fights → fight_stats → fighters) with PKs/FKs. Ideal reference for time-aware feature engineering.
3. **UFC Rankings** (martj42) — Weekly rankings by weight class since 2013, CC0. Join by fighter name + date.
4. **Ultimate UFC Dataset** (mdabbert) — 5,900+ fights, 160 columns, includes betting odds. CC BY 4.0.

### 2.3 Data Pipeline Flow

```
Kaggle CSVs / ufcscraper output
    │
    ▼
Raw ingestion (pandas) → data/raw/
    │
    ▼
Cleaning & normalization → standardize names, parse dates, handle missing values
    │
    ▼
Time-aware feature aggregation (CRITICAL — see §3)
    │
    ▼
Feature-engineered dataset → data/processed/
    │
    ▼
Model training (LightGBM) → models/
    │
    ▼
Predictions JSON → data/predictions/
```

---

## 3. Feature Engineering

### 3.1 Data Leakage Prevention (Hard Rules)

These rules are non-negotiable:

1. **No same-fight stats.** Strikes landed, takedowns scored, control time, knockdowns *in the fight being predicted* must never appear as features.
2. **No outcome-derived features.** Winner, finish method, finish round, bonus awards from the target fight are forbidden.
3. **Time-aware aggregation only.** All career statistics (SLpM, TD avg, etc.) must be computed from fights *strictly before* the target fight date. No future information leakage.
4. **Knockdowns — career rate only.** Career knockdown rate (scored and absorbed) is included as a single feature among many. It is NOT weighted specially. Same-fight knockdowns are excluded per rule #1. Rationale: ~90% of UFC knockdowns lead to finishes, so in-fight knockdown data is essentially synonymous with the outcome.

### 3.2 Feature Categories (9 Selected)

All features are computed as **Fighter A value** and **Fighter B value**, plus a **differential** (A minus B).

#### 3.2.1 Physical Attributes
- Height (cm), reach (cm), weight (lbs), age at fight time, stance (orthodox/southpaw/switch)
- Derived: height_diff, reach_advantage, age_gap

#### 3.2.2 Career Record
- Total wins, losses, draws, win streak, loss streak, UFC fight count
- Derived: win_rate, finish_rate (KO% + Sub%), decision_rate, experience_differential

#### 3.2.3 Striking Metrics (Career Averages)
- Significant strikes landed per minute (SLpM), sig. strikes absorbed per minute (SApM), striking accuracy %, strike defense %
- Derived: strike_differential (SLpM - SApM), accuracy_advantage, volume_differential

#### 3.2.4 Grappling Metrics (Career Averages)
- Takedown avg per 15 min, takedown accuracy %, takedown defense %, submission avg per 15 min
- Derived: grappling_advantage (TD offense vs opponent TD defense), submission_threat_differential

#### 3.2.5 Recent Form & Activity
- Days since last fight, results of last 3 and last 5 fights, recent finish rate
- Derived: momentum_score (weighted recent results), ring_rust_indicator, activity_level

#### 3.2.6 Betting Odds (Model B Only)
- Opening odds, closing odds, implied probability, line movement
- Derived: market_confidence, odds_differential
- Note: Excluded from Model A to enable "beating the market" analysis

#### 3.2.7 UFC Rankings
- Fighter rank at fight time (0 if unranked), ranked_flag (boolean)
- Derived: rank_differential, stepping_up_indicator (unranked vs ranked)

#### 3.2.8 Fighting Style Classification (see §4 for full system)
- 4 primary archetype scores (0-1 fuzzy), sub-type scores for dominant archetypes only (threshold > 0.5)
- Derived: style_clash_type, stylistic_advantage_score

#### 3.2.9 Fight Context
- Weight class (categorical), rounds scheduled (3 or 5), title fight flag, card position (early prelim / prelim / main card)
- Derived: weight_class_experience, distance_fighter_indicator

### 3.3 Weight-Class-Aware Features

Feature importance varies by weight class. Rather than training separate models (insufficient data at heavyweight), weight class is included as a categorical feature in a single model. LightGBM naturally learns interaction effects (e.g., "age matters more at flyweight than heavyweight").

The model performance dashboard will show feature importance broken down by weight class to validate these interactions.

### 3.4 Estimated Feature Count

~40-50 total features per fight (raw + derived + differentials).

---

## 4. Fighting Style Classification System

### 4.1 Hierarchical Structure

Four primary archetypes, each with sub-types. Fighters get fuzzy membership scores (0-1) at both levels.

```
├── STRIKER (primary score 0-1)
│   ├── Power Puncher — high KO%, high career KD rate, low-moderate SLpM, early finishes
│   ├── Counter-Striker — high accuracy, very low SApM, high strike defense, strong differential
│   └── Pressure Fighter — very high SLpM, high SApM (trades), lower accuracy, consistent output
│
├── WRESTLER (primary score 0-1)
│   ├── Control Wrestler — very high control time, high TD accuracy, high decision win %
│   └── Ground & Pound — high TD avg, high ground strikes, KO/TKO from position
│
├── GRAPPLER (primary score 0-1)
│   ├── Submission Hunter — high sub attempts/15 min, high sub win %, active from bottom, reversals
│   └── Positional Grappler — high control time, moderate subs, methodical guard passing
│
└── BALANCED (primary score 0-1)
    ├── Adaptive — moderate all metrics, win methods spread evenly, switches ranges
    └── Defense-First — high TD defense, high strike defense, low SApM, wins on points
```

### 4.2 Computation Rules

- Primary scores computed from career stat ratios (striking volume vs TD volume vs sub attempts, etc.)
- Sub-type scores computed only when the parent primary score exceeds **0.5 threshold** (tunable during model training)
- Sub-scores below threshold default to 0 — prevents overfitting on noise
- Model receives all scores as flat features; dashboard displays as nested bar charts

### 4.3 Example Scoring

**Alex Pereira:** Striker=0.88 (Power=0.95, Counter=0.30, Pressure=0.55), Wrestler=0.10, Grappler=0.08, Balanced=0.12

**Khabib Nurmagomedov:** Wrestler=0.90 (GnP=0.85, Control=0.70), Grappler=0.75 (Positional=0.80, SubHunter=0.45), Striker=0.35, Balanced=0.40

---

## 5. Model Architecture

### 5.1 Algorithm: LightGBM

- Gradient boosted decision trees — fast training, handles mixed feature types (continuous + categorical), built-in feature importance
- Supports the explainability goal via SHAP (SHapley Additive exPlanations) values per prediction

### 5.2 Two Model Variants

| | Model A (No Odds) | Model B (With Odds) |
|---|---|---|
| **Features** | All except betting odds (~40 features) | All including betting odds (~45 features) |
| **Purpose** | Pure stats-based prediction, can "beat the market" | Maximum accuracy, benchmarks against market |
| **Use case** | Value betting analysis, true model intelligence | Overall accuracy leader |

Same architecture, same hyperparameters, same training data. Only the feature set differs.

### 5.3 Training & Validation

- **Target variable:** Binary — Fighter A wins (1) or Fighter B wins (0)
- **Validation:** Time-based cross-validation (train on pre-2024 fights, validate on 2024+). Never randomize — temporal ordering prevents future leakage.
- **Hyperparameter tuning:** Bayesian optimization (Optuna) over learning rate, max depth, num leaves, feature fraction, regularization
- **Evaluation metrics:** Accuracy, AUC-ROC, log loss, calibration curve (predicted probability should match actual win rate)

### 5.4 Explainability

- **SHAP values** computed per prediction → identifies which features drove the outcome
- Feeds the dashboard's "Key Decision Factors" display (e.g., "↑ Reach advantage +4" · Impact: High")
- **Global feature importance** chart on the Model Performance page — shows which features matter most overall and per weight class

---

## 6. Tech Stack

### 6.1 Backend (Python)

| Component | Technology |
|-----------|------------|
| Data processing | pandas, numpy |
| ML model | LightGBM |
| Explainability | SHAP |
| Hyperparameter tuning | Optuna |
| Data scraping | ufcscraper (PyPI), requests |
| API layer | FastAPI |
| Data validation | pydantic |

### 6.2 Frontend (Next.js)

| Component | Technology |
|-----------|------------|
| Framework | Next.js (App Router) |
| UI components | shadcn/ui |
| Styling | Tailwind CSS |
| Charts | Recharts or Nivo |
| Theme | Dark mode, UFC-themed accent colors (red/dark) |

### 6.3 No Database Required (v1)

- Model reads from CSV/Parquet files
- Predictions stored as JSON
- FastAPI serves predictions from memory/file
- Database can be added later if needed for user accounts, saved predictions, etc.

---

## 7. Dashboard Design

### 7.1 Pages

| Page | Route | Description |
|------|-------|-------------|
| **Upcoming Predictions** | `/` | Main page. Next UFC event's full fight card with predictions, filtered by card position (main/prelim/early prelim). Shows both model variants' predictions, confidence %, style matchup labels, key decision factors per fight. |
| **Head-to-Head** | `/compare` | Select any two fighters. Side-by-side stats, style radar charts, overlaid bar metrics, model prediction if they fought. |
| **Model Performance** | `/performance` | Accuracy over time (line chart by event), accuracy by weight class (bar chart), accuracy by card position, rolling accuracy %, feature importance chart, Model A vs Model B comparison. |
| **Fighter Profiles** | `/fighters/[id]` | Individual deep-dive: physical stats, style hierarchy bar chart, fight history timeline, career stat trends, strengths/weaknesses. |
| **Event Archive** | `/history` | Browse past events with predictions vs actual results. Filter by date range, weight class, correct/incorrect. Validation tool. |

### 7.2 Design Language

- **Dark mode** default — dark backgrounds (#0a0a0f, #0d1117), subtle borders (#1e2a3a)
- **Accent colors:** Red (#e94560) for UFC branding, green (#06d6a0) for strong predictions, yellow (#ffd166) for toss-ups
- **Style archetype colors:** Striker red (#e94560), Wrestler blue (#4cc9f0), Grappler purple (#7b2ff7), Balanced green (#06d6a0)
- **Confidence color coding:** Green (65%+) = strong pick, yellow (55-64%) = lean/toss-up
- **Typography:** Geist Sans for UI, Geist Mono for stats/numbers
- **Components:** shadcn/ui Card, Tabs, Badge, Progress bar, Charts

---

## 8. Project Structure

```
ufc-prediction-model/
├── backend/                        # Python ML pipeline
│   ├── data/
│   │   ├── raw/                    # Original CSVs from Kaggle/scraper
│   │   ├── processed/              # Feature-engineered datasets
│   │   └── predictions/            # Model output JSONs
│   ├── features/
│   │   ├── physical.py             # Physical attribute features
│   │   ├── record.py               # Career record features
│   │   ├── striking.py             # Striking metric features
│   │   ├── grappling.py            # Grappling metric features
│   │   ├── form.py                 # Recent form features
│   │   ├── odds.py                 # Betting odds features
│   │   ├── rankings.py             # UFC ranking features
│   │   ├── style.py                # Fighting style classification
│   │   ├── context.py              # Fight context features
│   │   └── pipeline.py             # Orchestrates all feature modules
│   ├── models/
│   │   ├── train.py                # Model training (both variants)
│   │   ├── evaluate.py             # Validation, metrics, calibration
│   │   ├── predict.py              # Generate predictions for upcoming fights
│   │   └── explain.py              # SHAP value computation
│   ├── scrapers/
│   │   ├── kaggle_loader.py        # Load and merge Kaggle datasets
│   │   ├── ufcstats_scraper.py     # Wrapper around ufcscraper
│   │   └── api_client.py           # ufcapi.aristotle.me REST client
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes/
│   │   │   ├── predictions.py      # Prediction endpoints
│   │   │   ├── fighters.py         # Fighter data endpoints
│   │   │   ├── events.py           # Event/history endpoints
│   │   │   └── model_stats.py      # Model performance endpoints
│   │   └── schemas.py              # Pydantic response models
│   ├── requirements.txt
│   └── config.py                   # Paths, thresholds, model params
├── frontend/                       # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx                # Upcoming Predictions (main)
│   │   ├── compare/page.tsx        # Head-to-Head
│   │   ├── performance/page.tsx    # Model Performance
│   │   ├── fighters/[id]/page.tsx  # Fighter Profile
│   │   └── history/page.tsx        # Event Archive
│   ├── components/
│   │   ├── ui/                     # shadcn/ui components
│   │   ├── fight-card.tsx          # Fight prediction card
│   │   ├── fighter-comparison.tsx  # Side-by-side view
│   │   ├── style-chart.tsx         # Style hierarchy bars
│   │   ├── accuracy-chart.tsx      # Model performance charts
│   │   └── feature-importance.tsx  # SHAP visualization
│   ├── lib/
│   │   ├── api.ts                  # FastAPI client
│   │   ├── types.ts                # TypeScript interfaces
│   │   └── utils.ts                # Formatting, color helpers
│   ├── package.json
│   └── tailwind.config.ts
├── docs/
│   └── superpowers/specs/          # This spec and future docs
├── .gitignore
└── README.md
```

---

## 9. Prelim/Early Prelim Focus

The model predicts ALL UFC fights, but the dashboard and analysis emphasize prelim/early prelim fights because:

- Less "name value" pressure on fighters — fundamentals (stats, style matchups) are more predictive
- Odds markets are thinner — more opportunity for the model to find value
- Card position is included as a feature so the model can learn this pattern naturally
- The dashboard defaults to the Prelims tab and shows card-position-filtered accuracy metrics

---

## 10. Future Enhancements (Out of Scope for v1)

- Method of victory prediction (KO/Sub/Decision)
- Live odds integration (real-time line movement)
- User accounts and saved predictions
- Notification system before events
- Historical backtesting simulator
- Weight class-specific model variants (if data grows)
