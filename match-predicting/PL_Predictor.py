# 1X2 predictor (improved): time-decayed Poisson (attack/defense + home) with DC low-score tweak,
# extra features, and hyperparameter tuning via time-based validation.
# Input CSV must have: date, team, opponent, venue, gf, ga, result
# Output CSV: predictions_poisson_1x2.csv with:
# date, home_team, away_team, xG_home, xG_away, P_home, P_draw, P_away, predicted_winner, actual_winner

from pathlib import Path
import os, math
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import PoissonRegressor

# ------------------
# CONFIG (edit these)
# ------------------
START_DATE = "2022-01-01"     # prediction window start
END_DATE   = "2024-12-31"     # prediction window end
CSV_NAME   = "matches.csv"

# Candidate grids for hyperparameter tuning (kept small for speed; expand if quiser)
ALPHA_GRID      = [1e-4, 5e-4, 1e-3, 5e-3]
HALF_LIFE_GRID  = [180.0, 270.0, 365.0, 540.0]  # days
RHO_GRID        = [-0.25, -0.15, -0.10, -0.05, 0.0]

# Probability truncation for goal sums
MAX_GOALS = 10

# Validation window length (days) BEFORE START_DATE
VAL_WINDOW_DAYS = 180

# ------------------
# 1) Read & hygiene
# ------------------
os.chdir(Path(__file__).resolve().parent)
df = pd.read_csv(CSV_NAME, index_col=0)

required = {"date","team","opponent","venue","gf","ga","result"}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing CSV columns: {sorted(missing)}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().any():
    raise ValueError("Invalid 'date' entries.")

# home flag
v = df["venue"].astype(str).str.lower()
df["is_home"] = np.where(v.str.startswith("h"), 1, 0)

# basic keep
df = df.dropna(subset=["team","opponent","date"]).copy()
df = df.sort_values("date").reset_index(drop=True)

# ------------------
# 2) No-leak features (computed with shift(1))
# ------------------
# Rest (days since last match per team)
df["rest_days"] = (
    df.groupby("team")["date"].diff().dt.days
)
# Fill first-match rest with team median or global median
team_rest_median = df.groupby("team")["rest_days"].transform(lambda s: s.fillna(s.median()))
df["rest_days"] = df["rest_days"].fillna(team_rest_median).fillna(df["rest_days"].median())
df["rest_log1p"] = np.log1p(df["rest_days"].clip(lower=0))

# Form (rolling means over last 3 matches)
df["form_for_3"]  = df.groupby("team")["gf"].shift(1).rolling(3, min_periods=1).mean()
df["form_against_3"] = df.groupby("team")["ga"].shift(1).rolling(3, min_periods=1).mean()

# Opponent’s rest/form on the SAME date (merge by (opponent,date))
opp_side = df[["team","date","rest_log1p","form_for_3","form_against_3"]].copy()
opp_side = opp_side.rename(columns={
    "team":"opponent",
    "rest_log1p":"opp_rest_log1p",
    "form_for_3":"opp_form_for_3",
    "form_against_3":"opp_form_against_3"
})
df = df.merge(opp_side, on=["opponent","date"], how="left")

# Fill any remaining NaNs (early season / missing opponent history)
for col in ["form_for_3","form_against_3","opp_form_for_3","opp_form_against_3","opp_rest_log1p"]:
    df[col] = df[col].fillna(df[col].median())

# Seasonality (month -> sin/cos)
month = df["date"].dt.month.astype(float)
df["m_sin"] = np.sin(2*np.pi*month/12.0)
df["m_cos"] = np.cos(2*np.pi*month/12.0)

# ------------------
# 3) Train / Predict split
# ------------------
start_dt = pd.Timestamp(START_DATE)
end_dt   = pd.Timestamp(END_DATE)
val_start = start_dt - pd.Timedelta(days=VAL_WINDOW_DAYS)

# Training data: everything before START_DATE
train_all = df[df["date"] < start_dt].copy()
# Validation split inside training (last VAL_WINDOW_DAYS before START_DATE)
inner_train = train_all[train_all["date"] < val_start].copy()
inner_val   = train_all[(train_all["date"] >= val_start) & (train_all["date"] < start_dt) & (train_all["is_home"]==1)].copy()

# Prediction rows (only home rows to avoid duplicates)
pred = df[(df["date"] >= start_dt) & (df["date"] <= end_dt) & (df["is_home"]==1)].copy()

if inner_train.empty or inner_val.empty or pred.empty:
    raise ValueError("Empty inner_train / inner_val / pred. Adjust dates or check CSV.")

# ------------------
# 4) Design matrices (one model for goals_for of the 'row team')
# ------------------
teams_all = sorted(pd.Index(pd.unique(df[["team","opponent"]].values.ravel())).tolist())

def oh(names, prefix):
    # fixed categories to align columns across sets
    cats = pd.Categorical(names, categories=teams_all)
    return pd.get_dummies(cats, prefix=prefix, drop_first=True)

CONT_FEATURES = [
    "is_home",
    "rest_log1p", "opp_rest_log1p",
    "form_for_3", "form_against_3",
    "opp_form_for_3", "opp_form_against_3",
    "m_sin","m_cos"
]

def build_X(df_in, drop_first=True):
    Xt = oh(df_in["team"],     "team")   # attack
    Xo = oh(df_in["opponent"], "opp")    # defense
    Xc = df_in[CONT_FEATURES].astype(float)
    X = np.hstack([Xc.to_numpy(), Xt.to_numpy(), Xo.to_numpy()])
    cols = list(Xc.columns) + list(Xt.columns) + list(Xo.columns)
    return X, cols

# Fit columns from inner_train
X_tr, COLS = build_X(inner_train, drop_first=True)
y_tr = inner_train["gf"].to_numpy()

# Helper to build X for any dataframe using same columns (no drop at inference)
def build_X_like(df_in):
    Xt = oh(df_in["team"],     "team")
    Xo = oh(df_in["opponent"], "opp")
    Xc = df_in[CONT_FEATURES].astype(float)
    X = np.hstack([Xc.to_numpy(), Xt.to_numpy(), Xo.to_numpy()])
    # Reindex to training columns
    Xdf = pd.DataFrame(X, columns=list(Xc.columns)+list(Xt.columns)+list(Xo.columns))
    Xdf = Xdf.reindex(columns=COLS, fill_value=0.0)
    return Xdf.to_numpy()

# ------------------
# 5) Probabilities (fix: home win = lower triangle; away win = upper triangle)
# ------------------
_factorials = np.array([math.factorial(i) for i in range(MAX_GOALS+1)], dtype=float)

def poisson_vec(lmb):
    ex = math.exp(-lmb)
    ks = np.arange(MAX_GOALS+1, dtype=float)
    return ex * (lmb ** ks) / _factorials

def rho_correction(i, j, lam_h, lam_a, rho):
    if   i==0 and j==0: return 1 - (lam_h * lam_a * rho)
    elif i==0 and j==1: return 1 + (lam_h * rho)
    elif i==1 and j==0: return 1 + (lam_a * rho)
    elif i==1 and j==1: return 1 - rho
    return 1.0

def outcome_probs(lam_home, lam_away, rho=0.0):
    ph = poisson_vec(lam_home)
    pa = poisson_vec(lam_away)
    mat = np.outer(ph, pa)  # P(H=i, A=j)

    # Dixon–Coles low-score tweak (optional when rho!=0)
    if rho != 0.0:
        mat[0,0] *= rho_correction(0,0,lam_home,lam_away,rho)
        mat[1,0] *= rho_correction(1,0,lam_home,lam_away,rho)
        mat[0,1] *= rho_correction(0,1,lam_home,lam_away,rho)
        mat[1,1] *= rho_correction(1,1,lam_home,lam_away,rho)
        s = mat.sum()
        if s > 0: mat /= s

    p_home = float(np.tril(mat, k=-1).sum())  # i > j  (FIX)
    p_draw = float(np.trace(mat))             # i = j
    p_away = float(np.triu(mat, k=+1).sum())  # i < j  (FIX)
    return p_home, p_draw, p_away

# ------------------
# 6) Weights (time-decay) and inference helpers
# ------------------
def make_weights(dates, half_life_ref_date, half_life_days):
    delta_days = (half_life_ref_date - dates).dt.days.clip(lower=0)
    return 0.5 ** (delta_days / float(half_life_days))

def predict_lambdas(model, row_home):
    """Given a home-row df (single row), produce (lam_home, lam_away)."""
    # home lambda
    Xh = build_X_like(row_home)
    lam_h = float(model.predict(Xh)[0])
    # away lambda: swap team/opponent; flip is_home, rest & form features accordingly
    away_row = row_home.copy()
    away_row = away_row.rename(columns={"team":"opponent","opponent":"team"})
    # swap continuous features for "team" and "opponent"
    away_row["is_home"] = 0
    away_row["rest_log1p"], away_row["opp_rest_log1p"] = row_home["opp_rest_log1p"].values, row_home["rest_log1p"].values
    away_row["form_for_3"], away_row["opp_form_for_3"] = row_home["opp_form_for_3"].values, row_home["form_for_3"].values
    away_row["form_against_3"], away_row["opp_form_against_3"] = row_home["opp_form_against_3"].values, row_home["form_against_3"].values
    Xa = build_X_like(away_row)
    lam_a = float(model.predict(Xa)[0])
    return lam_h, lam_a

def decide_winner(p_home, p_draw, p_away, home, away):
    return home if p_home >= max(p_draw, p_away) else (away if p_away >= max(p_home, p_draw) else "Draw")

# ------------------
# 7) Hyperparameter tuning on inner_train -> inner_val
# ------------------
best = None  # (acc, alpha, half_life, rho, model)

for alpha, half_life, rho in product(ALPHA_GRID, HALF_LIFE_GRID, RHO_GRID):
    # weights computed relative to validation start (to emulate "train -> validate")
    w_tr = make_weights(inner_train["date"], val_start, half_life)
    model = PoissonRegressor(alpha=alpha, max_iter=2000)
    Xtr = build_X_like(inner_train)
    ytr = inner_train["gf"].to_numpy()
    model.fit(Xtr, ytr, sample_weight=w_tr)

    # evaluate on inner_val (home rows only already)
    rows = []
    for _, r in inner_val.iterrows():
        row = r.to_frame().T  # single-row df
        lam_h, lam_a = predict_lambdas(model, row)
        p_h, p_d, p_a = outcome_probs(lam_h, lam_a, rho=rho)
        pred_winner = decide_winner(p_h, p_d, p_a, r["team"], r["opponent"])
        res = str(r.get("result","")).upper()
        actual = r["team"] if res=="W" else (r["opponent"] if res=="L" else "Draw")
        rows.append(pred_winner == actual)
    acc = float(np.mean(rows)) if rows else 0.0

    if (best is None) or (acc > best[0]):
        best = (acc, alpha, half_life, rho, model)

val_acc, BEST_ALPHA, BEST_HALF_LIFE, BEST_RHO, tuned_model = best
print(f"[TUNING] best_acc={val_acc:.4f} | alpha={BEST_ALPHA} | half_life_days={BEST_HALF_LIFE} | rho={BEST_RHO}")

# ------------------
# 8) Refit on FULL training (all data before START_DATE) with best alpha/half-life
# ------------------
w_full = make_weights(train_all["date"], start_dt, BEST_HALF_LIFE)
X_full = build_X_like(train_all)
y_full = train_all["gf"].to_numpy()

final_model = PoissonRegressor(alpha=BEST_ALPHA, max_iter=2000)
final_model.fit(X_full, y_full, sample_weight=w_full)

# ------------------
# 9) Predict window [START_DATE, END_DATE] (home rows only)
# ------------------
rows = []
for _, r in pred.iterrows():
    row = r.to_frame().T
    lam_h, lam_a = predict_lambdas(final_model, row)
    p_h, p_d, p_a = outcome_probs(lam_h, lam_a, rho=BEST_RHO)
    res = str(r.get("result","")).upper()
    actual = r["team"] if res=="W" else (r["opponent"] if res=="L" else ("Draw" if res=="D" else None))
    winner = decide_winner(p_h, p_d, p_a, r["team"], r["opponent"])
    rows.append({
        "date": r["date"].date(),
        "home_team": r["team"],
        "away_team": r["opponent"],
        "xG_home": round(lam_h, 3),
        "xG_away": round(lam_a, 3),
        "P_home": round(p_h, 4),
        "P_draw": round(p_d, 4),
        "P_away": round(p_a, 4),
        "predicted_winner": winner,
        "actual_winner": actual
    })

out = pd.DataFrame(rows).sort_values(["date","home_team","away_team"]).reset_index(drop=True)

# ------------------
# 10) Accuracy (1X2) on the window (when actual is available)
# ------------------
mask = out["actual_winner"].notna()
acc_1x2 = float((out.loc[mask, "predicted_winner"] == out.loc[mask, "actual_winner"]).mean()) if mask.any() else float("nan")
print(f"[{START_DATE}..{END_DATE}] 1X2 accuracy = {acc_1x2:.4f} (n={int(mask.sum())})")

print(out.head(12).to_string(index=False))
out.to_csv("predictions_poisson_1x2.csv", index=False)
print("Saved predictions_poisson_1x2.csv")
