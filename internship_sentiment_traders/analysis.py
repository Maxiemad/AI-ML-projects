from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = DATA / "hyperliquid_trades.csv"
    if not trades_path.exists():
        alt = DATA / "hyperliquid_trades.zip"
        if alt.exists() and open(alt, "rb").read(4) != b"PK\x03\x04":
            trades_path = alt
        elif alt.exists():
            raise SystemExit("Found hyperliquid_trades.zip as real zip — unpack to hyperliquid_trades.csv first.")
    if not trades_path.exists():
        raise SystemExit(f"Missing trade file under {DATA}")

    raw = pd.read_csv(trades_path)
    fg = pd.read_csv(DATA / "fear_greed.csv")
    return raw, fg


def realized_closes(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["Direction"].astype(str).str.contains(r"Close", case=False, na=False)].copy()
    d["ts"] = pd.to_datetime(d["Timestamp IST"], dayfirst=True, format="%d-%m-%Y %H:%M", errors="coerce")
    d["d"] = d["ts"].dt.normalize()
    d["hour"] = d["ts"].dt.hour
    d["dow"] = d["ts"].dt.dayofweek
    d["is_long_closez"] = d["Direction"].astype(str).str.contains("Close Long", case=False, na=False).astype(int)
    return d.dropna(subset=["ts"])


def attach_sentiment(closes: pd.DataFrame, fg: pd.DataFrame) -> pd.DataFrame:
    g = fg.copy()
    g["d"] = pd.to_datetime(g["date"]).dt.normalize()
    g = g.sort_values("d").reset_index(drop=True)
    g["fg_lag1"] = g["value"].shift(1)
    g["cls_lag1"] = g["classification"].shift(1)
    cal = g[["d", "value", "classification", "fg_lag1", "cls_lag1"]].rename(
        columns={"value": "fg0", "classification": "cls0"}
    )
    m = closes.merge(cal, on="d", how="inner")

    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    rank = {c: i for i, c in enumerate(order)}
    m["fg_bucket"] = pd.Categorical(m["cls_lag1"], categories=order, ordered=True)

    def norm_coin(x: str) -> str:
        s = str(x)
        if s.startswith("@"):
            return "ALT_INDEX"
        return s[:12]

    m["coin_g"] = m["Coin"].map(norm_coin)
    return m


def bucket_ci(x: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(x))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        boots.append(np.median(s))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def fig_pnl_by_sentiment(df: pd.DataFrame) -> None:
    rows = []
    for b in df["fg_bucket"].cat.categories:
        sub = df.loc[df["cls_lag1"] == b, "Closed PnL"]
        if len(sub) < 30:
            continue
        med, lo, hi = bucket_ci(sub.values)
        rows.append({"bucket": b, "median_pnl": med, "lo": lo, "hi": hi, "n": int(len(sub))})
    p = pd.DataFrame(rows)
    if p.empty:
        return

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(p))
    ax.bar(x, p["median_pnl"], color="#2c5282", alpha=0.85)
    err = np.vstack([p["median_pnl"] - p["lo"], p["hi"] - p["median_pnl"]])
    ax.errorbar(x, p["median_pnl"], yerr=err, fmt="none", ecolor="#1a202c", capsize=3, alpha=0.7)
    ax.axhline(0, color="#742a2a", lw=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(p["bucket"], rotation=18, ha="right")
    ax.set_ylabel("Median realized PnL per close (bootstrap CI)")
    ax.set_title("Realized PnL vs prior-day Fear & Greed regime (closing trades)")
    fig.tight_layout()
    fig.savefig(OUT / "pnl_by_sentiment.png", dpi=160)
    plt.close(fig)


def trader_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    def lab(row):
        cls = row["cls_lag1"]
        if cls in ("Extreme Fear", "Fear"):
            return "fearish"
        if cls in ("Extreme Greed", "Greed"):
            return "greedish"
        return np.nan

    tmp = df.copy()
    tmp["polar"] = tmp.apply(lab, axis=1)
    agg = (
        tmp.dropna(subset=["polar"])
        .groupby(["Account", "polar"], as_index=False)["Closed PnL"]
        .agg(mean_pnl="mean", n="count")
    )
    wide = agg.pivot(index="Account", columns="polar", values="mean_pnl").rename(
        columns={"fearish": "mu_fear", "greedish": "mu_greed"}
    )
    cnt = df.groupby("Account").size().rename("n_close")
    wide = wide.join(cnt, how="inner")
    wide = wide.dropna(subset=["mu_fear", "mu_greed"])
    wide = wide[wide["n_close"] >= 12]

    if len(wide) < 10:
        return wide.assign(cluster=-1)

    X = wide[["mu_fear", "mu_greed"]].values
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    k = 3 if len(wide) >= 40 else 2
    gm = GaussianMixture(n_components=k, random_state=7, covariance_type="full")
    wide = wide.copy()
    wide["cluster"] = gm.fit_predict(Xs)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    pal = sns.color_palette("Set2", max(k, 3))
    for c in sorted(wide["cluster"].unique()):
        s = wide[wide["cluster"] == c]
        ax.scatter(s["mu_fear"], s["mu_greed"], s=18 + np.log1p(s["n_close"]), alpha=0.55, label=f"mixture {c}", color=pal[c % len(pal)])
    ax.axhline(0, color="#4a5568", lw=0.8)
    ax.axvline(0, color="#4a5568", lw=0.8)
    ax.set_xlabel("Mean PnL on fearish days (lagged FG)")
    ax.set_ylabel("Mean PnL on greedish days (lagged FG)")
    ax.set_title("Trader archetypes (Gaussian mixture on fear vs greed means)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "trader_archetypes.png", dpi=160)
    plt.close(fig)

    return wide.reset_index()


def stratified_contrarian(df: pd.DataFrame) -> dict:
    out = {}
    for b in ["Extreme Fear", "Extreme Greed"]:
        sub = df[df["cls_lag1"] == b]
        if len(sub) < 80:
            continue
        longs = sub[sub["is_long_closez"] == 1]["Closed PnL"]
        shorts = sub[sub["is_long_closez"] == 0]["Closed PnL"]
        stat, p = stats.mannwhitneyu(longs, shorts, alternative="two-sided")
        out[b] = {
            "n_long": int(len(longs)),
            "n_short": int(len(shorts)),
            "median_long": float(np.median(longs)),
            "median_short": float(np.median(shorts)),
            "mannwhitney_p": float(p),
        }
    return out


def ols_cluster_sentiment(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["y"] = np.sign(d["Closed PnL"]) * np.log1p(np.abs(d["Closed PnL"].clip(lower=0)))
    d["log_notional"] = np.log1p(d["Size USD"].clip(lower=0))
    d["long"] = d["is_long_closez"]

    X = d[["fg_lag1", "log_notional", "long"]].copy()
    X = sm.add_constant(X)
    y = d["y"].astype(float)
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["Account"].astype("category").cat.codes})

    d["fg_long"] = d["fg_lag1"] * d["long"]
    X2 = sm.add_constant(d[["fg_lag1", "long", "fg_long", "log_notional"]])
    ols_i = sm.OLS(y, X2).fit(cov_type="cluster", cov_kwds={"groups": d["Account"].astype("category").cat.codes})

    return {
        "n": int(len(d)),
        "r2": float(ols.rsquared),
        "coef_fg_lag1": float(ols.params.get("fg_lag1", float("nan"))),
        "p_fg_lag1": float(ols.pvalues.get("fg_lag1", float("nan"))),
        "coef_fg_x_long": float(ols_i.params.get("fg_long", float("nan"))),
        "p_fg_x_long": float(ols_i.pvalues.get("fg_long", float("nan"))),
    }


def build_lgbm_matrix(df: pd.DataFrame):
    d = df.copy()
    d["y_dir"] = (d["Closed PnL"] > 0).astype(int)
    d["long"] = d["is_long_closez"]

    le = LabelEncoder()
    d["coin_id"] = le.fit_transform(d["coin_g"].astype(str))

    feat_cols = ["fg_lag1", "fg0", "hour", "dow", "log_notional", "long", "coin_id"]
    d["log_notional"] = np.log1p(d["Size USD"].clip(lower=0))

    cutoff = d["d"].quantile(0.8)
    train = d[d["d"] <= cutoff]
    test = d[d["d"] > cutoff]

    X_tr, y_tr = train[feat_cols], train["y_dir"]
    X_te, y_te = test[feat_cols], test["y_dir"]
    return X_tr, X_te, y_tr, y_te, feat_cols


def _binary_report(y_true: np.ndarray, proba: np.ndarray, thresh: float) -> dict:
    y_pred = (proba >= thresh).astype(np.int8)
    return {
        "threshold": float(thresh),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, y_pred).ravel().tolist(),
    }


def lgbm_direction(X_tr, X_te, y_tr, y_te, feat_cols) -> dict:
    train = lgb.Dataset(X_tr, label=y_tr)
    valid = lgb.Dataset(X_te, label=y_te, reference=train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 3.0,
        "max_depth": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train,
        num_boost_round=400,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    pred_tr = booster.predict(X_tr, num_iteration=booster.best_iteration)
    t0 = time.perf_counter()
    pred = booster.predict(X_te, num_iteration=booster.best_iteration)
    dt = time.perf_counter() - t0

    y_te_np = np.asarray(y_te)
    p_clip = np.clip(pred, 1e-7, 1.0 - 1e-7)

    maj = int(pd.Series(y_tr).mode().iloc[0])
    baseline_maj = float(accuracy_score(y_te_np, np.full(len(y_te_np), maj)))
    always_positive_acc = float(accuracy_score(y_te_np, np.ones(len(y_te_np), dtype=int)))

    pr_baseline = float(np.mean(y_te_np))
    holdout = {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "split_rule": "by trade-date: earliest 80pct train, latest 20pct test",
        "positive_rate_train": float(pd.Series(y_tr).mean()),
        "positive_rate_test": pr_baseline,
        "auc_roc": float(roc_auc_score(y_te_np, pred)),
        "auc_pr": float(average_precision_score(y_te_np, pred)),
        "auc_pr_random_guess": pr_baseline,
        "log_loss": float(log_loss(y_te_np, p_clip)),
        "baseline_majority_class_accuracy": baseline_maj,
        "baseline_predict_always_positive_accuracy": always_positive_acc,
        "metrics_prob_threshold_0_5": _binary_report(y_te_np, pred, 0.5),
    }
    best_t, best_f1_tr = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1tr = f1_score(y_tr, (pred_tr >= t).astype(int), zero_division=0)
        if f1tr > best_f1_tr:
            best_f1_tr, best_t = float(f1tr), float(t)
    holdout["threshold_f1_max_on_train"] = best_t
    holdout["train_f1_at_that_threshold"] = best_f1_tr
    holdout["metrics_at_train_f1_threshold"] = _binary_report(y_te_np, pred, best_t)

    imp = pd.Series(booster.feature_importance(importance_type="gain"), index=feat_cols).sort_values(ascending=False)

    return {
        "auc_holdout": holdout["auc_roc"],
        "best_iter": int(booster.best_iteration),
        "importance": imp.to_dict(),
        "holdout_evaluation": holdout,
        "efficiency": {
            "predict_wall_ms": round(dt * 1000, 4),
            "test_rows": int(len(X_te)),
            "rows_per_second": round(len(X_te) / dt, 1) if dt > 0 else None,
            "num_features": len(feat_cols),
        },
    }


def permute_fg_null(df: pd.DataFrame, n_perm: int = 30, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    d = df[["d", "fg_lag1", "Closed PnL", "Size USD", "is_long_closez", "Account"]].copy()
    d["y"] = np.sign(d["Closed PnL"]) * np.log1p(np.abs(d["Closed PnL"].clip(lower=0)))
    d["log_notional"] = np.log1p(d["Size USD"].clip(lower=0))
    d["long"] = d["is_long_closez"]

    days = d["d"].unique()
    true_map = d.groupby("d")["fg_lag1"].first().to_dict()

    def r2_for_map(mp: dict) -> float:
        fg = d["d"].map(mp).astype(float)
        X = np.column_stack([np.ones(len(d)), fg, d["log_notional"].values, d["long"].values])
        y = d["y"].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum())

    obs = r2_for_map(true_map)
    pool = list(true_map.values())
    synth = []
    for _ in range(n_perm):
        shuffled = rng.permuted(pool)
        mp = {day: shuffled[i] for i, day in enumerate(days)}
        synth.append(r2_for_map(mp))
    synth = np.asarray(synth, float)
    pval = (1 + (synth >= obs).sum()) / (1 + n_perm)
    return {"r2_observed": obs, "r2_perm_mean": float(synth.mean()), "p_approx": float(pval), "n_perm": n_perm}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    raw, fg = load_frames()
    closes = realized_closes(raw)
    merged = attach_sentiment(closes, fg)
    merged = merged[np.isfinite(merged["fg_lag1"])].copy()

    summary = {
        "n_close_rows": int(len(closes)),
        "n_merged": int(len(merged)),
        "date_span": [str(merged["d"].min().date()), str(merged["d"].max().date())],
        "spearman_fg_vs_pnl": float(
            stats.spearmanr(merged["fg_lag1"], merged["Closed PnL"], nan_policy="omit").correlation
        ),
    }

    fig_pnl_by_sentiment(merged)
    arch = trader_archetypes(merged)
    if "cluster" in arch.columns and (arch["cluster"].to_numpy() >= 0).any():
        summary["archetype_n_traders"] = int(len(arch))
        summary["archetype_by_cluster"] = arch.groupby("cluster", dropna=False).size().astype(int).to_dict()
    summary["contrarian_mwu"] = stratified_contrarian(merged)
    summary["ols_cluster"] = ols_cluster_sentiment(merged)

    X_tr, X_te, y_tr, y_te, cols = build_lgbm_matrix(merged)
    summary["lgbm"] = lgbm_direction(X_tr, X_te, y_tr, y_te, cols)
    summary["perm_null_r2"] = permute_fg_null(merged, n_perm=40)

    top = pd.Series(summary["lgbm"]["importance"]).sort_values(ascending=False).head(6)
    summary["lgbm"]["importance_top"] = top.to_dict()

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
