"""ML layer.

1. blunder-risk model  -> P(blunder) for any position-state, + which factors drive it
2. rating-gain simulator -> how much Elo each fixable leak is worth
3. per-game rating delta model -> what actually correlates with winning

Usage: python model.py
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURES = [
    "move_no", "phase", "color", "time_class", "clock_left", "time_spent",
    "material_diff", "my_hanging_before", "eval_before", "rating_diff",
    "in_check_before", "was_capture", "opp_last_was_capture", "hour",
]
CAT = ["phase", "color", "time_class"]


LEAKS = {
    "blunders under 30s on the clock": lambda d: (d.clock_left < 30) & d.blunder,
    "blunders in the opening (<=move 12)": lambda d: (d.phase == "opening") & d.blunder,
    "blunders in the middlegame": lambda d: (d.phase == "middlegame") & d.blunder,
    "hanging a piece outright": lambda d: d.motif.fillna("").str.startswith("hangs_") & d.blunder,
    "blunders after opponent captured": lambda d: d.opp_last_was_capture & d.blunder,
}


def leak_removed_counts(df, per_game):
    """Per-game count of blunders matching each leak, aligned to per_game's row order."""
    return {name: df[mask_fn(df)].groupby("game_url").size().reindex(per_game.index).fillna(0)
            for name, mask_fn in LEAKS.items()}


def leak_elo(per_game, removed_counts):
    """Remove each leak's blunders from every game, recompute win rate off the
    empirical blunders-in-game -> win-rate curve, convert the shift to Elo."""
    wp = per_game.groupby(per_game.blunders.clip(upper=5)).won.mean()
    base = per_game.won.mean()
    out = {}
    for name, removed in removed_counts.items():
        adj = (per_game.blunders - removed).clip(lower=0)
        proj = wp.reindex(adj.clip(upper=5)).mean()
        d_wp = proj - base
        elo = 400 * np.log10(max(base + d_wp, .01) / max(1 - base - d_wp, .01)) - \
              400 * np.log10(max(base, .01) / max(1 - base, .01))
        out[name] = (d_wp, elo)
    return out


def bootstrap_leak_ci(per_game, removed_counts, n_boot=2000, seed=0):
    """95% percentile CI on each leak's Elo estimate, resampling games with replacement.

    The point estimate alone reads as more precise than it is -- some of these
    leaks move very few games, and the win-rate-by-blunder-count curve itself is
    thin in spots. This puts a number on that uncertainty instead of hiding it.
    """
    rng = np.random.default_rng(seed)
    n = len(per_game)
    elos = {name: [] for name in removed_counts}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pg_b = per_game.iloc[idx].reset_index(drop=True)
        removed_b = {name: r.iloc[idx].reset_index(drop=True) for name, r in removed_counts.items()}
        for name, (_, elo) in leak_elo(pg_b, removed_b).items():
            elos[name].append(elo)
    return {name: np.percentile(vals, [2.5, 97.5]) for name, vals in elos.items()}


def prep(df):
    X = df[FEATURES].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/moves.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.path).dropna(subset=["clock_left"])
    X, y, groups = prep(df), df.blunder.astype(int), df.game_url

    cat_mask = [c in CAT for c in X.columns]
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask, max_iter=300, learning_rate=0.06,
        max_leaf_nodes=15, l2_regularization=1.0, random_state=0)

    cv = GroupKFold(n_splits=5)
    p = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    print(f"blunder base rate {y.mean()*100:.1f}%")
    print(f"AUC {roc_auc_score(y, p):.3f} | Brier {brier_score_loss(y, p):.4f}\n")

    clf.fit(X, y)
    imp = permutation_importance(clf, X, y, n_repeats=8, random_state=0,
                                 scoring="roc_auc", n_jobs=-1)
    ranked = (pd.Series(imp.importances_mean, index=X.columns)
              .sort_values(ascending=False).head(10))
    print("WHAT DRIVES YOUR BLUNDERS (permutation importance on AUC)")
    print(ranked.round(4).to_string(), "\n")

    # highest-risk situations you actually land in often
    df = df.assign(risk=p)
    print("YOUR 8 HIGHEST-RISK RECURRING SITUATIONS")
    buckets = df.groupby(["phase", "time_class",
                          pd.cut(df.clock_left, [0, 30, 120, 1e9],
                                 labels=["<30s", "30s-2m", "2m+"])],
                         observed=True).agg(
        moves=("risk", "size"), pred_risk=("risk", "mean"), actual=("blunder", "mean"))
    buckets = buckets[buckets.moves >= 50].sort_values("actual", ascending=False)
    print((buckets.assign(pred_risk=lambda d: (d.pred_risk*100).round(1),
                          actual=lambda d: (d.actual*100).round(1))).head(8), "\n")

    # --- what a fix is worth ---
    print("=" * 60)
    print("LEAK VALUE — Elo-ish impact of removing each leak")
    per_game = df.groupby("game_url").agg(
        acpl=("cpl", "mean"), blunders=("blunder", "sum"),
        result=("result", "first"), rating=("my_rating", "first"))
    per_game["won"] = (per_game.result == "win").astype(int)
    # empirical: win prob vs blunder count
    wp = per_game.groupby(per_game.blunders.clip(upper=5)).won.mean()
    print("\nwin rate by blunders in the game:")
    print((wp * 100).round(1).to_string())

    removed_counts = leak_removed_counts(df, per_game)
    point = leak_elo(per_game, removed_counts)
    ci = bootstrap_leak_ci(per_game, removed_counts)
    ranked = sorted(point.items(), key=lambda kv: kv[1][1], reverse=True)
    for name, (d_wp, elo) in ranked:
        lo, hi = ci[name]
        n_moves = int(LEAKS[name](df).sum())
        print(f"  fix '{name}': +{d_wp*100:.1f} win% ≈ +{elo:.0f} Elo "
              f"(95% CI {lo:.0f} to {hi:.0f})  ({n_moves} moves)")

    # --- rating trajectory ---
    print("\n" + "=" * 60)
    ts = (df.sort_values("end_time").groupby("game_url")
            .agg(t=("end_time", "first"), r=("my_rating", "first")).sort_values("t"))
    if len(ts) > 30:
        recent = ts.tail(100)
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent.r.values, 1)
        cur = recent.r.iloc[-1]
        need = 1000 - cur
        print(f"current rating {cur:.0f}, trend {slope*100:+.1f} Elo per 100 games")
        if slope > 0:
            print(f"at this pace: ~{need/slope:.0f} more games to hit 1000")
        else:
            print("trend is flat/negative — the leak fixes above matter more than volume")


if __name__ == "__main__":
    main()
