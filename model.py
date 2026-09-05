"""ML layer.

1. blunder-risk model  -> P(blunder) for any position-state, + which factors drive it
2. rating-gain simulator -> how much Elo each fixable leak is worth
3. per-game rating delta model -> what actually correlates with winning

Usage: python model.py
"""
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


def prep(df):
    X = df[FEATURES].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X


def main():
    df = pd.read_parquet("data/moves.parquet").dropna(subset=["clock_left"])
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

    base = per_game.won.mean()
    for leak, mask in {
        "blunders under 30s on the clock": (df.clock_left < 30) & df.blunder,
        "blunders in the opening (<=move 12)": (df.phase == "opening") & df.blunder,
        "hanging a piece outright": df.motif.fillna("").str.startswith("hangs_") & df.blunder,
        "blunders after opponent captured": df.opp_last_was_capture & df.blunder,
    }.items():
        removed = df[mask].groupby("game_url").size()
        adj = (per_game.blunders - removed.reindex(per_game.index).fillna(0)).clip(lower=0)
        proj = wp.reindex(adj.clip(upper=5)).mean()
        d_wp = proj - base
        elo = 400 * np.log10(max(base + d_wp, .01) / max(1 - base - d_wp, .01)) - \
              400 * np.log10(max(base, .01) / max(1 - base, .01))
        print(f"  fix '{leak}': +{d_wp*100:.1f} win% ≈ +{elo:.0f} Elo  "
              f"({int(mask.sum())} moves)")

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
