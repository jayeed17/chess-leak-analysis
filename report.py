"""Where and when you blunder. Usage: python report.py [--min-games 5]"""
import argparse
import pandas as pd

pd.set_option("display.width", 140)


def rate(df, by, min_n=30):
    g = df.groupby(by).agg(
        moves=("cpl", "size"),
        acpl=("cpl", "mean"),
        blunder_pct=("blunder", "mean"),
        mistake_pct=("mistake", "mean"),
        best_pct=("played_best", "mean"),
    )
    g = g[g.moves >= min_n].sort_values("blunder_pct", ascending=False)
    g["acpl"] = g.acpl.round(1)
    for c in ("blunder_pct", "mistake_pct", "best_pct"):
        g[c] = (g[c] * 100).round(1)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/moves.parquet")
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()
    df = pd.read_parquet(a.path)

    print(f"\n{len(df):,} of your moves across {df.game_url.nunique():,} games")
    print(f"overall ACPL {df.cpl.mean():.1f} | blunder rate {df.blunder.mean()*100:.1f}% | "
          f"best-move rate {df.played_best.mean()*100:.1f}%\n")

    # --- WHERE ---
    print("=" * 60, "\nBY PHASE\n", rate(df, "phase", a.min_n), sep="")
    print("\nBY MOVE NUMBER (5-move buckets)")
    df["move_bucket"] = (df.move_no // 5 * 5).clip(upper=60)
    print(rate(df, "move_bucket", a.min_n).sort_index())

    print("\nBY COLOR\n", rate(df, "color", a.min_n), sep="")
    print("\nWORST OPENINGS (ECO)\n", rate(df, "eco", a.min_n).head(12), sep="")

    print("\nBLUNDER MOTIFS — what you actually hang")
    m = df[df.blunder].motif.value_counts(normalize=True).mul(100).round(1)
    print(m.head(10).to_string())

    print("\nSITUATIONAL TRIGGERS")
    trig = pd.DataFrame({
        "after opponent captured": df.groupby("opp_last_was_capture").blunder.mean(),
        "when already in check": df.groupby("in_check_before").blunder.mean(),
        "when I had a hanging piece": df.groupby(df.my_hanging_before >= 3).blunder.mean(),
        "when down material (<-2)": df.groupby(df.material_diff < -2).blunder.mean(),
    }).T.mul(100).round(1)
    trig.columns = ["no", "yes"]
    print(trig)

    # --- WHEN ---
    print("\n" + "=" * 60)
    print("BY TIME CONTROL\n", rate(df, "time_class", a.min_n), sep="")

    print("\nBY CLOCK REMAINING")
    df["clock_bucket"] = pd.cut(df.clock_left, [0, 10, 30, 60, 120, 300, 1e9],
                                labels=["<10s", "10-30s", "30-60s", "1-2m", "2-5m", "5m+"])
    print(rate(df.dropna(subset=["clock_left"]), "clock_bucket", a.min_n).sort_index())

    print("\nBY TIME SPENT ON THE MOVE")
    df["think_bucket"] = pd.cut(df.time_spent, [-1, 1, 3, 10, 30, 1e9],
                                labels=["<1s", "1-3s", "3-10s", "10-30s", "30s+"])
    print(rate(df.dropna(subset=["time_spent"]), "think_bucket", a.min_n).sort_index())

    print("\nBY HOUR OF DAY\n", rate(df, "hour", a.min_n).sort_index(), sep="")
    print("\nBY WEEKDAY\n", rate(df, "weekday", a.min_n), sep="")

    print("\nBY OPPONENT STRENGTH")
    df["opp_bucket"] = pd.cut(df.rating_diff, [-1e9, -150, -50, 50, 150, 1e9],
                              labels=["much stronger", "stronger", "even", "weaker", "much weaker"])
    print(rate(df, "opp_bucket", a.min_n).sort_index())

    # --- WORST GAMES / POSITIONS ---
    print("\n" + "=" * 60)
    print("YOUR 15 WORST SINGLE MOVES (drill these)")
    worst = df.nlargest(15, "cpl")[
        ["game_url", "move_no", "color", "san", "best_san", "cpl", "motif", "clock_left"]]
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
