"""Compare yourself against friends. API data only — no Stockfish, runs in seconds.

Usage: python compare.py jayeed101 friend1 friend2 --months 6
"""
import argparse, collections, datetime as dt, time
import pandas as pd
import chesscom

pd.set_option("display.width", 160)


def rating_table(users):
    rows = []
    for u in users:
        s = chesscom.stats(u) or {}
        p = chesscom.profile(u) or {}
        r = {"user": u, "joined": dt.datetime.fromtimestamp(p["joined"]).year if p.get("joined") else None}
        for tc in ("chess_bullet", "chess_blitz", "chess_rapid", "chess_daily"):
            d = s.get(tc)
            if not d:
                continue
            label = tc.split("_")[1]
            r[label] = d["last"]["rating"]
            r[f"{label}_peak"] = d.get("best", {}).get("rating")
            rec = d.get("record", {})
            n = sum(rec.get(k, 0) for k in ("win", "loss", "draw"))
            r[f"{label}_n"] = n
            r[f"{label}_win%"] = round(100 * rec.get("win", 0) / n, 1) if n else None
        if s.get("tactics"):
            r["tactics_peak"] = s["tactics"].get("highest", {}).get("rating")
        if s.get("puzzle_rush", {}).get("best"):
            r["puzzle_rush"] = s["puzzle_rush"]["best"]["score"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("user")


def game_frame(user, months):
    """Recent games as a flat frame, from the player's POV."""
    cutoff = (dt.date.today().replace(day=1) - dt.timedelta(days=31 * months)).strftime("%Y/%m")
    rows = []
    for g in chesscom.all_games(user, since=cutoff):
        if g.get("rules") != "chess":
            continue
        me = "white" if g["white"]["username"].lower() == user.lower() else "black"
        opp = "black" if me == "white" else "white"
        acc = (g.get("accuracies") or {}).get(me)
        end = dt.datetime.fromtimestamp(g["end_time"])
        rows.append({
            "user": user, "color": me,
            "result": g[me]["result"], "won": g[me]["result"] == "win",
            "rating": g[me]["rating"], "opp_rating": g[opp]["rating"],
            "opp": g[opp]["username"].lower(),
            "time_class": g.get("time_class"),
            "accuracy": acc,
            "eco": (g.get("eco") or "").rsplit("/", 1)[-1],
            "end": end, "hour": end.hour,
            "moves": g["pgn"].count(". ") if "pgn" in g else None,
            "url": g["url"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("users", nargs="+", help="you first, then friends")
    ap.add_argument("--months", type=int, default=6)
    a = ap.parse_args()
    users = [u.lower() for u in a.users]
    me = users[0]

    print("\n" + "=" * 70)
    print("RATINGS\n")
    print(rating_table(users).T.to_string())

    frames = []
    for u in users:
        print(f"\nfetching {u}'s last {a.months} months...")
        frames.append(game_frame(u, a.months))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)

    print("\n" + "=" * 70)
    print(f"LAST {a.months} MONTHS\n")
    summ = df.groupby("user").agg(
        games=("won", "size"),
        win_pct=("won", lambda s: round(100 * s.mean(), 1)),
        avg_opp=("opp_rating", "mean"),
        accuracy=("accuracy", "mean"),
        avg_moves=("moves", "mean"),
        rating_start=("rating", "first"),
        rating_end=("rating", "last"),
    ).round(1)
    summ["gain"] = summ.rating_end - summ.rating_start
    print(summ.to_string())

    print("\nWIN% BY COLOR")
    print((df.pivot_table(index="user", columns="color", values="won", aggfunc="mean") * 100)
          .round(1).to_string())

    print("\nWIN% BY TIME CLASS")
    print((df.pivot_table(index="user", columns="time_class", values="won", aggfunc="mean") * 100)
          .round(1).to_string())

    print("\nHOW GAMES END (your losses vs theirs)")
    ends = df[~df.won].pivot_table(index="result", columns="user", values="won",
                                   aggfunc="size", fill_value=0)
    print((ends / ends.sum() * 100).round(1).to_string())

    print("\nTOP OPENINGS — win% (min 5 games)")
    for u in users:
        d = df[df.user == u]
        g = d.groupby("eco").agg(n=("won", "size"), win=("won", "mean"))
        g = g[(g.n >= 5) & (g.index != "")].sort_values("win")
        if len(g):
            print(f"\n  {u} — worst:")
            print((g.assign(win=(g.win * 100).round(1)).head(5)).to_string())

    # head to head
    print("\n" + "=" * 70)
    print("HEAD TO HEAD\n")
    mine = df[df.user == me]
    for f in users[1:]:
        h2h = mine[mine.opp == f]
        if len(h2h):
            print(f"  {me} vs {f}: {len(h2h)} games, "
                  f"{h2h.won.sum()}W-{(~h2h.won).sum()}L "
                  f"({100*h2h.won.mean():.0f}%)")
        else:
            print(f"  {me} vs {f}: never played")

    # who is closing on 1000
    print("\nPACE TO 1000 (blitz/rapid, last 50 games)")
    for u in users:
        d = df[(df.user == u) & df.time_class.isin(["blitz", "rapid"])].tail(50)
        if len(d) < 10:
            continue
        slope = (d.rating.iloc[-1] - d.rating.iloc[0]) / len(d)
        cur = d.rating.iloc[-1]
        eta = f"{(1000-cur)/slope:.0f} games" if slope > 0 and cur < 1000 else \
              ("already there" if cur >= 1000 else "flat/declining")
        print(f"  {u:20} {cur:>5.0f}  {slope*50:+.0f}/50 games  -> {eta}")

    df.to_parquet("data/compare.parquet", index=False)
    print("\nwrote data/compare.parquet")


if __name__ == "__main__":
    main()
