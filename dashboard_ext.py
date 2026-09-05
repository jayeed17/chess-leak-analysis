"""Shared stats helpers + brilliancy tab for app.py.

Import into app.py:
    from dashboard_ext import wilson, rate_ci, rate_table, render_brilliancy_tab
"""
import math
import chess
import pandas as pd

MIN_N = 100          # below this, a rate is too thin to show without a warning
SOUND = ("brilliant", "sound_sac")


def wilson(k, n, z=1.96):
    """95% Wilson interval as (lo, hi) percentages. Safe at n=0."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def rate_ci(k, n, decimals=1):
    """'6.9% (CI 5.8-8.2, n=2239)' — never a bare percentage."""
    if n == 0:
        return "no data"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.{decimals}f}% (CI {lo:.1f}-{hi:.1f}, n={n:,})"


def rate_table(df, by, flag_col):
    """Group -> rate with Wilson bounds. Returns a frame ready to display."""
    g = df.groupby(by, observed=True)[flag_col].agg(["sum", "size"])
    g.columns = ["hits", "n"]
    g["rate_%"] = (100 * g.hits / g.n).round(1)
    bounds = g.apply(lambda r: wilson(r.hits, r.n), axis=1)
    g["ci_low"] = [round(b[0], 1) for b in bounds]
    g["ci_high"] = [round(b[1], 1) for b in bounds]
    g["thin"] = g.n < MIN_N
    return g


def overlapping(g):
    """True if every pair of intervals overlaps — i.e. nothing separates."""
    rows = list(g[["ci_low", "ci_high"]].itertuples(index=False))
    return all(not (a.ci_high < b.ci_low or b.ci_high < a.ci_low)
               for i, a in enumerate(rows) for b in rows[i + 1:])


def board_at(pgn_text, move_no, color):
    """Rebuild the position before a brilliancy. brilliancies.parquet stores
    move_no + color but not ply, so derive it: ply = (move_no-1)*2 + side."""
    import io, chess.pgn
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    target = (move_no - 1) * 2 + (0 if color == "white" else 1)
    board = game.board()
    for node in game.mainline():
        if board.ply() == target:
            return board
        board.push(node.move)
    return board


def render_brilliancy_tab(st, path="data/brilliancies.parquet"):
    """Streamlit tab for sacrifice analysis. Pass the streamlit module as `st`."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        st.info("No data/brilliancies.parquet yet — run `python brilliancy.py jayeed101`.")
        return

    df["sound"] = df.label.isin(SOUND)
    eligible = df[df.label != "sac_while_winning"]      # see note below

    st.subheader("Sacrifices")
    st.caption(
        "A sacrifice is scored against the **eligible** denominator: sound + unsound. "
        "`sac_while_winning` (engine eval already >500cp) is excluded because those "
        "positions can never be labelled sound by construction — including them "
        "silently penalises phases where evals run extreme, which is what made the "
        "first version of this analysis wrong."
    )

    # rate_ci() is one long string ('6.9% (CI 5.8-8.2, n=2,239)') -- st.metric
    # renders it at display size, which wraps and clips. kpi_row/split_rate
    # split the headline number from its interval instead.
    from theme import kpi_row, split_rate
    sound_value, sound_sub = split_rate(int(eligible.sound.sum()), len(eligible))
    kpi_row(st, [
        {"label": "Attempts", "value": f"{len(df):,}", "quality": "neutral"},
        {"label": "Sound", "value": sound_value, "sub": sound_sub, "quality": "good",
         "thin": len(eligible) < MIN_N},
        {"label": "Brilliant", "value": f"{int((df.label == 'brilliant').sum())}", "quality": "brilliant"},
    ])

    st.write("**Label breakdown**")
    counts = df.label.value_counts().rename_axis("label").reset_index(name="count")
    counts["share_%"] = (100 * counts["count"] / len(df)).round(1)
    st.dataframe(counts, hide_index=True, use_container_width=True)

    for label, col in [("By phase", "phase"), ("By piece", "piece"),
                       ("By time control", "time_class")]:
        st.write(f"**{label}** — eligible denominator, 95% Wilson CI")
        g = rate_table(eligible, col, "sound")
        st.dataframe(g, use_container_width=True)
        if overlapping(g):
            st.caption("⚠️ All intervals overlap — no group separates from any other. "
                       "Do not read a pattern here.")
        elif g.thin.any():
            st.caption("⚠️ Some groups have n < 100; treat those rates as indicative only.")

    st.write("**Conversion** — win rate in games containing a sound sacrifice")
    st.caption(
        "Selection runs *against* the sacrifice here: `sac_while_winning` filtering "
        "removes the easy wins, so this group is drawn from tighter, more contested "
        "games than the overall baseline. A lower win rate does not mean the sacrifice "
        "hurt."
    )
    good = df[df.sound]
    if len(good):
        conv = rate_table(good, "label", "won")
        st.dataframe(conv, use_container_width=True)

    st.write("**Your brilliancies**")
    b = df[df.label == "brilliant"].sort_values("margin", ascending=False)
    if not len(b):
        st.info("No moves met the brilliant bar. At this rating that's an ordinary result.")
        return
    st.dataframe(
        b[["end_time", "time_class", "move_no", "san", "piece", "margin", "won", "game_url"]],
        hide_index=True, use_container_width=True)

    pick = st.selectbox("Show position", b.index,
                        format_func=lambda i: f"move {b.loc[i,'move_no']} — {b.loc[i,'san']} "
                                              f"({b.loc[i,'piece']}, margin {b.loc[i,'margin']}cp)")
    st.caption("Board rendering reuses the PGN-replay path already in app.py — "
               "pass the cached PGN for this game_url into board_at().")
    return b.loc[pick]
