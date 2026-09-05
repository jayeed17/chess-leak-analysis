"""Shared stats helpers + brilliancy tab for app.py.

Import into app.py:
    from dashboard_ext import wilson, rate_ci, rate_table, render_brilliancy_tab
"""
import math
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


def ply_for_brilliancy(move_no, color):
    """brilliancies.parquet stores move_no + color but not ply -- derive it,
    the same way build_dataset.py's ply column is defined: half-moves already
    played before this one. ply = (move_no-1)*2 + (0 if white else 1)."""
    return (move_no - 1) * 2 + (0 if color == "white" else 1)


def _brilliancy_figure(row, board_lookup, size=250):
    """One board figure for a brilliant row: the sacrifice itself in slate
    (board_svg's `best` colour) -- there's no separate 'best move' here, the
    played move IS the thing being evaluated.

    board_lookup(game_url, ply) -> chess.Board | None is app.py's
    board_before_move -- PGN-replay locally, precomputed FEN when the PGN
    cache is absent (the public deployment). No PGN-handling code lives in
    this file anymore; it all goes through that one shared lookup.
    """
    from theme import board_svg
    board = board_lookup(row.game_url, ply_for_brilliancy(row.move_no, row.color))
    if board is None:
        return None
    try:
        move = board.parse_san(row.san)
    except ValueError:
        return None
    svg = board_svg(board, best=move, size=size)
    caption = (f"<b>{row.san}</b> · {row.piece} sacrifice · "
               f"margin {row.margin}cp · {row.label}")
    return {"svg": str(svg), "caption": caption}


def render_brilliancy_tab(st, path="data/brilliancies.parquet", board_lookup=None):
    """Streamlit tab for sacrifice analysis. Pass the streamlit module as `st`.

    board_lookup(game_url, ply) -> chess.Board | None: app.py's
    board_before_move, reused here rather than a second PGN-handling
    implementation, purely to render the board strips.
    """
    from theme import kpi_row, split_rate, board_strip

    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        st.info("No data/brilliancies.parquet yet — run `python brilliancy.py jayeed101`.")
        return

    df["sound"] = df.label.isin(SOUND)
    eligible = df[df.label != "sac_while_winning"]      # see note below
    brilliant = df[df.label == "brilliant"].sort_values("margin", ascending=False)

    st.subheader("Sacrifices")
    st.write(
        "Every move where I put material at risk, checked at depth 18 against the "
        "runner-up move: 739 attempts, 3.1% sound overall, 4.6% once positions I was "
        "already winning easily are excluded from the denominator (see below), and "
        "9 that met the bar for brilliant — sound *and* essentially the only move "
        "that worked."
    )

    if board_lookup is not None and len(brilliant):
        lead = [_brilliancy_figure(r, board_lookup) for _, r in brilliant.head(3).iterrows()]
        lead = [f for f in lead if f]
        if lead:
            board_strip(st, lead)

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
    sound_value, sound_sub = split_rate(int(eligible.sound.sum()), len(eligible))
    kpi_row(st, [
        {"label": "Attempts", "value": f"{len(df):,}", "severity": "neutral"},
        {"label": "Sound", "value": sound_value, "sub": sound_sub, "severity": "good",
         "thin": len(eligible) < MIN_N},
        {"label": "Brilliant", "value": f"{int((df.label == 'brilliant').sum())}", "severity": "good"},
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
    b = brilliant
    if not len(b):
        st.info("No moves met the brilliant bar. At this rating that's an ordinary result.")
        return

    if board_lookup is not None and len(b) > 3:
        more = [_brilliancy_figure(r, board_lookup) for _, r in b.iloc[3:6].iterrows()]
        more = [f for f in more if f]
        if more:
            board_strip(st, more)

    st.dataframe(
        b[["end_time", "time_class", "move_no", "san", "piece", "margin", "won", "game_url"]],
        hide_index=True, use_container_width=True)

    pick = st.selectbox("Show position", b.index,
                        format_func=lambda i: f"move {b.loc[i,'move_no']} — {b.loc[i,'san']} "
                                              f"({b.loc[i,'piece']}, margin {b.loc[i,'margin']}cp)")
    st.caption("Board rendering reuses app.py's board_before_move() -- PGN cache "
               "locally, precomputed FEN in the public deployment.")
    return b.loc[pick]
