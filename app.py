"""Streamlit dashboard over data/moves.parquet and data/brilliancies.parquet.

Usage: streamlit run app.py

Note: "blunder" here means wp_loss >= 20 (win-probability loss), not the
parquet's cpl-based `blunder` column. See report.py/model.py's comparison of
the two -- cpl alone overstates severity in already-decided positions
(a 900cp -> 400cp swing is still totally winning).

Every rate shown carries an n and a 95% Wilson CI (dashboard_ext.rate_ci /
rate_table) -- a filtered slice can drop to a handful of blunders, and a bare
percentage on 8 events is exactly how the piece and time-control findings
this project later retracted got made in the first place.
"""
import glob, io, json

import chess, chess.pgn, chess.svg
import pandas as pd
import streamlit as st

from dashboard_ext import wilson, rate_table, overlapping, render_brilliancy_tab, board_at, MIN_N
from theme import inject_theme, kpi_row, split_rate, QUALITY

DATA_PATH = "data/moves.parquet"
RAW_GLOB = "data/raw/*/*.json"
BLUNDER_WP = 20  # wp_loss threshold used throughout this dashboard
BAR_COLOR = "#769656"  # theme.DARK -- board dark-square color, all st.bar_chart bars

st.set_page_config(page_title="chess leak analysis", layout="wide")
inject_theme(st)


@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_parquet(path)
    df["is_blunder"] = df.wp_loss >= BLUNDER_WP
    df["date"] = pd.to_datetime(df.end_time).dt.date
    return df


def sidebar_filters(df):
    st.sidebar.header("Filters")
    tc = st.sidebar.multiselect("Time control", sorted(df.time_class.dropna().unique()),
                                 default=sorted(df.time_class.dropna().unique()))
    colors = st.sidebar.multiselect("Color", ["white", "black"], default=["white", "black"])
    phases = st.sidebar.multiselect("Phase", ["opening", "middlegame", "endgame"],
                                     default=["opening", "middlegame", "endgame"])
    date_range = st.sidebar.date_input("Date range", (df.date.min(), df.date.max()),
                                        min_value=df.date.min(), max_value=df.date.max())

    mask = df.time_class.isin(tc) & df.color.isin(colors) & df.phase.isin(phases)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask &= df.date.between(*date_range)
    return df[mask]


def render_kpis(df):
    """Top KPI row. Uses theme.kpi_row/split_rate rather than st.metric --
    rate_ci()'s one-line string ('6.9% (CI 5.8-8.2, n=2,239)') rendered at
    st.metric's display size wraps and clips; kpi_row splits value from
    interval so only the headline number is large."""
    if not len(df):
        kpi_row(st, [{"label": "Mean wp_loss", "value": "-", "quality": "neutral"}])
        return
    current = df.sort_values("end_time").my_rating.dropna()
    thin = len(df) < MIN_N
    blunder_value, blunder_sub = split_rate(int(df.is_blunder.sum()), len(df))
    best_value, best_sub = split_rate(int(df.played_best.sum()), len(df))
    kpi_row(st, [
        {"label": "Mean wp_loss", "value": f"{df.wp_loss.mean():.1f}", "quality": "inaccuracy"},
        {"label": f"Blunder rate (wp_loss≥{BLUNDER_WP})", "value": blunder_value, "sub": blunder_sub,
         "quality": "mistake", "thin": thin},
        {"label": "Best-move rate", "value": best_value, "sub": best_sub,
         "quality": "good", "thin": thin},
        {"label": "Rating (in filter)",
         "value": f"{current.iloc[-1]:.0f}" if len(current) else "-", "quality": "brilliant"},
    ])


def render_rate_section(df, group_col, title, order=None):
    """Bar chart + Wilson-CI table for blunder rate across a bucketed column.

    Every group carries n and a 95% CI (rate_table), with the same overlap/
    thin warnings the brilliancy tab uses -- a chart bar alone can't show
    that, so the table underneath it is not optional decoration.
    """
    st.subheader(title)
    if not len(df):
        st.caption("no moves in this filter")
        return
    g = rate_table(df, group_col, "is_blunder")
    if order:
        order = [o for o in order if o in g.index]
        g = g.reindex(order)
        # st.bar_chart re-sorts a plain string index alphabetically regardless of
        # row order -- an ordered CategoricalIndex is what actually gets respected
        g.index = pd.CategoricalIndex(g.index, categories=order, ordered=True)
    st.bar_chart(g["rate_%"], color=BAR_COLOR)
    st.dataframe(g, width="stretch")
    if overlapping(g):
        st.caption("⚠️ All intervals overlap — no group separates from any other. "
                   "Do not read a pattern here.")
    elif g.thin.any():
        st.caption(f"⚠️ Some groups have n < {MIN_N}; treat those rates as indicative only.")


def phase_section(df):
    render_rate_section(df, "phase", "Blunder rate by phase",
                         order=["opening", "middlegame", "endgame"])


def clock_section(df):
    d = df.dropna(subset=["clock_left"]).copy()
    d["clock_bucket"] = pd.cut(d.clock_left, [0, 10, 30, 60, 120, 300, 1e9],
                                labels=["<10s", "10-30s", "30-60s", "1-2m", "2-5m", "5m+"])
    render_rate_section(d, "clock_bucket", "Blunder rate vs. clock remaining")


def move_no_section(df):
    d = df.copy()
    d["move_bucket"] = (d.move_no // 5 * 5).clip(upper=60)
    render_rate_section(d, "move_bucket", "Blunder rate by move number")


def motif_breakdown(df):
    st.subheader("What you actually hang (blunder motifs)")
    blunders = df[df.is_blunder]
    n = len(blunders)
    if n == 0:
        st.caption("no blunders in this filter")
        return
    counts = blunders.motif.value_counts()
    g = pd.DataFrame({"count": counts, "n": n})
    g["share_%"] = (100 * g["count"] / n).round(1)
    bounds = [wilson(c, n) for c in g["count"]]
    g["ci_low"] = [round(b[0], 1) for b in bounds]
    g["ci_high"] = [round(b[1], 1) for b in bounds]
    st.bar_chart(g["share_%"], color=BAR_COLOR)
    st.dataframe(g, width="stretch")
    if n < MIN_N:
        st.caption(f"⚠️ Only {n} blunders (n) in this filter — motif shares are indicative only.")


@st.cache_data
def game_pgn_index():
    """game_url -> raw PGN text, built once from the cached monthly archives."""
    index = {}
    for f in glob.glob(RAW_GLOB):
        for g in json.loads(open(f).read()).get("games", []):
            if "pgn" in g:
                index[g["url"]] = g["pgn"]
    return index


def board_before_move(game_url, ply):
    """Replay the cached PGN to the position with `ply` half-moves already played."""
    pgn = game_pgn_index().get(game_url)
    if pgn is None:
        return None
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for node in game.mainline():
        if board.ply() == ply:
            return board
        board.push(node.move)
    return board if board.ply() == ply else None


def render_board_svg(board, arrows, flipped, caption):
    svg = chess.svg.board(board, arrows=arrows, size=400, flipped=flipped)
    st.components.v1.html(str(svg), height=420)
    st.caption(caption)


def render_worst_move(row):
    """Blunders tab board: your move (red) vs. engine best (green)."""
    board = board_before_move(row.game_url, row.ply)
    if board is None:
        st.warning("couldn't find/reconstruct this position from the cached PGN")
        return
    try:
        played = board.parse_san(row.san)
    except ValueError:
        st.warning(f"couldn't replay move {row.san!r} on the reconstructed board")
        return
    arrows = [chess.svg.Arrow(played.from_square, played.to_square, color=QUALITY["mistake"])]
    if row.best_san and row.best_san != row.san:
        best = board.parse_san(row.best_san)
        arrows.append(chess.svg.Arrow(best.from_square, best.to_square, color=QUALITY["good"]))
    render_board_svg(board, arrows, flipped=(row.color == "black"),
                      caption=f"red = your move ({row.san}) · green = engine best ({row.best_san}) · "
                              f"cpl {row.cpl:.0f} · wp_loss {row.wp_loss:.1f} · {row.motif}")
    st.markdown(f"[open game on chess.com]({row.game_url})")


def render_brilliancy_move(row):
    """Sacrifices tab board: the sacrifice itself (no separate 'best move' here --
    the played move IS the thing being evaluated). Reuses game_pgn_index() for the
    PGN text and dashboard_ext.board_at() for the ply-from-move_no+color derivation,
    rather than a second copy of either."""
    pgn = game_pgn_index().get(row.game_url)
    if pgn is None:
        st.warning("couldn't find the cached PGN for this game")
        return
    board = board_at(pgn, row.move_no, row.color)
    try:
        move = board.parse_san(row.san)
    except ValueError:
        st.warning(f"couldn't replay move {row.san!r} on the reconstructed board")
        return
    arrows = [chess.svg.Arrow(move.from_square, move.to_square, color=QUALITY["brilliant"])]
    render_board_svg(board, arrows, flipped=(row.color == "black"),
                      caption=f"{row.san} · {row.piece} sacrifice · margin {row.margin}cp · {row.label}")
    st.markdown(f"[open game on chess.com]({row.game_url})")


def worst_moves_table(df):
    st.subheader("Worst moves — click a row to see the board")
    sort_col = st.radio("Rank by", ["wp_loss", "cpl"], horizontal=True,
                         help="wp_loss is the honest one -- cpl alone overstates "
                              "severity in already-decided positions")
    worst = (df.nlargest(20, sort_col)
               [["game_url", "move_no", "color", "san", "best_san", "cpl", "wp_loss",
                 "motif", "clock_left", "ply"]]
               .reset_index(drop=True))
    event = st.dataframe(worst, on_select="rerun", selection_mode="single-row", width="stretch")
    if event.selection.rows:
        render_worst_move(worst.iloc[event.selection.rows[0]])


def blunders_tab():
    df_all = load_data()
    df = sidebar_filters(df_all)
    st.caption(f"{len(df):,} moves / {df.game_url.nunique():,} games in view "
               f"(of {len(df_all):,} moves / {df_all.game_url.nunique():,} total)")

    render_kpis(df)
    phase_section(df)
    left, right = st.columns(2)
    with left:
        clock_section(df)
    with right:
        move_no_section(df)
    motif_breakdown(df)
    worst_moves_table(df)


def sacrifices_tab():
    sel = render_brilliancy_tab(st)
    if sel is not None:
        render_brilliancy_move(sel)


def main():
    st.title("Chess leak analysis — jayeed101")
    tab_blunders, tab_sac = st.tabs(["Blunders", "Sacrifices"])
    with tab_blunders:
        blunders_tab()
    with tab_sac:
        sacrifices_tab()


if __name__ == "__main__":
    main()
