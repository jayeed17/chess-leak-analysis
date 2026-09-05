"""Streamlit dashboard over data/moves.parquet.

Usage: streamlit run app.py

Note: "blunder" here means wp_loss >= 20 (win-probability loss), not the
parquet's cpl-based `blunder` column. See report.py/model.py's comparison of
the two -- cpl alone overstates severity in already-decided positions
(a 900cp -> 400cp swing is still totally winning). cpl/ACPL is still shown
for reference.
"""
import glob, io, json

import chess, chess.pgn, chess.svg
import pandas as pd
import streamlit as st

DATA_PATH = "data/moves.parquet"
RAW_GLOB = "data/raw/*/*.json"
BLUNDER_WP = 20  # wp_loss threshold used throughout this dashboard

st.set_page_config(page_title="chess leak analysis", layout="wide")


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


def kpi_row(df):
    current = df.sort_values("end_time").my_rating.dropna()
    cols = st.columns(4)
    cols[0].metric("ACPL", f"{df.cpl.mean():.0f}" if len(df) else "-")
    cols[1].metric(f"Blunder rate (wp_loss≥{BLUNDER_WP})",
                    f"{df.is_blunder.mean()*100:.1f}%" if len(df) else "-")
    cols[2].metric("Best-move rate", f"{df.played_best.mean()*100:.1f}%" if len(df) else "-")
    cols[3].metric("Rating (in filter)", f"{current.iloc[-1]:.0f}" if len(current) else "-")


def blunder_vs_clock_chart(df):
    st.subheader("Blunder rate vs. clock remaining")
    d = df.dropna(subset=["clock_left"]).copy()
    d["clock_bucket"] = pd.cut(d.clock_left, [0, 10, 30, 60, 120, 300, 1e9],
                                labels=["<10s", "10-30s", "30-60s", "1-2m", "2-5m", "5m+"])
    g = d.groupby("clock_bucket", observed=True).is_blunder.mean().mul(100)
    st.bar_chart(g)


def blunder_by_move_chart(df):
    st.subheader("Blunder rate by move number")
    d = df.copy()
    d["move_bucket"] = (d.move_no // 5 * 5).clip(upper=60)
    g = d.groupby("move_bucket").is_blunder.mean().mul(100)
    st.bar_chart(g)


def motif_breakdown(df):
    st.subheader("What you actually hang (blunder motifs)")
    m = df[df.is_blunder].motif.value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(m)


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


def render_move(row):
    board = board_before_move(row.game_url, row.ply)
    if board is None:
        st.warning("couldn't find/reconstruct this position from the cached PGN")
        return
    try:
        played = board.parse_san(row.san)
    except ValueError:
        st.warning(f"couldn't replay move {row.san!r} on the reconstructed board")
        return
    arrows = [chess.svg.Arrow(played.from_square, played.to_square, color="#cc0000")]
    if row.best_san and row.best_san != row.san:
        best = board.parse_san(row.best_san)
        arrows.append(chess.svg.Arrow(best.from_square, best.to_square, color="#00aa00"))
    svg = chess.svg.board(board, arrows=arrows, size=400, flipped=(row.color == "black"))
    st.components.v1.html(str(svg), height=420)
    st.caption(f"red = your move ({row.san}) · green = engine best ({row.best_san}) · "
               f"cpl {row.cpl:.0f} · wp_loss {row.wp_loss:.1f} · {row.motif}")
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
        render_move(worst.iloc[event.selection.rows[0]])


def main():
    st.title("Chess leak analysis — jayeed101")
    df_all = load_data()
    df = sidebar_filters(df_all)
    st.caption(f"{len(df):,} moves / {df.game_url.nunique():,} games in view "
               f"(of {len(df_all):,} moves / {df_all.game_url.nunique():,} total)")

    kpi_row(df)
    left, right = st.columns(2)
    with left:
        blunder_vs_clock_chart(df)
    with right:
        blunder_by_move_chart(df)
    motif_breakdown(df)
    worst_moves_table(df)


if __name__ == "__main__":
    main()
