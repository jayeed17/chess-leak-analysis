"""Streamlit dashboard over data/moves.parquet and data/brilliancies.parquet.

Usage: streamlit run app.py

Reads data/ (the full local pipeline output) when present, falling back to
public_data/ (a small, pre-exported subset committed to the repo -- see
export_public.py) when it isn't. That's the case on Streamlit Community
Cloud: a fresh git clone has no data/ (gitignored) and no data/raw/ PGN
cache, so the deployed app runs entirely off public_data/. Locally, nothing
changes -- data/ is always checked first.

Note: "blunder" here means wp_loss >= 20 (win-probability loss), not the
parquet's cpl-based `blunder` column. See report.py/model.py's comparison of
the two -- cpl alone overstates severity in already-decided positions
(a 900cp -> 400cp swing is still totally winning).

Every rate shown carries an n and a 95% Wilson CI (dashboard_ext.rate_ci /
rate_table) -- a filtered slice can drop to a handful of blunders, and a bare
percentage on 8 events is exactly how the piece and time-control findings
this project later retracted got made in the first place.
"""
import glob, io, json, os

import chess, chess.pgn
import pandas as pd
import streamlit as st

from dashboard_ext import wilson, rate_table, overlapping, render_brilliancy_tab, ply_for_brilliancy, MIN_N
from theme import (inject_theme, page_header, kpi_row, split_rate, SEVERITY, BAR_BASE,
                    board_svg, board_strip, severity_bar, trend_chart)


def _first_existing(*paths):
    """First path that exists on disk; the last one otherwise, so a missing
    file still fails with a clear FileNotFoundError instead of silently
    picking nothing."""
    return next((p for p in paths if os.path.exists(p)), paths[-1])


DATA_PATH = _first_existing("data/moves.parquet", "public_data/moves.parquet")
BRILLIANCIES_PATH = _first_existing("data/brilliancies.parquet", "public_data/brilliancies.parquet")
PUBLIC_BOARDS_PATH = "public_data/boards.parquet"
RAW_GLOB = "data/raw/*/*.json"
BLUNDER_WP = 20  # wp_loss threshold used throughout this dashboard
LINKEDIN_URL = "https://www.linkedin.com/in/zayeed-bin-kabir/"

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
        kpi_row(st, [{"label": "Mean wp_loss", "value": "-", "severity": "neutral"}])
        return
    current = df.sort_values("end_time").my_rating.dropna()
    thin = len(df) < MIN_N
    blunder_value, blunder_sub = split_rate(int(df.is_blunder.sum()), len(df))
    best_value, best_sub = split_rate(int(df.played_best.sum()), len(df))
    kpi_row(st, [
        {"label": "Mean wp_loss", "value": f"{df.wp_loss.mean():.1f}", "severity": "inaccuracy"},
        {"label": f"Blunder rate (wp_loss≥{BLUNDER_WP})", "value": blunder_value, "sub": blunder_sub,
         "severity": "blunder", "thin": thin},
        {"label": "Best-move rate", "value": best_value, "sub": best_sub,
         "severity": "good", "thin": thin},
        {"label": "Rating (in filter)",
         "value": f"{current.iloc[-1]:.0f}" if len(current) else "-", "severity": "neutral"},
    ])


# Matches the tactical/hangs definition used everywhere else in this project
# (report.py, model.py's motif comparison) -- deliberately excludes allows_mate,
# same as the established 69.2%/26.7% finding. Including allows_mate would push
# the combined share to ~71.6% and make this chart's caption disagree with every
# other place that number gets quoted.
HANGS_MOTIFS = {"hangs_pawn", "hangs_knight", "hangs_bishop", "hangs_rook", "hangs_queen"}
TACTICAL_MOTIFS = HANGS_MOTIFS | {"allows_double_attack", "allows_fork_check", "loses_material"}

TREND_METRICS = {
    "Blunder rate": "is_blunder",
    "Mean wp_loss": "wp_loss",
    "Best-move rate": "played_best",
    "Rating": "my_rating",
}


def trend_section(df):
    st.subheader("Trend")
    choice = st.selectbox("Metric", list(TREND_METRICS.keys()))
    value_col = TREND_METRICS[choice]
    if not len(df):
        st.caption("no moves in this filter")
        return
    # BAR_BASE, not a severity color -- this is a series over time, not a ranking
    trend_chart(st, df, choice, value_col, color=BAR_BASE)
    if choice == "Rating":
        st.caption("~1,024 games over 8 months, and rating runs flat to negative "
                   "across that span -- more volume hasn't moved it on its own.")
    else:
        st.caption("Weekly mean over whatever the sidebar filters currently select. "
                   "The rating trend above these charts is the one line confirmed "
                   "flat to negative; this view is here to eyeball the others, not "
                   "a claim about their shape.")


def render_rate_section(df, group_col, title, order=None):
    """Chart + Wilson-CI table for blunder rate across a bucketed column.

    Every group carries n and a 95% CI (rate_table), with the same overlap/
    thin warnings the brilliancy tab uses -- a chart bar alone can't show
    that, so the table underneath it is not optional decoration.

    severity_bar highlights only the single worst bar in red -- these are
    all blunder-rate charts, so higher_is_worse=True (its default) is right
    for every one of them.
    """
    st.subheader(title)
    if not len(df):
        st.caption("no moves in this filter")
        return
    g = rate_table(df, group_col, "is_blunder")
    if order:
        g = g.reindex([o for o in order if o in g.index])
    severity_bar(st, g["rate_%"], group_col, "Blunder rate (%)", order=order)
    st.dataframe(g, width="stretch")
    if overlapping(g):
        st.caption("⚠️ All intervals overlap — no group separates from any other. "
                   "Do not read a pattern here.")
    elif g.thin.any():
        st.caption(f"⚠️ Some groups have n < {MIN_N}; treat those rates as indicative only.")


def phase_section(df):
    render_rate_section(df, "phase", "Blunder rate by phase",
                         order=["opening", "middlegame", "endgame"])
    st.caption("Middlegame is the worst phase (9.4%) against 5.3% in the opening and "
               "5.8% in the endgame. An earlier cut using raw centipawn loss showed "
               "the endgame as worst instead -- that turned out to be a measurement "
               "artifact, not a real pattern.")


def clock_section(df):
    d = df.dropna(subset=["clock_left"]).copy()
    d["clock_bucket"] = pd.cut(d.clock_left, [0, 10, 30, 60, 120, 300, 1e9],
                                labels=["<10s", "10-30s", "30-60s", "1-2m", "2-5m", "5m+"])
    render_rate_section(d, "clock_bucket", "Blunder rate vs. clock remaining")
    st.caption("Blunder rate does rise as the clock runs down, but blitz and rapid "
               "blunder at nearly the same overall rate (12.7% vs 13.3%) and blunder "
               "rate is actually *higher* on moves I spent longer thinking on -- the "
               "wrong direction for a pure time-pressure story. Clock is a real but "
               "minor factor, not the driver. The red bar here is the highest *rate*, "
               "not the biggest problem -- under 30s on the clock was also the "
               "smallest of five simulated leaks (~3.7 Elo), because so few of my "
               "moves actually happen there.")


def move_no_section(df):
    d = df.copy()
    d["move_bucket"] = (d.move_no // 5 * 5).clip(upper=60)
    render_rate_section(d, "move_bucket", "Blunder rate by move number")
    st.caption("Rate climbs through the opening and stays elevated across the "
               "middlegame stretch, which lines up with middlegame being the "
               "single worst phase above.")


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
    # NOT a rate ranking -- these bars sum to ~100% of blunders, so there's no
    # single "worst" one. Highlighting the max would just flag whichever
    # category the tactical mass happens to be split across (that was the
    # actual bug: "positional" looked worst only because tactical blunders are
    # divided across four separate categories). Colour comes from an explicit
    # tactical/positional classification instead.
    severity_bar(st, g["share_%"], "motif", "Share of blunders (%)", highlight=TACTICAL_MOTIFS)
    st.dataframe(g, width="stretch")
    if n < MIN_N:
        st.caption(f"⚠️ Only {n} blunders (n) in this filter — motif shares are indicative only.")

    # computed from raw counts, not the already-rounded share_% column, to
    # avoid compounding rounding error across several summed categories
    tactical_pct = 100 * g["count"].reindex(TACTICAL_MOTIFS).fillna(0).sum() / n
    hangs_pct = 100 * g["count"].reindex(HANGS_MOTIFS).fillna(0).sum() / n
    st.caption(f"Red = tactical oversights (hangs, forks, double attacks, lost material): "
               f"{tactical_pct:.1f}% of blunders in this filter, {hangs_pct:.1f}% of them "
               f"outright hangs. Not subtle positional drift, mostly a piece left undefended "
               f"on the wrong square.")


@st.cache_data
def game_pgn_index():
    """game_url -> raw PGN text, built once from the cached monthly archives.
    Empty on Streamlit Community Cloud -- data/raw/ isn't in the repo -- which
    is exactly the condition board_before_move() below falls back on."""
    index = {}
    for f in glob.glob(RAW_GLOB):
        for g in json.loads(open(f).read()).get("games", []):
            if "pgn" in g:
                index[g["url"]] = g["pgn"]
    return index


@st.cache_data
def load_boards():
    """Precomputed FEN for the positions the board strips display -- built by
    export_public.py. Only consulted when the PGN cache (data/raw/) isn't
    available, which is the public deployment's normal state by design (it
    has no data/raw/ at all, so game_pgn_index() above is always empty)."""
    try:
        b = pd.read_parquet(PUBLIC_BOARDS_PATH)
    except FileNotFoundError:
        b = pd.DataFrame(columns=["game_url", "ply", "fen", "played_uci", "best_uci", "caption"])
    return b.set_index(["game_url", "ply"])


def board_before_move(game_url, ply):
    """Board with `ply` half-moves already played. PGN cache first (replays
    the actual game -- works for any position, that's the local dev case);
    precomputed FEN second (only covers the ~9 positions the board strips
    show, but needs no PGN at all -- that's the public deployment case).
    None if neither has this position."""
    pgn = game_pgn_index().get(game_url)
    if pgn is not None:
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        for node in game.mainline():
            if board.ply() == ply:
                return board
            board.push(node.move)
        return board if board.ply() == ply else None
    boards = load_boards()
    key = (game_url, ply)
    return chess.Board(boards.loc[key, "fen"]) if key in boards.index else None


def worst_blunder_figure(row, size=250):
    """One board figure for the static worst-blunders strip: played (red) vs.
    engine best (slate), via theme.board_svg -- same colours as the big
    interactive board below, just smaller and non-interactive."""
    board = board_before_move(row.game_url, row.ply)
    if board is None:
        return None
    try:
        played = board.parse_san(row.san)
    except ValueError:
        return None
    best = None
    if row.best_san and row.best_san != row.san:
        try:
            best = board.parse_san(row.best_san)
        except ValueError:
            best = None
    svg = board_svg(board, played=played, best=best, size=size)
    caption = (f"<b>{row.san}</b> instead of {row.best_san} · "
               f"wp_loss {row.wp_loss:.0f} · {row.motif}")
    return {"svg": str(svg), "caption": caption}


def worst_blunders_strip(df):
    st.subheader("Three worst blunders")
    if not len(df):
        st.caption("no moves in this filter")
        return
    worst = df.nlargest(3, "wp_loss")
    figures = [worst_blunder_figure(row) for _, row in worst.iterrows()]
    figures = [f for f in figures if f]
    if figures:
        board_strip(st, figures)


def render_worst_move(row):
    """Blunders tab's big interactive board: your move (red) vs. engine best (slate)."""
    board = board_before_move(row.game_url, row.ply)
    if board is None:
        st.warning("couldn't find/reconstruct this position from the cached PGN")
        return
    try:
        played = board.parse_san(row.san)
    except ValueError:
        st.warning(f"couldn't replay move {row.san!r} on the reconstructed board")
        return
    best = None
    if row.best_san and row.best_san != row.san:
        try:
            best = board.parse_san(row.best_san)
        except ValueError:
            best = None
    svg = board_svg(board, played=played, best=best, size=400)
    st.components.v1.html(str(svg), height=420)
    st.caption(f"red = your move ({row.san}) · slate = engine best ({row.best_san}) · "
               f"cpl {row.cpl:.0f} · wp_loss {row.wp_loss:.1f} · {row.motif}")
    st.markdown(f"[open game on chess.com]({row.game_url})")


def render_brilliancy_move(row):
    """Sacrifices tab's big interactive board: the sacrifice itself, slate --
    there's no separate 'best move' here, the played move IS the good one."""
    board = board_before_move(row.game_url, ply_for_brilliancy(row.move_no, row.color))
    if board is None:
        st.warning("couldn't find/reconstruct this position -- no PGN cache and it "
                    "isn't one of the precomputed board-strip positions")
        return
    try:
        move = board.parse_san(row.san)
    except ValueError:
        st.warning(f"couldn't replay move {row.san!r} on the reconstructed board")
        return
    svg = board_svg(board, best=move, size=400)
    st.components.v1.html(str(svg), height=420)
    st.caption(f"{row.san} · {row.piece} sacrifice · margin {row.margin}cp · {row.label}")
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


def blunders_tab(df):
    st.write(
        "Every rated rapid and blitz move I've played since January, scored by "
        "Stockfish and by how much of my win probability each move actually cost -- "
        "not just raw centipawns, which overstate severity once a position is "
        "already decided."
    )
    render_kpis(df)
    trend_section(df)
    phase_section(df)
    left, right = st.columns(2)
    with left:
        clock_section(df)
    with right:
        move_no_section(df)
    motif_breakdown(df)
    worst_blunders_strip(df)
    worst_moves_table(df)


def sacrifices_tab():
    sel = render_brilliancy_tab(st, path=BRILLIANCIES_PATH, board_lookup=board_before_move)
    if sel is not None:
        render_brilliancy_move(sel)


CADENCE_NOTE = "updated roughly every two months"


def main():
    df_all = load_data()
    df = sidebar_filters(df_all)
    filtered_line = (f"{len(df):,} moves / {df.game_url.nunique():,} games in view "
                      f"(of {len(df_all):,} moves / {df_all.game_url.nunique():,} total)")
    # coverage window's end date is the max end_time actually in the data --
    # not today's date, not a hardcoded string -- so this can't go stale
    # silently. Loading this in six months should show a six-month-old range,
    # not something that quietly still looks current.
    coverage_line = (f"{df_all.game_url.nunique():,} games · "
                      f"{df_all.end_time.min():%Y-%m-%d} to {df_all.end_time.max():%Y-%m-%d} · "
                      f"{CADENCE_NOTE}")
    dateline = f"{filtered_line}<br>{coverage_line}"
    page_header(st, name="Zayeed Bin Kabir", handle="jayeed101",
                linkedin_url=LINKEDIN_URL, dateline=dateline)

    tab_blunders, tab_sac = st.tabs(["Blunders", "Sacrifices"])
    with tab_blunders:
        blunders_tab(df)
    with tab_sac:
        sacrifices_tab()


if __name__ == "__main__":
    main()
