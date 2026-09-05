"""Build a small, self-contained public dataset for the deployed dashboard.

Streamlit Community Cloud gets a fresh git clone: no data/ (gitignored), no
data/raw/ PGN cache. app.py falls back to public_data/ when data/ is absent
(see its module docstring) -- this script builds that fallback:

  public_data/moves.parquet        the columns app.py's Blunders tab reads,
                                    nothing else, so the file stays small.
  public_data/brilliancies.parquet the columns app.py's Sacrifices tab reads.
  public_data/boards.parquet       precomputed FEN + played/best move (UCI) +
                                    caption, for ONLY the positions the static
                                    board strips display (3 worst blunders +
                                    up to 6 brilliancies). This is what lets
                                    those boards render with zero dependency
                                    on data/raw/ in the public deployment --
                                    everything else (the interactive
                                    worst-moves table, the brilliancy
                                    selectbox) still needs data/raw/ and so
                                    only works locally.

Run after build_dataset.py / brilliancy.py, before deploying:
    python export_public.py
"""
import glob, io, json, os

import chess, chess.pgn
import pandas as pd

MOVES_PATH = "data/moves.parquet"
BRILLIANCIES_PATH = "data/brilliancies.parquet"
RAW_GLOB = "data/raw/*/*.json"
OUT_DIR = "public_data"

# Exactly the columns app.py / dashboard_ext.py read off each dataframe.
# Keep this in sync if the dashboard starts reading a new column -- that's
# the whole point of listing it explicitly rather than just dropping a few
# known-large ones.
MOVES_COLUMNS = [
    "game_url", "end_time", "time_class", "color", "phase", "my_rating",
    "move_no", "san", "best_san", "cpl", "wp_loss", "motif", "clock_left",
    "played_best", "ply",
]
BRILLIANCY_COLUMNS = [
    "game_url", "end_time", "time_class", "color", "won", "move_no",
    "phase", "san", "piece", "label", "margin",
]


def game_pgn_index():
    index = {}
    for f in glob.glob(RAW_GLOB):
        for g in json.loads(open(f).read()).get("games", []):
            if "pgn" in g:
                index[g["url"]] = g["pgn"]
    return index


def board_before_ply(pgn_index, game_url, ply):
    pgn = pgn_index.get(game_url)
    if pgn is None:
        return None
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for node in game.mainline():
        if board.ply() == ply:
            return board
        board.push(node.move)
    return board if board.ply() == ply else None


def blunder_board_row(pgn_index, row):
    """One boards.parquet row for a worst-blunder strip figure."""
    board = board_before_ply(pgn_index, row.game_url, row.ply)
    if board is None:
        return None
    try:
        played = board.parse_san(row.san)
    except ValueError:
        return None
    best_uci = ""
    if row.best_san and row.best_san != row.san:
        try:
            best_uci = board.parse_san(row.best_san).uci()
        except ValueError:
            pass
    caption = (f"<b>{row.san}</b> instead of {row.best_san} · "
               f"wp_loss {row.wp_loss:.0f} · {row.motif}")
    return {"game_url": row.game_url, "ply": row.ply, "fen": board.fen(),
            "played_uci": played.uci(), "best_uci": best_uci, "caption": caption}


def brilliancy_board_row(pgn_index, row):
    """One boards.parquet row for a brilliancy strip figure. brilliancies.parquet
    has move_no + color, not ply -- same derivation as dashboard_ext.ply_for_brilliancy."""
    ply = (row.move_no - 1) * 2 + (0 if row.color == "white" else 1)
    board = board_before_ply(pgn_index, row.game_url, ply)
    if board is None:
        return None
    try:
        move = board.parse_san(row.san)
    except ValueError:
        return None
    caption = (f"<b>{row.san}</b> · {row.piece} sacrifice · "
               f"margin {row.margin}cp · {row.label}")
    return {"game_url": row.game_url, "ply": ply, "fen": board.fen(),
            "played_uci": move.uci(), "best_uci": move.uci(), "caption": caption}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pgn_index = game_pgn_index()
    print(f"{len(pgn_index)} cached games available for board export")

    moves = pd.read_parquet(MOVES_PATH)
    moves[MOVES_COLUMNS].to_parquet(f"{OUT_DIR}/moves.parquet", index=False)
    print(f"wrote {OUT_DIR}/moves.parquet: {len(moves):,} rows, "
          f"{len(MOVES_COLUMNS)} columns (was {len(moves.columns)})")

    board_rows = []
    for _, row in moves.nlargest(3, "wp_loss").iterrows():
        r = blunder_board_row(pgn_index, row)
        if r:
            board_rows.append(r)
    print(f"  {len(board_rows)}/3 worst-blunder boards exported "
          f"({3 - len(board_rows)} skipped -- not in the PGN cache)")

    try:
        brilliancies = pd.read_parquet(BRILLIANCIES_PATH)
    except FileNotFoundError:
        brilliancies = None
        print("no data/brilliancies.parquet -- skipping Sacrifices export "
              "(run brilliancy.py first if you want that tab populated)")

    if brilliancies is not None:
        brilliancies[BRILLIANCY_COLUMNS].to_parquet(f"{OUT_DIR}/brilliancies.parquet", index=False)
        print(f"wrote {OUT_DIR}/brilliancies.parquet: {len(brilliancies):,} rows, "
              f"{len(BRILLIANCY_COLUMNS)} columns (was {len(brilliancies.columns)})")

        brilliant = (brilliancies[brilliancies.label == "brilliant"]
                     .sort_values("margin", ascending=False))
        before = len(board_rows)
        for _, row in brilliant.head(6).iterrows():   # strip 1 (top 3) + strip 2 (next 3)
            r = brilliancy_board_row(pgn_index, row)
            if r:
                board_rows.append(r)
        n_wanted = min(6, len(brilliant))
        print(f"  {len(board_rows) - before}/{n_wanted} brilliancy boards exported")

    boards = pd.DataFrame(board_rows)
    boards.to_parquet(f"{OUT_DIR}/boards.parquet", index=False)
    print(f"wrote {OUT_DIR}/boards.parquet: {len(boards)} board positions total")

    total_kb = sum(os.path.getsize(f"{OUT_DIR}/{f}") for f in os.listdir(OUT_DIR)) / 1024
    print(f"\n{OUT_DIR}/ total size: {total_kb:.0f} KB")


if __name__ == "__main__":
    main()
