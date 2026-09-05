"""Export your worst positions as a multi-game PGN, importable into a Lichess Study.

Each "game" in the file is one puzzle: the position right before your blunder,
with the engine's best move as the main line (the solution) and your actual
move attached as a side variation for comparison.

Selection is weighted toward middlegame and hangs_* blunders -- the two
findings that survived every correction in this project (worst phase, most
common concrete mistake). It's not a pure top-100-by-wp_loss list because a
pure list would mostly reflect what got tagged mate/allows_mate in already-
decided games, which isn't what's worth drilling.

Usage: python drill.py [--n 100] [--path data/moves.parquet] [--out data/drills/worst_100.pgn]
"""
import argparse, glob, io, json, os

import chess, chess.pgn
import pandas as pd

RAW_GLOB = "data/raw/*/*.json"


def game_pgn_index():
    """game_url -> raw PGN text, built once from the cached monthly archives."""
    index = {}
    for f in glob.glob(RAW_GLOB):
        for g in json.loads(open(f).read()).get("games", []):
            if "pgn" in g:
                index[g["url"]] = g["pgn"]
    return index


def board_before_move(game_url, ply, pgn_index):
    """Replay the cached PGN to the position with `ply` half-moves already played."""
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


def pick_worst(df, n, middlegame_frac=0.4, hangs_frac=0.3):
    """Top wp_loss moves, with dedicated quotas for middlegame and hangs_* blunders.

    A single "middlegame OR hangs_*" pool lets whichever is easier to fill
    (severe hangs_* turn out to cluster in the endgame, where a hung queen
    is starker) crowd out the other. Reserving separate quotas guarantees
    both findings actually show up, while the remaining slots still come
    from the overall pool so genuinely severe blunders elsewhere aren't lost.
    """
    remaining = df
    mg = remaining[remaining.phase == "middlegame"].nlargest(int(n * middlegame_frac), "wp_loss")
    remaining = remaining.drop(mg.index)
    hangs = remaining[remaining.motif.fillna("").str.startswith("hangs_")].nlargest(int(n * hangs_frac), "wp_loss")
    remaining = remaining.drop(hangs.index)
    rest = remaining.nlargest(n - len(mg) - len(hangs), "wp_loss")
    return pd.concat([mg, hangs, rest]).sort_values("wp_loss", ascending=False)


def make_puzzle(row, pgn_index, idx):
    """Build one puzzle as a chess.pgn.Game: position + solution + your move as a variation."""
    board = board_before_move(row.game_url, row.ply, pgn_index)
    if board is None:
        return None
    try:
        best_move = board.parse_san(row.best_san)
        blunder_move = board.parse_san(row.san)
    except ValueError:
        return None

    puzzle = chess.pgn.Game()
    puzzle.setup(board)
    puzzle.headers["Event"] = f"Drill {idx} - {row.motif} ({row.phase})"
    puzzle.headers["Site"] = row.game_url
    puzzle.headers["Date"] = row.end_time.strftime("%Y.%m.%d") if pd.notna(row.end_time) else "????.??.??"
    puzzle.headers["Round"] = str(idx)
    puzzle.headers["White"] = "?"
    puzzle.headers["Black"] = "?"
    puzzle.headers["Result"] = "*"

    solution = puzzle.add_variation(best_move)
    solution.comment = (f"Solution. You played {row.san} instead "
                         f"(wp_loss {row.wp_loss:.0f}, motif: {row.motif}).")
    puzzle.add_variation(blunder_move)
    return puzzle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--path", default="data/moves.parquet")
    ap.add_argument("--out", default="data/drills/worst_100.pgn")
    a = ap.parse_args()

    df = pd.read_parquet(a.path)
    worst = pick_worst(df, a.n)
    pgn_index = game_pgn_index()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    written, skipped = 0, 0
    with open(a.out, "w") as f:
        for idx, (_, row) in enumerate(worst.iterrows(), start=1):
            puzzle = make_puzzle(row, pgn_index, idx)
            if puzzle is None:
                skipped += 1
                continue
            print(puzzle, file=f, end="\n\n")
            written += 1

    print(f"wrote {written} puzzles to {a.out} ({skipped} skipped -- couldn't reconstruct/replay)")
    print(f"\nphase mix:\n{worst.phase.value_counts().to_string()}")
    print(f"\nmotif mix:\n{worst.motif.value_counts().to_string()}")


if __name__ == "__main__":
    main()
