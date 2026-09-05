"""PGN -> one row per move-you-played, with engine centipawn loss, clock, and motif tags.

Usage:  python build_dataset.py <username> [--depth 12] [--since 2025/01] [--workers 4]
Output: data/moves.parquet
"""
import argparse, glob, io, re, os, datetime as dt
from concurrent.futures import ProcessPoolExecutor

import chess, chess.pgn, chess.engine
import pandas as pd

import chesscom

ENGINE_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CLK = re.compile(r"\[%clk\s+(\d+):(\d+):([\d.]+)\]")


def clk_seconds(comment):
    m = CLK.search(comment or "")
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def phase(board):
    n = sum(len(board.pieces(p, c)) for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            for c in (chess.WHITE, chess.BLACK))
    if board.fullmove_number <= 12:
        return "opening"
    return "endgame" if n <= 6 else "middlegame"


def cp(score, pov):
    """Score -> centipawns from `pov`, mates clamped."""
    return score.pov(pov).score(mate_score=2000)


def hanging_value(board, color):
    """Total value of `color`'s pieces attacked by the opponent and not defended."""
    total = 0
    for sq, pc in board.piece_map().items():
        if pc.color != color or pc.piece_type == chess.KING:
            continue
        if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
            total = max(total, PIECE_VAL[pc.piece_type])
    return total


def motif(board_after, reply, me):
    """Cheap label for *what* the blunder allowed."""
    if reply is None:
        return "none"
    b = board_after.copy()
    gives_check = b.gives_check(reply)
    is_cap = b.is_capture(reply)
    victim = b.piece_at(reply.to_square)
    b.push(reply)
    if b.is_checkmate():
        return "allows_mate"
    if is_cap and victim and not board_after.is_attacked_by(me, reply.to_square):
        return f"hangs_{chess.piece_name(victim.piece_type)}"
    # fork: after the reply, 2+ of my pieces are attacked
    attacked = sum(1 for sq, pc in b.piece_map().items()
                   if pc.color == me and PIECE_VAL[pc.piece_type] >= 3
                   and b.is_attacked_by(not me, sq))
    if gives_check and attacked >= 1:
        return "allows_fork_check"
    if attacked >= 2:
        return "allows_double_attack"
    if is_cap:
        return "loses_material"
    return "positional"


def analyse_game(args):
    game_json, user, depth = args
    try:
        game = chess.pgn.read_game(io.StringIO(game_json["pgn"]))
    except Exception:
        return []
    if game is None:
        return []

    white = game_json["white"]["username"].lower()
    me = chess.WHITE if white == user else chess.BLACK
    my_res = game_json["white" if me == chess.WHITE else "black"]["result"]
    opp = game_json["black" if me == chess.WHITE else "white"]
    my_rating = game_json["white" if me == chess.WHITE else "black"]["rating"]
    eco = (game_json.get("eco") or "").rsplit("/", 1)[-1]
    end_ts = game_json.get("end_time")
    end_dt = dt.datetime.fromtimestamp(end_ts) if end_ts else None

    rows = []
    limit = chess.engine.Limit(depth=depth)
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as eng:
        eng.configure({"Threads": 1, "Hash": 64})
        board = game.board()
        prev_clk = None
        for node in game.mainline():
            move = node.move
            if board.turn != me:
                board.push(move)
                continue

            info = eng.analyse(board, limit)
            best_cp = cp(info["score"], me)
            best_move = info["pv"][0] if info.get("pv") else None

            before = board.copy()
            board.push(move)
            after_info = eng.analyse(board, limit)
            actual_cp = cp(after_info["score"], me)
            reply = after_info["pv"][0] if after_info.get("pv") else None

            loss = max(0, best_cp - actual_cp)
            c = clk_seconds(node.comment)
            spent = (prev_clk - c) if (prev_clk is not None and c is not None) else None
            prev_clk = c

            rows.append({
                "game_url": game_json["url"],
                "end_time": end_dt,
                "hour": end_dt.hour if end_dt else None,
                "weekday": end_dt.strftime("%a") if end_dt else None,
                "time_class": game_json.get("time_class"),
                "time_control": game_json.get("time_control"),
                "rules": game_json.get("rules"),
                "color": "white" if me == chess.WHITE else "black",
                "my_rating": my_rating,
                "opp_rating": opp["rating"],
                "rating_diff": my_rating - opp["rating"],
                "result": my_res,
                "eco": eco,
                "ply": before.ply(),
                "move_no": before.fullmove_number,
                "phase": phase(before),
                "san": before.san(move),
                "best_san": before.san(best_move) if best_move else None,
                "played_best": bool(best_move and move == best_move),
                "eval_before": best_cp,
                "eval_after": actual_cp,
                "cpl": loss,
                "blunder": loss >= 200,
                "mistake": 100 <= loss < 200,
                "inaccuracy": 50 <= loss < 100,
                "clock_left": c,
                "time_spent": spent,
                "in_check_before": before.is_check(),
                "was_capture": before.is_capture(move),
                "opp_last_was_capture": bool(before.move_stack and
                                             before.is_capture(before.move_stack[-1])),
                "my_hanging_before": hanging_value(before, me),
                "my_hanging_after": hanging_value(board, me),
                "material_diff": sum(
                    PIECE_VAL[p] * (len(before.pieces(p, me)) - len(before.pieces(p, not me)))
                    for p in PIECE_VAL),
                "motif": motif(board, reply, me) if loss >= 200 else None,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--since", default=None, help="YYYY/MM lower bound")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--time-class", default=None, help="filter e.g. blitz,rapid")
    a = ap.parse_args()
    user = a.username.lower()

    print("fetching archives...")
    games = chesscom.all_games(user, since=a.since)
    games = [g for g in games if g.get("rules") == "chess" and "pgn" in g]
    if a.time_class:
        keep = set(a.time_class.split(","))
        games = [g for g in games if g.get("time_class") in keep]
    print(f"analysing {len(games)} games at depth {a.depth} on {a.workers} workers...")

    # checkpoint every 5000 rows so a crash/interruption doesn't lose the whole run
    os.makedirs("data/parts", exist_ok=True)
    for old in glob.glob("data/parts/*.parquet"):
        os.remove(old)  # stale parts from a prior run would get concatenated back in below

    rows, part = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(analyse_game, [(g, user, a.depth) for g in games], chunksize=4)):
            rows.extend(r)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(games)} games, {len(rows)} moves", flush=True)
            if len(rows) >= 5000:
                pd.DataFrame(rows).to_parquet(f"data/parts/{part:04d}.parquet", index=False)
                rows, part = [], part + 1
                print(f"  checkpoint {part} @ game {i+1}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(f"data/parts/{part:04d}.parquet", index=False)

    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("data/parts/*.parquet"))],
                   ignore_index=True)
    df.to_parquet("data/moves.parquet", index=False)
    print(f"wrote data/moves.parquet: {len(df)} moves, {df.game_url.nunique()} games")


if __name__ == "__main__":
    main()
