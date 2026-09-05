"""Find your brilliant moves.

Two-pass so it stays cheap:
  pass 1 - replay cached PGNs, flag every move where you OFFERED material.
           Pure board logic, no engine. Catches ~1-2% of moves.
  pass 2 - deep-verify only those candidates at high depth with multipv=2.
           A real brilliancy is usually the *only* move that works, which is
           what multipv=2 measures.

Depth matters here in a way it doesn't for blunders. A sacrifice is exactly the
kind of move a shallow search misjudges, so this defaults to depth 18. Because
only candidates get analysed, that's still fast.

Usage: python brilliancy.py jayeed101 [--depth 18] [--since 2026/01]
Output: data/brilliancies.parquet + a report
"""
import argparse, io, math, os, datetime as dt

import chess, chess.pgn, chess.engine
import pandas as pd

import chesscom

ENGINE_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def win_pct(cp_val):
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp_val)) - 1)


def offers_material(board, move):
    """True if this move puts material at risk. Deliberately loose — pass 2 filters."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type == chess.KING:
        return False
    victim = board.piece_at(move.to_square)
    gained = VALS[victim.piece_type] if victim else 0
    risk = VALS[piece.piece_type]
    if risk - gained < 2:          # not enough at stake to be a sacrifice
        return False
    b = board.copy()
    b.push(move)
    sq = move.to_square
    attackers = b.attackers(not board.turn, sq)
    if not attackers:
        return False
    defenders = b.attackers(board.turn, sq)
    if not defenders:
        return True                # left hanging outright
    cheapest = min(VALS[b.piece_at(s).piece_type] for s in attackers)
    return cheapest < risk         # can be taken by something cheaper


def scan_game(game_json, user):
    """Pass 1. Returns list of (board_before, move, node, ply)."""
    try:
        game = chess.pgn.read_game(io.StringIO(game_json["pgn"]))
    except Exception:
        return []
    if game is None:
        return []
    me = chess.WHITE if game_json["white"]["username"].lower() == user else chess.BLACK
    out, board = [], game.board()
    for node in game.mainline():
        mv = node.move
        if board.turn == me and offers_material(board, mv):
            out.append((board.copy(), mv, board.ply()))
        board.push(mv)
    return out


def classify(eng, board, move, depth):
    """Pass 2. Deep-verify one candidate."""
    limit = chess.engine.Limit(depth=depth)
    me = board.turn
    infos = eng.analyse(board, limit, multipv=2)
    best_cp = infos[0]["score"].pov(me).score(mate_score=2000)
    second_cp = (infos[1]["score"].pov(me).score(mate_score=2000)
                 if len(infos) > 1 else best_cp)

    after = board.copy()
    after.push(move)
    actual_cp = eng.analyse(after, limit)["score"].pov(me).score(mate_score=2000)

    wp_loss = max(0.0, win_pct(best_cp) - win_pct(actual_cp))
    margin = best_cp - second_cp          # how much better best is than the runner-up

    # already winning easily, or already lost -> not a brilliancy either way
    if best_cp > 500:
        label = "sac_while_winning"
    elif actual_cp < -100:
        label = "unsound_sac"
    elif wp_loss > 5:
        label = "unsound_sac"
    elif margin >= 100:
        label = "brilliant"           # sound sac AND essentially the only move
    else:
        label = "sound_sac"           # sound, but other moves also worked
    return label, wp_loss, best_cp, actual_cp, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--since", default="2026/01")
    ap.add_argument("--time-class", default="rapid,blitz")
    a = ap.parse_args()
    user = a.username.lower()
    keep = set(a.time_class.split(","))

    games = [g for g in chesscom.all_games(user, since=a.since)
             if g.get("rules") == "chess" and g.get("time_class") in keep and "pgn" in g]
    print(f"scanning {len(games)} games for material offers...")

    cands = []
    for g in games:
        for board, mv, ply in scan_game(g, user):
            cands.append((g, board, mv, ply))
    print(f"{len(cands)} candidate sacrifices -> verifying at depth {a.depth}")

    rows = []
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as eng:
        eng.configure({"Threads": os.cpu_count() or 2, "Hash": 256})
        for i, (g, board, mv, ply) in enumerate(cands):
            label, wp_loss, best_cp, actual_cp, margin = classify(eng, board, mv, a.depth)
            me = board.turn
            piece = board.piece_at(mv.from_square)
            end = dt.datetime.fromtimestamp(g["end_time"])
            side = "white" if me == chess.WHITE else "black"
            rows.append({
                "game_url": g["url"], "end_time": end, "hour": end.hour,
                "time_class": g.get("time_class"), "color": side,
                "result": g[side]["result"],
                "won": g[side]["result"] == "win",
                "move_no": board.fullmove_number,
                "phase": "opening" if board.fullmove_number <= 12 else (
                    "endgame" if sum(len(board.pieces(p, c)) for p in
                                     (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                                     for c in (True, False)) <= 6 else "middlegame"),
                "san": board.san(mv),
                "piece": chess.piece_name(piece.piece_type),
                "piece_value": VALS[piece.piece_type],
                "was_capture": board.is_capture(mv),
                "label": label, "wp_loss": round(wp_loss, 1),
                "eval_before": best_cp, "eval_after": actual_cp, "margin": margin,
                "my_rating": g[side]["rating"],
                "opp_rating": g["black" if side == "white" else "white"]["rating"],
            })
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(cands)}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/brilliancies.parquet", index=False)

    print("\n" + "=" * 60)
    print("SACRIFICE BREAKDOWN\n")
    print(df.label.value_counts().to_string())
    good = df[df.label.isin(["brilliant", "sound_sac"])]
    print(f"\n{len(good)} sound sacrifices ({len(df[df.label=='brilliant'])} brilliant) "
          f"out of {len(df)} attempts — {100*len(good)/max(len(df),1):.1f}% hit rate")

    if len(good):
        print("\nWHICH PIECE YOU SACRIFICE")
        print(df.pivot_table(index="piece", columns="label", values="san",
                             aggfunc="size", fill_value=0).to_string())

        print("\nBY TIME CONTROL — sound-sac hit rate")
        hit = df.assign(sound=df.label.isin(["brilliant", "sound_sac"]))
        print((hit.groupby("time_class").sound.agg(
            attempts="size", sound_pct=lambda s: round(100 * s.mean(), 1))).to_string())

        print("\nBY PHASE")
        print((hit.groupby("phase").sound.agg(
            attempts="size", sound_pct=lambda s: round(100 * s.mean(), 1))).to_string())

        print("\nDID YOU CONVERT IT?")
        conv = good.groupby("label").won.agg(games="size",
                                             win_pct=lambda s: round(100 * s.mean(), 1))
        print(conv.to_string())
        print(f"\n  overall win rate in these games: {100*good.won.mean():.1f}%")
        print(f"  your baseline win rate: see report.py (~52-58%)")

        print("\nYOUR BRILLIANCIES")
        b = df[df.label == "brilliant"].nlargest(15, "margin")
        print(b[["end_time", "time_class", "move_no", "san", "piece",
                 "margin", "won", "game_url"]].to_string(index=False))

    print("\nwrote data/brilliancies.parquet")


if __name__ == "__main__":
    main()
