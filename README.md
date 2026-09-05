# chess-leak-analysis

Analyzed ~1,024 of my own Chess.com games (rapid + blitz, 2026) with Stockfish at
depth 12-18 to find where my rating actually leaks — not where I assumed it did.

## Findings

**69.2% of my blunders are tactical oversights, and 26.7% are outright hangs.**
Not subtle positional drift — mostly "this piece was undefended and I moved
somewhere that let it be taken." That's the most robust finding in this project:
it doesn't depend on the leak simulator (see Limitations), and it holds after
correcting the win-probability metric that everything else here is built on.

**Middlegame is my worst phase, not the endgame.** Blunder rate by
win-probability loss: middlegame 9.4%, endgame 5.8%, opening 5.3%. My first pass
at this used raw centipawn loss and found the *opposite* — endgame looked worst
(ACPL 150). That was a measurement artifact (see below), not a real pattern.

**Time pressure is not the driver of my rating gradient (rapid 545 > blitz 513 >
bullet 439).** Blitz and rapid blunder at almost the same rate (12.7% vs 13.3%),
and think-time runs the wrong direction for a "rushing" story — blunder rate is
*higher* on moves I spent longer on (23.8% at 30s+ vs 8.3% at <1s), which is
reverse causation (hard positions cause both long think and blunders), not
evidence that slowing down helps. Low-clock blunders come out as the smallest of
five simulated leaks. Clock management is a real, minor lever — not the story.

**~1,024 games over 8 months with a flat-to-negative rating trend — volume isn't
the fix.** Whatever the leak is, playing more of the same hasn't touched it.

## What got retracted, and why that's here

Two findings didn't survive scrutiny and were pulled rather than kept:

- **"Endgame is my worst phase"** (step 3) was built on uncapped centipawn loss,
  which inflates exactly where evals are already extreme — i.e. endgames. Once
  moves were re-scored by win-probability loss instead, endgame dropped to the
  *best* phase and middlegame took its place. Uncapped CPL is still in the data
  (`cpl` column) for comparison; nothing here trusts it as a severity measure
  anymore.
- **The original brilliant-sacrifice phase gap** (step 5) was partly an
  artifact of its own denominator: `sac_while_winning` positions were never
  eligible to be labeled sound in the first place, and they land in the endgame
  denominator 4x more often than the opening one. Recomputed on the eligible
  denominator with Fisher's exact test, part of the gap survives (p=0.0085) and
  part was the artifact.

Two subgroup claims (piece type, time control) were also walked back after
Fisher's exact / Wilson CIs showed they rested on single-digit event counts —
not wrong exactly, just not supported by the data at this sample size.

A repo that shows its own corrections is worth more than one that only shows
conclusions. The commit history has all of this if you want the receipts.

## Limitations

`model.py`'s leak simulator (the "+N Elo" numbers) is a **priority hint, not a
causal estimate.** Two independent problems, not one:

1. **Reverse causation.** It's built on the empirical relationship between
   blunder count and win rate in a game. Losing positions produce more
   blunders — tilt, desperation, a position that was already falling apart —
   not only the other way around. A confidence interval around this estimate
   quantifies sampling noise, not this bias; it makes the number more precise
   without making it more true.
2. **The curve is a step function, not a gradient.** At ~2.2 blunders/game,
   win rate is ~82% at zero blunders and ~40-50% for *any* nonzero count, with
   no statistically detectable difference between having 1, 2, 3, 4, or 5+.
   So a leak's simulated Elo value mostly reflects how often that blunder type
   is the *one* blunder standing between a 0-blunder game and a 1-blunder game
   — not how objectively damaging it is. Two leak categories can show
   overlapping confidence intervals and still both be "real" in the sense that
   fixing either would plausibly help; the ranking just can't tell you which
   one deserves credit for a given game.

Use the leak ranking to decide what to drill first. Don't read the Elo numbers
as a forecast of what fixing something is worth.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stockfish (needed for move evaluation)
brew install stockfish          # mac
sudo apt install stockfish      # linux  -> binary lands in /usr/games/stockfish
export STOCKFISH_PATH=$(which stockfish)
```

Set `CHESS_UA` to your own contact info — Chess.com 403s requests without a
real User-Agent (`chesscom.py` falls back to a generic one otherwise).

## Run

```bash
python chesscom.py YOUR_USERNAME                            # sanity check: ratings + game count
python build_dataset.py YOUR_USERNAME --depth 12             # -> data/moves.parquet
python report.py                                             # where + when you blunder
python model.py                                              # ML layer + Elo leak valuation (see Limitations)
python brilliancy.py YOUR_USERNAME --depth 18                 # sound-sacrifice / brilliancy detection
python drill.py                                               # -> data/drills/worst_100.pgn, import into a Lichess Study
streamlit run app.py                                          # interactive dashboard over moves.parquet
```

Depth 12 ≈ 0.15s/move. ~1,000 blitz+rapid games ≈ 32k of your moves ≈ 15-20 min
on 7-8 cores. Start with `--since 2026/01 --time-class blitz,rapid` while
iterating. `brilliancy.py` is slower per-candidate (depth 18, multipv=2) but
only runs on the ~1-2% of moves that offer material, so a full run still lands
around an hour, not days — check progress before assuming it's stuck.

## Files

| file | what it does |
|---|---|
| `chesscom.py` | API client. Serial requests, exponential backoff on 429, past months cached to `data/raw/` so re-runs are free. |
| `build_dataset.py` | PGN → one row per move **you** played. Stockfish evals the position before and after; stores both raw centipawn loss (`cpl`) and win-probability loss (`wp_loss`, the one that isn't skewed by already-decided positions). Also pulls `[%clk]` for clock/think-time, and tags every move's motif (`hangs_queen`, `allows_mate`, `allows_fork_check`, ...). Checkpoints every 5,000 rows to `data/parts/` so an interruption doesn't lose the whole run. |
| `report.py` | The "where and when" cuts: phase, move number, ECO, color, clock bucket, think-time bucket, hour of day, opponent strength, plus your worst single moves with links. |
| `model.py` | Gradient-boosted P(blunder) model with `GroupKFold` on game_url (no leakage from the same game). Permutation importance = what actually drives your blunders. Then a leak simulator (see Limitations) with a bootstrap 95% CI on every leak's Elo estimate. |
| `brilliancy.py` | Two-pass sacrifice finder: cheap board-logic filter, then depth-18 multipv=2 verification on the ~1-2% of moves that offer material. Reports sound-sac hit rate by piece/time-control/phase with Wilson CIs, and flags which comparisons are actually statistically supported. |
| `drill.py` | Exports your worst 100 positions (by `wp_loss`, weighted toward middlegame and hangs_* blunders) as a multi-game PGN — FEN start, engine's best move as the solution, your actual move as a side variation. Import straight into a Lichess Study. |
| `app.py` | Streamlit dashboard over `data/moves.parquet`: filters, KPIs, blunder-rate charts, motif breakdown, and a worst-moves table that renders the board (`chess.svg`) on row click. |
| `compare.py` | Friend comparison from API data only — no Stockfish, runs in seconds. |

## Key definitions

- **CPL** = `eval(best move) - eval(your move)`, from your side, clamped at ±2000 for mates. Cheap, standard, but overstates severity in already-decided positions (a 900cp → 400cp swing is still a totally winning position, not a 500cp disaster).
- **wp_loss** = the same comparison run through a logistic win-probability curve first. This is the metric everything in this README's Findings section is based on; `cpl` is kept in the data for comparison, not as the primary severity measure.
- Blunder ≥ 200cp (`cpl` basis) or ≥20 win-probability points (`wp_loss` basis) — the two thresholds catch overlapping but not identical sets of moves; see the commit history for how much that overlap matters.
- Group your CV folds by game. Moves inside one game are massively correlated — random splits will give you a fake 0.95 AUC.

## What's next

- **Opponent-mirror.** Same pipeline on opponents' moves in the same games — separates "I threw this game" from "we both blundered, they blundered last."
- **Tilt detection.** Add `losses_in_last_3` as a feature, re-run `model.py`. Given the reverse-causation problem above, this might explain *why* the blunder-count curve looks the way it does, not just add a feature.

## API notes worth knowing

- All timestamps are Unix seconds.
- Archives are `/games/{YYYY}/{MM}` — past months never change, so cache aggressively.
- `accuracies` is only present on games Chess.com already analyzed. Don't rely on it; compute your own.
- Serial requests are unlimited. Parallel requests get 429s — the client here is serial for fetching and only parallelizes the local Stockfish work.
- Filter to `rules == "chess"` unless you want variants polluting the stats.
