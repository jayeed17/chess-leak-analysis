# chess-improve

Pull your Chess.com games → find where/when you blunder → model it → 1000 Elo.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stockfish (needed for move evaluation)
brew install stockfish          # mac
sudo apt install stockfish      # linux  -> binary lands in /usr/games/stockfish
export STOCKFISH_PATH=$(which stockfish)
```

Edit the `UA` string in `chesscom.py` — Chess.com 403s requests without a real User-Agent.

## Run

```bash
python chesscom.py YOUR_USERNAME                      # sanity check: ratings + game count
python build_dataset.py YOUR_USERNAME --depth 12      # -> data/moves.parquet
python report.py                                      # where + when you blunder
python model.py                                       # ML layer + Elo leak valuation
```

Depth 12 ≈ 0.15s/move. 500 blitz games ≈ 20k of your moves ≈ 20 min on 4 cores.
Start with `--since 2025/01 --time-class blitz,rapid` while you're iterating.

## Files

| file | what it does |
|---|---|
| `chesscom.py` | API client. Serial requests, exponential backoff on 429, past months cached to `data/raw/` so re-runs are free. |
| `build_dataset.py` | PGN → one row per move **you** played. Stockfish evals the position before and after, difference = centipawn loss. Also pulls `[%clk]` out of the PGN so you get clock-left and think-time per move, and tags every blunder with a motif (`hangs_queen`, `allows_mate`, `allows_fork_check`, ...). |
| `report.py` | The "where and when" cuts: phase, move number, ECO, color, clock bucket, think-time bucket, hour of day, opponent strength, plus your 15 worst single moves with links. |
| `model.py` | Gradient-boosted P(blunder) model with `GroupKFold` on game_url (no leakage from the same game). Permutation importance = what actually drives your blunders. Then a leak simulator: remove a category of blunder, recompute win rate, convert to Elo. |

## Key definitions

- **CPL** = `eval(best move) - eval(your move)`, from your side, clamped at ±2000 for mates.
- blunder ≥ 200cp, mistake 100–200, inaccuracy 50–100. Standard thresholds.
- Group your CV folds by game. Moves inside one game are massively correlated — random splits will give you a fake 0.95 AUC.

## What to add next

**High value, low effort**
1. **Opening book gap report.** For every position, check if you left theory (compare against a Lichess opening explorer dump for your rating band) and measure CPL by "in book / out of book". Below 1000 the answer is usually "you leave book on move 4 and lose 40cp immediately."
2. **Spaced-repetition drill deck.** Export your worst 100 positions to a `.pgn` with the correct move as the solution, import into Lichess Study or Chessable. Re-run monthly and drop positions you now solve.
3. **Opponent-mirror.** Same pipeline on your opponents' moves — tells you which of your losses were actually *your* fault vs. them just playing well. Below 1000 a lot of losses are "we both blundered, they blundered last."

**Medium**
4. **Time-management model.** Regress think-time against position complexity (number of legal moves, is-capture-available, eval volatility). Finds the "you think 30s on move 6 and then flag" pattern.
5. **Streak / tilt detection.** Order games by end_time, add `losses_in_last_3` as a feature. Tilt is a real, measurable, and *fixable* Elo leak.
6. **Puzzle-endpoint integration.** `/pub/puzzle` daily + your motif distribution → serve yourself puzzles matching the motif you fail most.

**Portfolio-grade**
7. **Streamlit dashboard.** Filters by time control and date range, embeds boards with `chess.svg`, click a row → see the position. This is the version you put on the portfolio site, not the CLI.
8. **Rating forecaster.** Bayesian state-space / Kalman filter on your rating series with blunder-rate as a covariate → credible interval on "when do I hit 1000." Much more honest than a linear trend.
9. **Move-quality model without an engine.** Train a small NN on `(board tensor) → CPL` from your labelled data. Now you can score positions instantly, and it's a real modelling story: representation choice (8x8x12 planes), CNN vs. gradient boosting on handcrafted features, calibration.
10. **Compare against a rating cohort.** Pull `/pub/country/US/players`, filter to ~800–1000, run the same pipeline on a sample. "My endgame ACPL is 40% worse than my rating peers" is a far sharper finding than an absolute number.

## API notes worth knowing

- All timestamps are Unix seconds.
- Archives are `/games/{YYYY}/{MM}` — past months never change, so cache aggressively.
- `accuracies` is only present on games Chess.com already analyzed. Don't rely on it; compute your own.
- Serial requests are unlimited. Parallel requests get 429s — the client here is serial for fetching and only parallelizes the local Stockfish work.
- Filter to `rules == "chess"` unless you want variants polluting the stats.
