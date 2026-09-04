# Claude Code prompt + GitHub setup

## 1. Paste this into Claude Code in your VS Code terminal

```
I'm building a chess improvement tool in this repo. Context:

- My Chess.com username is jayeed101. Goal: reach 1000 Elo this year.
- Existing files: chesscom.py (API client, caches monthly archives to data/raw/),
  build_dataset.py (PGN -> per-move rows with Stockfish centipawn loss, clock data
  from [%clk] tags, and blunder motif tags -> data/moves.parquet), report.py
  (aggregations), model.py (blunder-risk classifier + Elo leak valuation),
  compare.py (friend comparison from API data only).

Do these in order, and stop after each so I can check the output:

1. Set up the venv, install requirements.txt, install Stockfish, and confirm
   `python chesscom.py jayeed101` prints my ratings. Fix anything that breaks.

2. Run `python build_dataset.py jayeed101 --depth 12 --time-class blitz,rapid`.
   If it's slower than ~20 min, tell me before continuing rather than waiting.
   Report how many games and moves landed in data/moves.parquet.

3. Run report.py and model.py. Then read the actual output and tell me, in plain
   terms: what are my three biggest leaks, ranked by the Elo value the leak
   simulator assigns them? Don't just paste tables at me — interpret them.

4. Build a Streamlit dashboard (app.py) over data/moves.parquet:
   - sidebar filters: time control, date range, color, phase
   - KPI row: ACPL, blunder rate, best-move rate, current rating
   - blunder rate vs. clock-remaining chart
   - blunder rate by move number
   - motif breakdown (what I actually hang)
   - a table of worst moves where clicking a row renders the board with
     chess.svg, showing my move vs. the engine's best move

Rules:
- Don't add dependencies beyond requirements.txt + streamlit without asking.
- data/ stays gitignored.
- If an engine call is the bottleneck, profile before optimizing.
- Keep functions small enough that I can read them. I'd rather understand this
  than have it be clever.
```

That last block matters. Without it Claude Code will happily add four libraries and refactor everything into a class hierarchy you didn't ask for.

## 2. Push to GitHub

```bash
cd chess-improve

cat > .gitignore <<'EOF'
data/
.venv/
__pycache__/
*.pyc
*.parquet
.DS_Store
EOF

git init
git add .
git commit -m "chess.com game analysis pipeline: engine eval, blunder motifs, ML leak valuation"

# make the repo on github.com first (public, no README/gitignore — you have them)
git remote add origin https://github.com/jayeed17/chess-improve.git
git branch -M main
git push -u origin main
```

Or with the GitHub CLI, one line:

```bash
gh repo create chess-improve --public --source=. --push
```

### Before you push

- **Take your email out of `chesscom.py`.** The `UA` string is your contact info in a public repo. Either use your GitHub noreply address or read it from an env var:
  ```python
  UA = os.environ.get("CHESS_UA", "chess-improve/0.1 (github.com/jayeed17)")
  ```
- **Confirm `data/` is actually ignored** — `git status` before the first commit. Your parquet files are a few MB of your own game history; no harm, but they'll bloat the repo and they're regenerable.
- **Commit a sample output.** A `docs/sample_report.txt` with your real report.py output (or a screenshot of the dashboard) is what makes a recruiter stop scrolling. A repo of scripts with no visible result gets skipped.

### README pitch line

Your current README is a good internal doc but a weak pitch. Once you have real numbers, lead with the finding, not the setup:

> Analyzed 1,847 of my own Chess.com games with Stockfish to find where my rating actually leaks. Result: 38% of my blunders happen with under 30 seconds on the clock, worth an estimated +XX Elo if fixed — more than any opening study would return.

That's the same move you made with the bank churn project, where the pitch anchored on the transaction-count finding rather than the pipeline.
