# Wordle Solver & Interactive Assistant

A terminal Wordle simulator, Shannon entropy solver, and interactive helper supporting both the **Modern NYT dictionary** (`3,158` answers / `14,855` guesses, default) and the **Original 2021 dictionary** (`2,315` answers / `12,972` guesses via `--original`).

## Quick Start

```powershell
# 1. Interactive Assistant (External Wordle Helper — e.g., NYT Wordle)
.\.venv\Scripts\python.exe main.py assistant

# 2. Practice Mode (Play against a hidden word with live recommendations)
.\.venv\Scripts\python.exe main.py assistant --random-target
.\.venv\Scripts\python.exe main.py assistant --target karat

# 3. Watch an Agent Play Automatically
.\.venv\Scripts\python.exe main.py play --agent entropy --target karat,watch
```

### Assistant Controls
- **Guess prompt**: Press `Enter` to use the `#1` recommended word, or type any 5-letter word.
- **Feedback prompt**: Enter 5 symbols (`G`/`2` = Green, `Y`/`1` = Yellow, `.`/`B`/`0` = Gray — e.g., `01100` or `.YY..`).
- **Commands**: `undo` (or `back`) to step back a turn, `q` to quit.

## Benchmarks, Tuning & Tests

```powershell
# Run unit & property tests
.\.venv\Scripts\pytest.exe -v

# Tune epsilon on the 20% dev split
.\.venv\Scripts\python.exe main.py tune

# Run full benchmark evaluation across all agents
.\.venv\Scripts\python.exe main.py evaluate
```

> **Tip**: Add `--original` to any command (`assistant`, `play`, `tune`, `evaluate`) to use the classic 2021 `2,315`-word answer list instead of the Modern NYT default.

