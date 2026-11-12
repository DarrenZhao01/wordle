# Architecture Decision Records (ADR) — Wordle Agent & Terminal Prototype

This document records the core architectural, algorithmic, and experimental design decisions made during the implementation of the Wordle Simulator, Baseline Agents, Information-Gain Solver, and Interactive Assistant.

---

## ADR-001: Base-3 (`uint8`) Feedback Encoding and Hash-Validated Batched Pattern Matrix

- **Status**: Accepted
- **Context**:
  - Wordle feedback assigns one of three states (`Gray = 0`, `Yellow = 1`, `Green = 2`) to each of the 5 letter positions, yielding $3^5 = 243$ possible patterns.
  - Evaluating all $|G| = 12,972$ allowed guesses against $|A| = 2,315$ possible answers requires $30,030,180$ feedback evaluations on Turn 1 alone. Recomputing string feedback online or using unvalidated disk caches risks either prohibitive latency or silent corruption if word lists change.
  - Furthermore, naive 3D NumPy broadcasting over `(12972, 2315, 5)` can spike intermediate memory consumption.
- **Decision**:
  1. Encode each 5-position feedback pattern as an unsigned byte $\text{pattern\_id} = \sum_{i=0}^{4} t_i \cdot 3^i \in [0, 242]$ (`dtype=np.uint8`), where `242` represents `ALL_GREEN`.
  2. Keep a pure-Python two-pass scalar reference implementation (`compute_feedback_scalar` in `src/engine.py`) as the ground-truth specification, tested on hand-verified duplicate-letter cases (`ALLEY` vs. `APPLE`, `SPEED` vs. `STEAL`, `EERIE` vs. `ELITE`).
  3. Generate `data/pattern_matrix.npy` (`(12972, 2315)`, $\approx 30.0\text{ MB}$) in guess batches (`batch_size = 512`) and maintain both row-contiguous `(12972, 2315)` and column-contiguous transposed `(2315, 12972)` views in memory.
  4. Persist SHA-256 hashes of the raw files and canonical sorted word lists, array shapes/dtypes, and encoding version in `data/cache_meta.json`, rejecting and rebuilding the cache on any mismatch.
- **Alternatives Considered**:
  - *String/tuple feedback representations*: 5–10x larger memory footprint and cannot index directly into `np.bincount` lookup tables.
  - *Full unbatched 3D vectorization*: Allocates large temporary arrays across $30\text{M} \times 5$ elements during cache generation.
- **Consequences**:
  - Disk cache loads and validates in $\approx 10.5\text{ ms}$.
  - Maintaining the transposed `(2315, 12972)` view alongside column-sorted run-length entropy accumulation for small candidate sets ($|C| \le 80$) reduced mean online selection time for the Entropy Agent to $4.031\text{ ms/guess}$.

---

## ADR-002: Caching the Full Turn-1 Entropy Vector Instead of a Single Opening Word

- **Status**: Accepted
- **Context**:
  - The Entropy Agent scores guesses using $\text{Score}(g, C) = H(g, C) + \varepsilon \cdot \mathbb{I}(g \in C)$.
  - On Turn 1 ($C = A$, $|A| = 2,315$), the highest-entropy non-answer guess (`soare`, $H = 5.885960\text{ bits}$) exceeds the highest-entropy candidate answer (`raise`, $H = 5.877910\text{ bits}$) by only $\Delta H = 0.008050419557\text{ bits}$.
  - Caching a single fixed opening word would silently invalidate hyperparameter tuning whenever $\varepsilon$ crosses $\Delta H$.
- **Decision**:
  - Precompute and cache the full 1D vector of Turn-1 entropies `data/opening_entropies.npy` (`shape = (12972,)`, `dtype = float64`, $\approx 104\text{ KB}$), governed by `data/cache_meta.json`.
  - On Turn 1, `EntropyAgent` computes `opening_entropies + epsilon * in_c_mask` (and applies the `allow_non_answers` filter) dynamically over the cached vector.
- **Alternatives Considered**:
  - *Hardcoding or caching a single opening string*: Fails to switch from `soare` ($\varepsilon < 0.00805$) to `raise` ($\varepsilon \ge 0.00805$).
  - *Recomputing Turn-1 partitions from `pattern_matrix` at the start of every game*: Adds $\approx 40\text{ ms}$ of redundant computation to every single game during 2,315-word benchmarks.
- **Consequences**:
  - Turn-1 selection takes $\approx 0.1\text{ ms}$ while remaining mathematically exact for any $\varepsilon \ge 0$ and for both full-dictionary and candidate-only modes.

---

## ADR-003: Candidate-Only Baselines (`C_idx`) and Candidate-Restricted Entropy Ablation

- **Status**: Accepted
- **Context**:
  - In Cyrus Freshman's dictionaries, $10,657$ of the $12,972$ allowed words ($82.2\%$) are non-answers.
  - Allowing simple baselines (Random-Consistent and Letter-Frequency) to guess from the full $12,972$ pool complicates state tracking and confounds the evaluation of the scoring function with the use of non-answer guesses.
  - Furthermore, for Letter-Frequency scoring, adding ad-hoc duplicate-letter penalties obscures the baseline's mathematical definition.
- **Decision**:
  1. Restrict `RandomConsistentAgent` and `LetterFrequencyAgent` to select among the current remaining possible answers `c_idx` ($C \subseteq A$).
  2. Define the primary `LetterFrequencyAgent` using the direct positional frequency sum $\sum_{i=0}^{4} P(c[i] = g[i] \mid c \in C)$, providing the duplicate-discounted score (`unique_letters_only=True`) as an explicit ablation variant (`letter_frequency_unique`).
  3. Add an explicit `allow_non_answers: bool` flag to `EntropyAgent` so `entropy_candidates_only` (`allow_non_answers=False`) restricts guesses to `c_idx`, cleanly isolating the benefit of the entropy objective from the benefit of exploratory non-answer guesses.
- **Consequences & Empirical Validation**:
  - Over the exact same candidate pool $C$, direct positional frequency (`3.6397` avg, `14` losses) slightly outperformed duplicate-discounted frequency (`3.6423` avg, `15` losses), and `entropy_candidates_only` improved further to `3.5983` avg (`11` losses).
  - Enabling full-dictionary exploratory guesses in `entropy` (`allow_non_answers=True`) reduced average guesses from `3.5983` to `3.4713` and eliminated all `11` remaining losses (`100.00%` win rate).

---

## ADR-004: Explicit Stateful Agent Lifecycle and Final-Turn Rule (`turns_remaining == 1`)

- **Status**: Accepted
- **Context**:
  - Passing full game history lists into stateless `select_guess(history)` calls blurs where candidate filtering occurs and makes it harder to detect inconsistent state transitions.
  - Additionally, a pure information-gain policy without turn-horizon awareness can select an exploratory non-answer word on Turn 6 (`turns_remaining == 1`), which has a $0\%$ probability of winning the game.
- **Decision**:
  1. Enforce a strict four-step lifecycle between `WordleSimulator` and `BaseAgent`:
     `agent.reset(seed)` $\to$ `agent.select_guess(turns_remaining)` $\to$ `simulator.step(guess)` $\to$ `agent.observe(guess, pattern_id)`.
  2. Raise a diagnostic `RuntimeError` inside `agent.observe()` if an observation ever reduces `len(c_idx)` to `0` during simulation.
  3. When `turns_remaining == 1`, `EntropyAgent` explicitly masks its candidate pool to `c_idx`, choosing the highest-entropy remaining candidate answer (breaking ties alphabetically) as a deterministic tie-break among candidates with equal $1/|C|$ immediate win probability.
- **Alternatives Considered**:
  - *Setting $\varepsilon = \infty$ on the final turn*: Multiplying `np.inf * in_c_mask` produces IEEE 754 `NaN` (`inf * 0.0`) for non-answers. Using an explicit index mask (`c_guess_indices`) avoids floating-point hazards cleanly.
- **Consequences**:
  - Guarantees that the agent never wastes its 6th and final guess on a known non-answer while maintaining deterministic tie-breaking.

---

## ADR-005: Split-Invariant Candidate Initialization and Hash-Derived Per-Game Seeding

- **Status**: Accepted
- **Context**:
  - Evaluating or tuning on a subset of targets (such as the 463-word Development split or 1,852-word Test split) must never restrict the agent's initial candidate dictionary to that split, which would leak target-pool information.
  - For stochastic baselines (`RandomConsistentAgent`), running a sequential RNG across a subset of targets would produce different per-target trajectories than running across all $2,315$ words.
- **Decision**:
  1. `agent.reset()` always initializes `c_idx` to all $2,315$ answers (`np.arange(2315)`), verified by an explicit runtime check in `play_single_game`.
  2. Derive each game's RNG seed deterministically from `(run_seed, target_word)` via:
     `int.from_bytes(hashlib.sha256(f"{run_seed}:{target_word}".encode("utf-8")).digest()[:8], "big", signed=False)`.
  3. Run each `(run_seed, target_word)` pair once, persist the canonical per-game logs (`results/game_logs_*.jsonl`), and compute Dev (463), Test (1,852), and All (2,315) summaries by slicing those records.
- **Consequences**:
  - Guarantees $100\%$ identical trajectories for any target word regardless of whether it is run via `main.py play --target`, Dev tuning, or full benchmark evaluation.

---

## ADR-006: Two-Tier Candidate Pool Fallback ($2,315 \to 12,972$) in the Interactive Assistant

- **Status**: Accepted
- **Context**:
  - While Josh Wardle's original Wordle used the $2,315$-word solution list (`wordle-answers-alphabetical.txt`), the New York Times editorial team now occasionally selects daily Wordle answers from the broader $12,972$ allowed-guess dictionary (e.g., `KARAT`, `BALSA`, `GUANO`, `KAZOO`, `LASER`, `PANKO`, `SNAFU`).
  - During live testing on NYT Wordle (`KARAT`) with guesses `CRANE` (`01100`) $\to$ `TOILS` (`10000`) $\to$ `DUMPY` (`00000`), the first two guesses reduced the $2,315$ list to `PARTY, RATTY, WARTY`, and `DUMPY` (`00000`) eliminated all three—triggering a false `0 possible answers` warning even though `KARAT` (and 5 other words in the $12,972$ dictionary) matched all feedback.
- **Decision**:
  - In `src/assistant.py`, maintain both the primary candidate set `agent.c_idx` ($\subseteq 2,315$ answers) and a parallel consistency filter `expanded_pool` ($\subseteq 12,972$ allowed words).
  - Keep the $2,315$ candidate set as the primary pool so standard Wordle games benefit from the curated answer prior (avoiding obscure Scrabble words on early turns).
  - If a user-entered feedback pattern leaves `0` matches in the $2,315$ list **but $>0$ matches in `expanded_pool`**, notify the user and automatically transition the session to `expanded_pool` (computing exact entropies and candidate memberships over the matching words from the $12,972$ dictionary).
  - Reject feedback as a user typo only when `0` words match in **both** the $2,315$ list and the full $12,972$ dictionary, and preserve full `undo` support across tier transitions.
- **Alternatives Considered**:
  - *Treating all $12,972$ words as equally likely answers from Turn 1*: Degrades recommendation quality on 99%+ of games by treating obscure plurals and rare words (`AAHED`, `ZYMIC`) as equally likely solutions from the start.
  - *Hard-failing when the $2,315$ list hits $0$*: Leaves the user stranded whenever NYT Wordle picks an expanded-dictionary word like `KARAT`.
- **Consequences**:
  - Preserves optimal $3.47$-guess performance on standard solutions while seamlessly handling NYT expanded-dictionary puzzles like `KARAT` without false error blocks.

