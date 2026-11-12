# Wordle Agent & Terminal Prototype — Implementation Plan (Weeks 1–4)

## 1. Architecture & Directory Layout

```text
d:/Github/wordle/
├── data/
│   ├── wordle-answers-alphabetical.txt   # 2,315 possible hidden answers (A)
│   ├── wordle-allowed-guesses.txt        # 10,657 additional valid guesses
│   ├── splits.json                       # Pinned dev (463) / test (1,852) word lists & hashes
│   ├── pattern_matrix.npy                # Cached (12972, 2315) uint8 feedback matrix
│   ├── opening_entropies.npy             # Cached (12972,) float64 Turn-1 entropy vector
│   └── cache_meta.json                   # Cache validation metadata (hashes, ordering, version)
├── src/
│   ├── __init__.py
│   ├── dictionary.py                     # Word list loader, SHA-256 verification & split builder
│   ├── engine.py                         # Scalar reference feedback, pattern encoding & Simulator
│   ├── precompute.py                     # Batched pattern matrix & opening entropy cache builder
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                       # Abstract BaseAgent with explicit lifecycle hooks
│   │   ├── random_agent.py               # Random-Consistent Baseline Agent (over C_idx)
│   │   ├── frequency_agent.py            # Positional Letter-Frequency Baseline Agent (over C_idx)
│   │   └── entropy_agent.py              # Information-Gain (Shannon Entropy) Solver
│   ├── assistant.py                      # Interactive Assistant & 2,315 -> 12,972 NYT fallback
│   └── evaluate.py                       # Reproducible benchmark runner, logs & split aggregation
├── tests/
│   └── test_wordle.py                    # Correctness tests for feedback, cache, state & assistant
├── results/                              # Generated JSON summaries and per-game CSV/JSONL logs
├── PLAN.md                               # Implementation plan & locked-in specifications
├── REPORT.md                             # Full technical write-up & experimental results
├── ADR.md                                # Architecture Decision Records (ADR-001 through ADR-006)
└── main.py                               # Unified CLI (`play`, `assistant`, `tune`, `evaluate`)
```

---

## 2. Locked-In Design Specifications

### 2.1 Dictionary & Evaluation Splits (`src/dictionary.py`)
- **Dictionaries**:
  - `answers` ($A$): 2,315 sorted 5-letter lowercase words from `wordle-answers-alphabetical.txt`.
  - `allowed_guesses` ($G$): 12,972 sorted 5-letter lowercase words formed by $A \cup \text{non\_answer\_guesses}$.
  - Computes and records the SHA-256 hash of both raw files and the canonical sorted word lists.
- **Target Splits vs. Agent Candidate Pool**:
  - Using a fixed split seed (`split_seed = 42`), $A$ is partitioned into:
    - **Development Split (20%)**: 463 target words.
    - **Test Split (80%)**: 1,852 target words.
  - Both splits and their hashes are persisted to `data/splits.json`.
  - **Strict Invariant**: Every game—whether evaluated on Dev, Test, or All—initializes the agent's candidate answer set `C_idx` to all 2,315 possible answers. The split only selects which hidden targets are evaluated or aggregated.

### 2.2 Feedback Rules & Simulator (`src/engine.py`)
- **Base-3 Pattern Encoding (`uint8`)**:
  - Each position $i \in \{0, 1, 2, 3, 4\}$ has trit $t_i \in \{0\text{ (Gray)}, 1\text{ (Yellow)}, 2\text{ (Green)}\}$.
  - Encoded as $\text{pattern\_id} = \sum_{i=0}^{4} t_i \cdot 3^i \in [0, 242]$, with `ALL_GREEN = 242`.
- **Two-Pass Scalar Reference Implementation (`compute_feedback_scalar`)**:
  - Pass 1 marks exact matches ($t_i = 2$) and counts unmatched letters in the hidden answer.
  - Pass 2 iterates $i = 0 \dots 4$ for non-green positions; if the guessed letter has a positive unmatched count in the answer pool, marks $t_i = 1$ and decrements the pool count; otherwise marks $t_i = 0$.
- **Simulator (`WordleSimulator`)**:
  - Owns `target`, `max_turns = 6`, `turns_used`, and `is_over`.
  - Validates that `guess` is in $G$ (raises `ValueError` if outside $G$) and that the game has not ended (raises `RuntimeError` if `step()` is called after solving or exhausting 6 turns).
  - Returns `StepResult(guess, pattern_id, turn_number, turns_remaining, solved, game_over)`.

### 2.3 Validated Pattern & Opening-Entropy Cache (`src/precompute.py`)
- **Batched Pattern Matrix Generation**:
  - Computes `pattern_matrix.npy` of shape `(12972, 2315)` and dtype `np.uint8` in guess batches (e.g., 512 guesses per batch) to keep peak intermediate memory small.
- **Opening-Entropy Vector Cache**:
  - Computes and stores `opening_entropies.npy` of shape `(12972,)` and dtype `np.float64`, representing the exact one-step Shannon entropy $H(g, A)$ for every $g \in G$ against all 2,315 initial answers.
  - This allows any $\varepsilon$ setting or candidate restriction flag to select the highest-scoring opening word under its exact policy without recomputing Turn-1 partitions.
- **Cache Validation (`cache_meta.json`)**:
  - Stores `encoding_version`, `entropy_version`, `answers_sha256`, `guesses_sha256`, `shape`, `dtype`, and first/last word checks. Loading rejects any mismatched cache and regenerates cleanly.

### 2.4 Agent Lifecycle & Policies (`src/agents/`)
- **Explicit Agent Interface**:
  ```python
  agent.reset(seed=game_seed)
  guess = agent.select_guess(turns_remaining=simulator.turns_remaining)
  result = simulator.step(guess)
  agent.observe(guess, result.pattern_id)
  ```
- **Candidate Update & Diagnostic Check**:
  - `observe(guess, pattern_id)` updates `C_idx = C_idx[M[g_idx, C_idx] == pattern_id]`.
  - If `len(C_idx) == 0`, raises a `RuntimeError` diagnostic exception immediately.
  - Whenever `len(C_idx) == 1`, every agent returns the sole remaining answer in `C_idx`.
- **Baseline 1: Random-Consistent Agent (`RandomConsistentAgent`)**:
  - Selects uniformly at random from the current remaining possible answers `C_idx` using a local `random.Random(game_seed)` instance.
- **Baseline 2: Positional Letter-Frequency Agent (`LetterFrequencyAgent`)**:
  - Scores and selects among the current remaining possible answers `C_idx`.
  - Primary baseline score (with `unique_letters_only: bool = False` by default):
    $$\text{Score}(g) = \sum_{i=0}^{4} P(c[i] == g[i] \mid c \in C)$$
  - Optional flag `unique_letters_only: bool = True` keeps only the highest positional frequency per distinct letter in $g$.
  - Ties are broken alphabetically.
- **Primary Solver: Information-Gain Entropy Agent (`EntropyAgent`)**:
  - Configurable with `epsilon: float` and `allow_non_answers: bool = True`:
    - When `allow_non_answers=True`, evaluates all $g \in G$ ($12,972$ words).
    - When `allow_non_answers=False`, restricts candidate guesses strictly to the current remaining candidate answers `C_idx`.
  - **One-Step Shannon Entropy**:
    $$H(g, C) = -\sum_{p \,:\, n_p(g) > 0} \frac{n_p(g)}{|C|} \log_2 \frac{n_p(g)}{|C|}$$
  - **Near-Tie Answer Preference ($\varepsilon$)**:
    $$\text{Score}(g, C) = H(g, C) + \varepsilon \cdot \mathbb{I}(g \in C)$$
    With exact ties resolved in favor of $g \in C$, then alphabetically.
  - **Final-Turn Rule (`turns_remaining == 1`)**:
    - When `turns_remaining == 1`, candidate guesses are explicitly restricted via index mask to `C_idx`, selecting the highest-entropy candidate within `C_idx` and breaking ties alphabetically (a deterministic tie-break among candidates with equal $1/|C|$ immediate win probability).

### 2.5 Reproducible Evaluation Protocol (`src/evaluate.py`)
- **Deterministic Per-Game Seeding**:
  - For any `run_seed` and `target_word`:
    ```python
    digest = hashlib.sha256(f"{run_seed}:{target_word}".encode("utf-8")).digest()
    game_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    ```
- **Single-Pass Canonical Game Records**:
  - Evaluations run over the 2,315 answers and store complete per-game records (`target`, `split`, `agent_config`, `run_seed`, `game_seed`, `guesses`, `pattern_ids`, `candidate_counts_after_turn`, `solved`, `turns_used`, ` penalized_guesses`, `turn_times_ms`).
  - Dev (463), Test (1,852), and All (2,315) metrics are computed directly by filtering those canonical game records.
- **Aggregated Metrics**:
  1. **Average Guesses (All Games)**: Unsolved games after 6 guesses stop immediately and receive penalty `7`.
  2. **Average Guesses (Won Games Only)**: Mean guesses across solved games ($\le 6$ turns).
  3. **Win Rate (%)**: Percentage of games solved within 6 guesses. (For Random-Consistent across 10 seeds `0..9`, reports both mean and standard deviation / min-max across seeds).
  4. **Worst-Case & Guess Distribution**: Maximum turns among wins (`max_win_turns`), total losses (`loss_count`), and histogram over `{1, 2, 3, 4, 5, 6, 7 (loss)}`.
  5. **Candidate-Set Trajectories ($t = 1..6$)**:
     - `carried_forward_mean`: Solved games carry forward `1` for subsequent turns; averaged over all games in the split.
     - `unsolved_only_mean` & `unsolved_count`: Averaged strictly over games still unsolved *after* turn $t$ (`null` when `unsolved_count == 0`).
  6. **Runtime Metrics**: Reports online selection time per guess (`mean_select_ms`) separately from cache load/generation time, noting whether Turn-1 opening entropies were served from cache.

