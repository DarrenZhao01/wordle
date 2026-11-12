# Wordle Agent & Terminal Prototype: Technical Write-Up

## Abstract
We present the design, verification, and empirical evaluation of a Wordle game simulator, three progressive agent architectures, and an interactive assistant covering Weeks 1–4 of the project roadmap plus the interactive CLI extension: (1) a **Random-Consistent Baseline**, (2) a **Positional Letter-Frequency Baseline**, and (3) a **One-Step Information-Gain (Shannon Entropy) Solver**. All agents operate under strict separation from the hidden answer, starting every game with the full dictionary of $2,315$ possible solutions and updating their candidate state via precomputed ternary feedback patterns. Across all $2,315$ Wordle answers, the Random-Consistent baseline averages $4.0854 \pm 0.0173$ guesses ($98.15\%$ win rate), the Positional Letter-Frequency baseline averages $3.6397$ guesses ($99.40\%$ win rate, $14$ losses), a candidate-restricted Entropy variant averages $3.5983$ guesses ($99.52\%$ win rate, $11$ losses), and the full Information-Gain Entropy solver ($\varepsilon = 0.01$) achieves a **$100.00\%$ win rate** ($0$ losses across all $2,315$ targets) with an average of **$3.4713$ guesses** ($3.4644$ on the 20% development split and $3.4730$ on the held-out 80% test split) at $4.031\text{ ms}$ per guess. Finally, we evaluate our interactive assistant on live New York Times Wordle gameplay (`KARAT`), demonstrating a two-tier candidate fallback ($2,315 \to 12,972$) that seamlessly handles NYT-curated answers outside Josh Wardle's original $2,315$-word list.

---

## 1. System Architecture & Verification

### 1.1 Dictionaries & Target Splits (`src/dictionary.py`, `data/splits.json`)
We use the canonical Wordle dictionaries published by Cyrus Freshman:
- **Possible Answers ($A$)**: $|A| = 2,315$ alphabetically sorted 5-letter target words (`data/wordle-answers-alphabetical.txt`, SHA-256: `5209b35f823f8b80...`).
- **Allowed Guesses ($G$)**: $|G| = 12,972$ alphabetically sorted 5-letter valid words formed by the union of $A$ and $10,657$ additional non-answer guesses (`data/wordle-allowed-guesses.txt`, SHA-256: `99be2e38dadf3e26...`).
- **Deterministic Evaluation Splits**: Using a fixed split seed (`split_seed = 42`), $A$ is partitioned into a **20% Development Split ($463$ targets)** for hyperparameter tuning and an **80% Held-Out Test Split ($1,852$ targets)** for final reporting, saved with SHA-256 integrity hashes in `data/splits.json`.
- **No-Leakage Invariant**: Regardless of whether a hidden target is drawn from the Development, Test, or Full set, every game initializes the agent's candidate answer set $C_0$ to all $2,315$ answers in $A$. The split determines only which targets are simulated or aggregated.

### 1.2 Feedback Engine & Game Simulator (`src/engine.py`)
- **Base-3 Pattern Encoding (`uint8`)**: Each position $i \in \{0, 1, 2, 3, 4\}$ receives a ternary trit $t_i \in \{0\text{ (Gray)}, 1\text{ (Yellow)}, 2\text{ (Green)}\}$, packed into a single unsigned byte:
  $$\text{pattern\_id} = \sum_{i=0}^{4} t_i \cdot 3^i \in [0, 242], \quad \text{ALL\_GREEN} = 242.$$
- **Two-Pass Duplicate-Letter Rule (`compute_feedback_scalar`)**:
  1. **Pass 1 (Greens)**: Marks exact positional matches ($t_i = 2$) and increments an unmatched-letter counter for all non-matching positions in the target answer.
  2. **Pass 2 (Yellows & Grays)**: Iterates left-to-right ($i = 0 \dots 4$) over non-green guess positions; if the guessed letter has a positive available count in the target's unmatched pool, marks $t_i = 1$ and decrements the pool count; otherwise marks $t_i = 0$.
- **Simulator (`WordleSimulator`)**: Encapsulates the hidden target word, validates that guesses belong to $G$, tracks remaining turns ($T_{\max} = 6$), prevents calls to `step()` after game termination, and returns only `(pattern_id, turn_number, turns_remaining, solved, game_over)` to the agent.

### 1.3 Validated Pattern & Opening-Entropy Cache (`src/precompute.py`, `data/cache_meta.json`)
To avoid redundant string comparisons during gameplay and evaluation:
- **Batched Pattern Matrix (`data/pattern_matrix.npy`)**: A `(12972, 2315)` `uint8` matrix ($\approx 30.0\text{ MB}$) generated in 512-guess batches and verified against the scalar reference implementation. In memory, both row-major `(12972, 2315)` and transposed `(2315, 12972)` contiguous views are maintained for fast candidate slicing.
- **Opening-Entropy Vector (`data/opening_entropies.npy`)**: A `(12972,)` `float64` vector storing the exact Turn-1 Shannon entropy $H(g, A)$ for every $g \in G$ against all $2,315$ initial answers. Caching the full Turn-1 vector rather than a single word allows any near-tie preference parameter $\varepsilon$ or candidate restriction mode to select its exact policy-consistent opener without recomputing Turn-1 partitions.
- **Empirical Turn-1 Entropy Findings**:
  - Highest-entropy overall guess: `soare` ($H = 5.885960\text{ bits}$, non-answer)
  - Second highest overall: `roate` ($H = 5.882779\text{ bits}$, non-answer)
  - Highest-entropy answer-list word: `raise` ($H = 5.877910\text{ bits}$, in $A$)
  - Exact opener switching threshold:
    $$\Delta H = H(\text{soare}, A) - H(\text{raise}, A) = 0.008050419557\text{ bits}.$$
    Consequently, the one-step entropy policy selects `soare` on Turn 1 for $0 \le \varepsilon < \Delta H$ and switches to `raise` for $\varepsilon \ge \Delta H$.
- **Metadata Validation (`data/cache_meta.json`)**: Stores file/list SHA-256 digests, word counts, boundary words, array shapes/dtypes, and algorithm version tags. Cache loading automatically rejects mismatched metadata and regenerates cleanly.

### 1.4 Correctness & Property Testing (`tests/test_wordle.py`)
All 11 test suites in `tests/test_wordle.py` pass in $0.63\text{ s}$, verifying:
1. Independently specified duplicate-letter edge cases (`ALLEY` vs. `APPLE` $\to$ `(G, Y, ., Y, .)`, `SPEED` vs. `STEAL`, `SPEED` vs. `ABIDE`, `SPEED` vs. `ERASE`, `EERIE` vs. `ELITE`).
2. Bijective base-3 encoding/decoding across all $243$ patterns and the diagonal identity $M[w, w] = 242$.
3. Agreement between the batched matrix, opening entropy cache, and scalar reference functions.
4. Rejection of tampered cache metadata.
5. Simulator dictionary enforcement and post-game termination errors.
6. Monotonic candidate reduction ($1 \le |C_{t}| \le |C_{t-1}|$), survival of the true hidden target at every turn, immediate selection when $|C| = 1$, diagnostic `RuntimeError` when an impossible observation leaves $|C| = 0$, and clean state reset between games.
7. Exact entropy and tie-breaking on small candidate sets (`FIGHT/LIGHT/MIGHT/SIGHT` where exploratory `FILMY` yields $2.0\text{ bits}$ vs. $0.8113\text{ bits}$ for `SIGHT`), final-turn restriction when `turns_remaining == 1`, seed invariance across evaluation subsets, and interactive assistant recovery (including NYT expanded-dictionary fallback).

---

## 2. Agent Formulations (`src/agents/`)

All agents share a stateful lifecycle (`reset(seed)` $\to$ `select_guess(turns_remaining)` $\to$ `simulator.step(guess)` $\to$ `observe(guess, pattern_id)`) and maintain the active candidate answer index set `c_idx` ($C \subseteq A$). Whenever $|C| = 1$, every agent immediately guesses that sole remaining answer.

1. **Random-Consistent Baseline (`RandomConsistentAgent`)**:
   - Chooses uniformly at random from the remaining possible answers $C$ using an isolated `random.Random(game_seed)` instance, where `game_seed` is deterministically derived from `SHA256(f"{run_seed}:{target}")`.
2. **Positional Letter-Frequency Baseline (`LetterFrequencyAgent`)**:
   - Scores and selects among the remaining possible answers $C$ using column-wise letter frequencies:
     $$\text{Score}_{\text{freq}}(g, C) = \sum_{i=0}^{4} P(c[i] = g[i] \mid c \in C) = \frac{1}{|C|} \sum_{i=0}^{4} \sum_{c \in C} \mathbb{I}(c[i] = g[i]).$$
   - We also evaluate an optional duplicate-discounted variant (`letter_frequency_unique`) that counts only the highest positional frequency per distinct letter in $g$. Ties break alphabetically.
3. **Information-Gain (Shannon Entropy) Solver (`EntropyAgent`)**:
   - For each candidate guess $g$, partitions the remaining answers $C$ by feedback outcome $p \in [0, 242]$ with partition counts $n_p(g, C)$ and computes expected information gain (in bits):
     $$H(g, C) = -\sum_{p \,:\, n_p(g, C) > 0} \frac{n_p(g, C)}{|C|} \log_2 \frac{n_p(g, C)}{|C|}.$$
   - **Near-Tie Answer Preference ($\varepsilon$)**: Combines information gain with an answer-membership bonus:
     $$\text{Score}_{\text{ent}}(g, C) = H(g, C) + \varepsilon \cdot \mathbb{I}(g \in C),$$
     breaking ties within $10^{-12}$ in favor of $g \in C$, then alphabetically.
   - **Final-Turn Rule (`turns_remaining == 1`)**: On Turn 6, an exploratory word outside $C$ has zero probability of winning. When `turns_remaining == 1`, the agent explicitly masks its search pool to $C$, selecting the highest-entropy candidate within $C$ (breaking ties alphabetically).
   - **Candidate-Only Ablation (`entropy_candidates_only`)**: Setting `allow_non_answers = False` restricts guesses to $C$ on all turns, isolating the impact of the entropy objective from the ability to play non-answer exploratory words.

---

## 3. Experimental Evaluation & Results

### 3.1 Development-Split Hyperparameter Tuning ($\varepsilon$)
We tuned the near-tie preference parameter $\varepsilon$ exclusively on the **20% Development Split ($463$ targets)**, where each game starts with all $2,315$ answers as initial candidates (`results/tuning_dev.json`):

| $\varepsilon$ | Turn-1 Opener | Avg Guesses (All / Pen-7) | Avg Guesses (Won Only) | Win Rate (%) | Losses | Mean Select Time (ms/guess) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0.0000` | `soare` | `3.4924` | `3.4924` | `100.00%` | `0` | `3.970` |
| **`0.0100`** | **`raise`** | **`3.4644`** | **`3.4644`** | **`100.00%`** | **`0`** | **`4.212`** |
| `0.0500` | `raise` | `3.4644` | `3.4644` | `100.00%` | `0` | `4.323` |
| `0.1000` | `raise` | `3.4644` | `3.4644` | `100.00%` | `0` | `3.984` |
| `0.2000` | `raise` | `3.4773` | `3.4773` | `100.00%` | `0` | `3.756` |

**Finding**: At $\varepsilon = 0.0$, the solver opens with the non-answer `soare` and strictly prefers marginally higher entropy over immediate win opportunities, averaging $3.4924$ guesses. Crossing the $\Delta H \approx 0.0081$ threshold ($\varepsilon \in [0.01, 0.10]$) switches the opener to `raise` and allows candidate answers with near-maximal entropy to be guessed directly, reducing the average guess count by $0.028$ guesses to **$3.4644$** with zero losses. Increasing $\varepsilon$ further ($\ge 0.20$) begins sacrificing too much information gain for premature answer attempts, degrading average guesses to $3.4773$. We select $\varepsilon^* = 0.01$.

---

### 3.2 Primary Benchmark Comparison Across Splits
All agents were evaluated across all $2,315$ answers (with `random_consistent` run over $10$ seeds `0..9`, totaling $23,150$ random-baseline games), and metrics were sliced into **Dev ($463$)**, **Test ($1,852$)**, and **All ($2,315$)** from the canonical game records (`results/benchmark_summary.json`). Unsolved games stop after 6 turns and receive a penalty of $7$ guesses.

#### Table 1: Held-Out Test Split ($1,852$ Targets)
| Agent | Guess Pool | Opener | Avg Guesses (All / Pen-7) | Avg Guesses (Won Only) | Win Rate (%) | Max Win / Losses | Mean Select (ms/guess) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10 seeds) | $C$ | *(random)* | `4.0916 ± 0.0234` | `4.0354 ± 0.0151` | `98.10% ± 0.39%` | `6 / 35.1` | `0.002` |
| `letter_frequency` (direct sum) | $C$ | `slate` | `3.6425` | `3.6188` | `99.30%` | `6 / 13` | `0.052` |
| `letter_frequency_unique` | $C$ | `slate` | `3.6409` | `3.6172` | `99.30%` | `6 / 13` | `1.441` |
| `entropy_candidates_only` ($\varepsilon=0.01$) | $C$ | `raise` | `3.5929` | `3.5744` | `99.46%` | `6 / 10` | `0.121` |
| **`entropy`** ($\varepsilon=0.01$) | **All $12,972$** | **`raise`** | **`3.4730`** | **`3.4730`** | **`100.00%`** | **`6 / 0`** | **`4.024`** |

#### Table 2: Full Dictionary ($2,315$ Targets)
| Agent | Guess Pool | Opener | Avg Guesses (All / Pen-7) | Avg Guesses (Won Only) | Win Rate (%) | Max Win / Losses | Mean Select (ms/guess) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10 seeds) | $C$ | *(random)* | `4.0854 ± 0.0173` | `4.0304 ± 0.0117` | `98.15% ± 0.31%` | `6 / 42.9` | `0.002` |
| `letter_frequency` (direct sum) | $C$ | `slate` | `3.6397` | `3.6193` | `99.40%` | `6 / 14` | `0.052` |
| `letter_frequency_unique` | $C$ | `slate` | `3.6423` | `3.6204` | `99.35%` | `6 / 15` | `1.439` |
| `entropy_candidates_only` ($\varepsilon=0.01$) | $C$ | `raise` | `3.5983` | `3.5820` | `99.52%` | `6 / 11` | `0.121` |
| **`entropy`** ($\varepsilon=0.01$) | **All $12,972$** | **`raise`** | **`3.4713`** | **`3.4713`** | **`100.00%`** | **`6 / 0`** | **`4.031`** |

---

### 3.3 Full Guess-Count Distributions (All $2,315$ Targets)
| Agent | 1 Guess | 2 Guesses | 3 Guesses | 4 Guesses | 5 Guesses | 6 Guesses | Loss (7) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10-seed mean) | `1.5 (0.06%)` | `86.9 (3.75%)` | `575.9 (24.88%)` | `933.0 (40.30%)` | `526.4 (22.74%)` | `148.4 (6.41%)` | `42.9 (1.85%)` |
| `letter_frequency` | `1 (0.04%)` | `146 (6.31%)` | `900 (38.88%)` | `975 (42.12%)` | `239 (10.32%)` | `40 (1.73%)` | `14 (0.60%)` |
| `letter_frequency_unique` | `1 (0.04%)` | `146 (6.31%)` | `906 (39.14%)` | `966 (41.73%)` | `234 (10.11%)` | `47 (2.03%)` | `15 (0.65%)` |
| `entropy_candidates_only` | `1 (0.04%)` | `131 (5.66%)` | `999 (43.15%)` | `919 (39.70%)` | `207 (8.94%)` | `47 (2.03%)` | `11 (0.48%)` |
| **`entropy`** ($\varepsilon=0.01$) | **`1 (0.04%)`** | **`57 (2.46%)`** | **`1174 (50.71%)`** | **`1018 (43.97%)`** | **`63 (2.72%)`** | **`2 (0.09%)`** | **`0 (0.00%)`** |

**Key Observations**:
1. **Direct vs. Duplicate-Discounted Frequency**: When choosing among remaining candidate answers $C$, direct positional frequency (`3.6397` avg, `14` losses) slightly outperforms the duplicate-discounted variant (`3.6423` avg, `15` losses), confirming that once $C$ narrows to candidates with repeated letters, zeroing duplicate positional contributions slightly handicaps valid multi-letter answers.
2. **Scoring Function vs. Exploratory Guess Pool**: Over the exact same candidate-only pool $C$, replacing positional letter frequency with Shannon entropy (`entropy_candidates_only`) reduces average guesses from `3.6397` to `3.5983` and cuts losses from `14` to `11`. Expanding the search space to all $12,972$ allowed guesses (`entropy`) accounts for the largest gain—cutting average guesses to **`3.4713`**, solving **$53.2\%$ of all games in $\le 3$ turns** and **$97.2\%$ in $\le 4$ turns**, and eliminating all $11$ rhyming-family losses (such as `_ATCH`, `_IGHT`, `_OUND`, and `_ASTE`).

---

### 3.4 Candidate-Set Size Reduction Trajectories (All $2,315$ Targets)
We report both the **Carried-Forward Mean** (where solved games carry forward $|C_t| = 1$ on subsequent turns, averaged over all $2,315$ games) and the **Unsolved-After-Turn Mean** (averaged strictly over the `[count]` of games still unsolved after turn $t$, with `null` when $0$ unsolved games remain):

| Agent | Metric | After Turn 1 | After Turn 2 | After Turn 3 | After Turn 4 | After Turn 5 | After Turn 6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` | Carried-Forward | `216.55` | `18.44` | `2.85` | `1.30` | `1.06` | `1.01` |
| *(23,150 games)* | Unsolved `[count]` | `216.7 [23135]` | `19.1 [22266]` | `3.6 [16507]` | `2.0 [7177]` | `1.7 [1913]` | `1.6 [429]` |
| `letter_frequency` | Carried-Forward | `71.57` | `5.12` | `1.44` | `1.06` | `1.02` | `1.00` |
| *(2,315 games)* | Unsolved `[count]` | `71.6 [2314]` | `5.4 [2168]` | `1.8 [1268]` | `1.5 [293]` | `1.7 [54]` | `1.4 [14]` |
| `entropy_candidates_only` | Carried-Forward | `61.00` | `4.36` | `1.35` | `1.06` | `1.01` | `1.00` |
| *(2,315 games)* | Unsolved `[count]` | `61.0 [2314]` | `4.6 [2183]` | `1.7 [1184]` | `1.6 [265]` | `1.4 [58]` | `1.4 [11]` |
| **`entropy`** | Carried-Forward | **`61.00`** | **`3.50`** | **`1.07`** | **`1.00`** | **`1.00`** | **`1.00`** |
| *(2,315 games)* | Unsolved `[count]` | **`61.0 [2314]`** | **`3.6 [2257]`** | **`1.2 [1083]`** | **`1.1 [65]`** | **`1.0 [2]`** | **`null [0]`** |

---

## 4. Interactive Assistant & Live NYT Wordle Case Study (`src/assistant.py`)

### 4.1 Dual-Workflow Interactive Assistant
The interactive terminal assistant (`python main.py assistant`) supports two workflows:
1. **External Wordle Helper Mode (default)**: The user enters their guess (or presses `Enter` to accept the `#1` recommended guess) and the 5-position feedback received from an external Wordle game using flexible notation (`G`/`2` = Green, `Y`/`1` = Yellow, `.`/`B`/`X`/`0`/`-`/`_` = Gray).
2. **Practice / Comparison Mode (`--target WORD` or `--random-target`)**: The simulator holds a hidden target word, automatically scores the user's guesses, and prints a compact comparison whenever the user's guess differs from the `#1` recommendation:
   ```text
   Comparison: Your guess CRANE (H = 5.743 bits, possible answer, 142 groups) vs #1 RAISE (H = 5.878 bits, possible answer, 132 groups)
   ```

### 4.2 Real-World Discovery: Two-Tier Candidate Fallback ($2,315 \to 12,972$) on `KARAT`
During live testing on a New York Times Wordle puzzle with hidden answer `KARAT` and human guess sequence `CRANE` (`01100` / `.YY..`) $\to$ `TOILS` (`10000` / `Y....`) $\to$ `DUMPY` (`00000` / `.....`):
- Turns 1 and 2 reduced the original $2,315$ candidate set $C$ to $3$ words: `PARTY, RATTY, WARTY` (all ending in `Y`).
- Turn 3 (`DUMPY` $\to$ `.....`) eliminated `Y`, leaving **$0$ candidates in the original $2,315$ answer list**, because `KARAT` resides in the $10,657$ non-answer list (`wordle-allowed-guesses.txt`) that the New York Times editorial team now occasionally draws from.
- To preserve the curated $2,315$ prior on standard games while supporting modern NYT puzzles, `src/assistant.py` maintains a parallel consistency filter over the full $12,972$ dictionary. When `DUMPY` (`00000`) eliminated the last $3$ words in the $2,315$ list, the assistant automatically transitioned to the $6$ consistent words in the $12,972$ dictionary (`ATTAR, GARTH, JARTA, KARAT, RABAT, RATHA`) and identified that $5$ of those candidates (including `KARAT` and `GARTH`) achieved the maximum possible entropy of $\log_2(6) = 2.585\text{ bits}$, solving the puzzle on Turn 4.

---

## 5. Reproducing the Experiments via CLI (`main.py`)

```powershell
# 1. Run the 11 unit and property test suites
.\.venv\Scripts\pytest.exe -v

# 2. Launch the Interactive Wordle Assistant (External NYT Helper or Practice Mode)
.\.venv\Scripts\python.exe main.py assistant
.\.venv\Scripts\python.exe main.py assistant --target watch

# 3. Simulate individual games in the terminal with ANSI board rendering and |C| tracking
.\.venv\Scripts\python.exe main.py play --agent entropy --target watch,apple --epsilon 0.01

# 4. Tune epsilon on the 20% Development Split (463 words)
.\.venv\Scripts\python.exe main.py tune

# 5. Run the full benchmark suite across all 2,315 words and export JSON/JSONL logs
.\.venv\Scripts\python.exe main.py evaluate
```
