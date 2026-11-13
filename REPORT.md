# Wordle Agent & Terminal Prototype: Technical Write-Up

## Abstract
We present the design, verification, and empirical evaluation of a Wordle game simulator, three progressive agent architectures, and an interactive assistant covering Weeks 1–4 of the project roadmap plus the interactive CLI extension: (1) a **Random-Consistent Baseline**, (2) a **Positional Letter-Frequency Baseline**, and (3) a **One-Step Information-Gain (Shannon Entropy) Solver**. We evaluate all agents across two canonical Wordle dictionaries:
1. **Curated Modern NYT WordleBot Dictionary (Default)**: $3,158$ possible answers and $14,855$ allowed guesses (`data/nyt-answers-alphabetical.txt` and `data/nyt-allowed-guesses.txt`).
2. **Original 2021 Wordle Dictionary (`--original`)**: $2,315$ possible answers and $12,972$ allowed guesses (`data/wordle-answers-alphabetical.txt` and `data/wordle-allowed-guesses.txt`).

On the **Modern NYT ($3,158$-word)** benchmark, the Random-Consistent baseline averages $4.2809 \pm 0.0171$ guesses ($95.80\%$ win rate), the Positional Letter-Frequency baseline averages $4.0358$ guesses ($97.56\%$ win rate, $77$ losses), a candidate-restricted Entropy variant averages $3.7885$ guesses ($98.29\%$ win rate, $54$ losses), and the full Information-Gain Entropy solver ($\varepsilon = 0.01$, opening with `tarse`) achieves a **$100.00\%$ win rate** ($0$ losses across all $3,158$ targets) with an average of **$3.5678$ guesses** ($3.5839$ on the 20% Dev split and $3.5637$ on the 80% Test split) at $5.569\text{ ms}$ per guess. On the **Original ($2,315$-word)** benchmark (`--original`), the full Entropy solver ($\varepsilon = 0.01$, opening with `raise`) likewise achieves a **$100.00\%$ win rate** ($0$ losses) with an average of **$3.4713$ guesses** at $4.031\text{ ms}$ per guess.

---

## 1. System Architecture & Verification

### 1.1 Dictionaries & Target Splits (`src/dictionary.py`)
The system supports two validated dictionary editions, with **Modern NYT** as the default and `--original` to select the 2021 Wordle lists:
- **Modern NYT Dictionary (`edition = "modern"`, default)**:
  - **Possible Answers ($A_{\text{modern}}$)**: $|A_{\text{modern}}| = 3,158$ sorted 5-letter target words (`data/nyt-answers-alphabetical.txt`, SHA-256: `850555a3879710cc...`), which includes NYT-curated answers such as `karat`, `guano`, `kazoo`, `laser`, `panko`, and `snafu`.
  - **Allowed Guesses ($G_{\text{modern}}$)**: $|G_{\text{modern}}| = 14,855$ sorted 5-letter valid words ($3,158$ answers $\cup$ $11,697$ non-answers from `data/nyt-allowed-guesses.txt`, SHA-256: `ecf2a06d2f35878b...`).
  - **Splits (`data/splits_modern.json`, `split_seed = 42`)**: **632 Dev** targets (20%) and **2,526 Test** targets (80%).
- **Original 2021 Dictionary (`edition = "original"`, `--original`)**:
  - **Possible Answers ($A_{\text{orig}}$)**: $|A_{\text{orig}}| = 2,315$ sorted 5-letter target words (`data/wordle-answers-alphabetical.txt`, SHA-256: `5209b35f823f8b80...`).
  - **Allowed Guesses ($G_{\text{orig}}$)**: $|G_{\text{orig}}| = 12,972$ sorted 5-letter valid words ($2,315$ answers $\cup$ $10,657$ non-answers from `data/wordle-allowed-guesses.txt`, SHA-256: `99be2e38dadf3e26...`).
  - **Splits (`data/splits.json`, `split_seed = 42`)**: **463 Dev** targets (20%) and **1,852 Test** targets (80%).
- **No-Leakage Invariant**: Every game initializes the agent's candidate answer set $C_0$ to all $|A|$ answers of the active dictionary edition regardless of whether the target belongs to the Dev or Test split.

### 1.2 Feedback Engine & Game Simulator (`src/engine.py`)
- **Base-3 Pattern Encoding (`uint8`)**: Each position $i \in \{0, 1, 2, 3, 4\}$ receives a ternary trit $t_i \in \{0\text{ (Gray)}, 1\text{ (Yellow)}, 2\text{ (Green)}\}$, packed into a single unsigned byte:
  $$\text{pattern\_id} = \sum_{i=0}^{4} t_i \cdot 3^i \in [0, 242], \quad \text{ALL\_GREEN} = 242.$$
- **Two-Pass Duplicate-Letter Rule (`compute_feedback_scalar`)**:
  1. **Pass 1 (Greens)**: Marks exact positional matches ($t_i = 2$) and increments an unmatched-letter counter for all non-matching positions in the target answer.
  2. **Pass 2 (Yellows & Grays)**: Iterates left-to-right ($i = 0 \dots 4$) over non-green guess positions; if the guessed letter has a positive available count in the target's unmatched pool, marks $t_i = 1$ and decrements the pool count; otherwise marks $t_i = 0$.
- **Simulator (`WordleSimulator`)**: Encapsulates the hidden target word, validates that guesses belong to $G$, tracks remaining turns ($T_{\max} = 6$), prevents calls to `step()` after game termination, and returns only `(pattern_id, turn_number, turns_remaining, solved, game_over)` to the agent.

### 1.3 Validated Pattern & Opening-Entropy Cache (`src/precompute.py`)
Both dictionary editions maintain separate hash-validated caches on disk (`pattern_matrix_modern.npy` / `opening_entropies_modern.npy` / `cache_meta_modern.json` and `pattern_matrix.npy` / `opening_entropies.npy` / `cache_meta.json`):
- **Empirical Turn-1 Entropy Comparison Across Editions**:
  - **Modern NYT ($3,158$ answers / $14,855$ guesses)**:
    - Top overall opener: `tarse` ($H = 5.887629\text{ bits}$, non-answer)
    - Top answer-list opener: `raise` ($H = 5.832455\text{ bits}$, in $A_{\text{modern}}$)
    - Exact switching threshold: $\Delta H_{\text{modern}} = 0.055174\text{ bits}$
  - **Original 2021 ($2,315$ answers / $12,972$ guesses)**:
    - Top overall opener: `soare` ($H = 5.885960\text{ bits}$, non-answer)
    - Top answer-list opener: `raise` ($H = 5.877910\text{ bits}$, in $A_{\text{orig}}$)
    - Exact switching threshold: $\Delta H_{\text{orig}} = 0.008050\text{ bits}$

### 1.4 Correctness & Property Testing (`tests/test_wordle.py`)
All 12 test suites in `tests/test_wordle.py` pass in $0.63\text{ s}$ (once caches are built), verifying duplicate-letter feedback rules, base-3 encoding, matrix vs. scalar reference agreement, metadata hash validation, simulator invariants, candidate tracking monotonicity, final-turn rules, seed invariance, and native `KARAT` resolution in the Modern NYT dictionary.

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
   - Computes expected information gain (in bits) across feedback outcomes $p \in [0, 242]$:
     $$H(g, C) = -\sum_{p \,:\, n_p(g, C) > 0} \frac{n_p(g, C)}{|C|} \log_2 \frac{n_p(g, C)}{|C|}, \quad \text{Score}_{\text{ent}}(g, C) = H(g, C) + \varepsilon \cdot \mathbb{I}(g \in C).$$
   - **Final-Turn Rule (`turns_remaining == 1`)**: Explicitly masks the search pool to $C$, selecting the highest-entropy candidate within $C$ (breaking ties alphabetically).
   - **Candidate-Only Ablation (`entropy_candidates_only`)**: Restricts guesses to $C$ on all turns (`allow_non_answers = False`).

---

## 3. Experimental Evaluation: Modern NYT Dictionary (Default, 3,158 Answers)

### 3.1 Development-Split Hyperparameter Tuning ($\varepsilon$ on 632 Modern NYT Dev Targets)
Full tuning logs are saved in `results/tuning_dev_modern.json`:

| $\varepsilon$ | Turn-1 Opener | Avg Guesses (All / Pen-7) | Avg Guesses (Won Only) | Win Rate (%) | Losses | Mean Select Time (ms/guess) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0.0000` | `tarse` | `3.5886` | `3.5886` | `100.00%` | `0` | `6.312` |
| **`0.0100`** | **`tarse`** | **`3.5839`** | **`3.5839`** | **`100.00%`** | **`0`** | **`6.085`** |
| `0.0500` | `tarse` | `3.5839` | `3.5839` | `100.00%` | `0` | `5.880` |
| `0.0600` | `raise` | `3.6313` | `3.6313` | `100.00%` | `0` | `5.347` |
| `0.1000` | `raise` | `3.6329` | `3.6329` | `100.00%` | `0` | `5.346` |
| `0.2000` | `raise` | `3.6392` | `3.6392` | `100.00%` | `0` | `5.245` |

**Finding**: On the 3,158-word Modern NYT dictionary, `tarse` ($5.8876\text{ bits}$) outperforms `raise` ($5.8325\text{ bits}$) by $0.0552\text{ bits}$ on Turn 1. Setting $\varepsilon^* = 0.01$ retains `tarse` as the opener while breaking near-ties in favor of candidate answers on subsequent turns, achieving $3.5839$ average guesses and a $100.00\%$ win rate on the Dev split.

### 3.2 Modern NYT Benchmark Results (`results/benchmark_summary_modern.json`)

#### Table 1A: Modern NYT Held-Out Test Split ($2,526$ Targets)
| Agent | Guess Pool | Opener | Avg Guesses (All / Pen-7) | Avg Guesses (Won Only) | Win Rate (%) | Max Win / Losses | Mean Select (ms/guess) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10 seeds) | $C$ | *(random)* | `4.2816 ± 0.0191` | `4.1666 ± 0.0130` | `95.94% ± 0.37%` | `6 / 102.5` | `0.002` |
| `letter_frequency` (direct sum) | $C$ | `samey` | `4.0289` | `3.9591` | `97.70%` | `6 / 58` | `0.060` |
| `letter_frequency_unique` | $C$ | `samey` | `3.9976` | `3.9020` | `96.91%` | `6 / 78` | `1.849` |
| `entropy_candidates_only` ($\varepsilon=0.01$) | $C$ | `raise` | `3.7835` | `3.7291` | `98.34%` | `6 / 42` | `0.156` |
| **`entropy`** ($\varepsilon=0.01$) | **All $14,855$** | **`tarse`** | **`3.5637`** | **`3.5637`** | **`100.00%`** | **`6 / 0`** | **`5.557`** |

#### Table 1B: Modern NYT All Targets ($3,158$ Words)
| Agent | Guess Pool | Opener | Avg Guesses (All / Pen-7) | Avg Guesses (Won Only) | Win Rate (%) | Max Win / Losses | Mean Select (ms/guess) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10 seeds) | $C$ | *(random)* | `4.2809 ± 0.0171` | `4.1618 ± 0.0133` | `95.80% ± 0.33%` | `6 / 132.5` | `0.002` |
| `letter_frequency` (direct sum) | $C$ | `samey` | `4.0358` | `3.9617` | `97.56%` | `6 / 77` | `0.060` |
| `letter_frequency_unique` | $C$ | `samey` | `3.9911` | `3.9029` | `97.15%` | `6 / 90` | `1.848` |
| `entropy_candidates_only` ($\varepsilon=0.01$) | $C$ | `raise` | `3.7885` | `3.7326` | `98.29%` | `6 / 54` | `0.156` |
| **`entropy`** ($\varepsilon=0.01$) | **All $14,855$** | **`tarse`** | **`3.5678`** | **`3.5678`** | **`100.00%`** | **`6 / 0`** | **`5.569`** |

#### Table 1C: Modern NYT Full Guess-Count Distributions ($3,158$ Targets)
| Agent | 1 Guess | 2 Guesses | 3 Guesses | 4 Guesses | 5 Guesses | 6 Guesses | Loss (7) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10-seed mean) | `1.0 (0.03%)` | `101.3 (3.21%)` | `658.8 (20.86%)` | `1186.0 (37.56%)` | `802.8 (25.42%)` | `275.6 (8.73%)` | `132.5 (4.20%)` |
| `letter_frequency` | `1 (0.03%)` | `112 (3.55%)` | `841 (26.63%)` | `1350 (42.75%)` | `604 (19.13%)` | `173 (5.48%)` | `77 (2.44%)` |
| `letter_frequency_unique` | `1 (0.03%)` | `112 (3.55%)` | `906 (28.69%)` | `1372 (43.45%)` | `519 (16.43%)` | `158 (5.00%)` | `90 (2.85%)` |
| `entropy_candidates_only` | `1 (0.03%)` | `136 (4.31%)` | `1177 (37.27%)` | `1284 (40.66%)` | `390 (12.35%)` | `116 (3.67%)` | `54 (1.71%)` |
| **`entropy`** ($\varepsilon=0.01$) | **`0 (0.00%)`** | **`69 (2.18%)`** | **`1413 (44.74%)`** | **`1497 (47.40%)`** | **`172 (5.45%)`** | **`7 (0.22%)`** | **`0 (0.00%)`** |

#### Table 1D: Modern NYT Candidate-Set Trajectories ($3,158$ Targets)
| Agent | Metric | After Turn 1 | After Turn 2 | After Turn 3 | After Turn 4 | After Turn 5 | After Turn 6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` | Carried-Forward | `303.85` | `27.52` | `4.24` | `1.67` | `1.18` | `1.05` |
| *(31,580 games)* | Unsolved `[count]` | `303.9 [31570]` | `28.4 [30557]` | `5.3 [23969]` | `2.7 [12109]` | `2.4 [4081]` | `2.2 [1325]` |
| `letter_frequency` | Carried-Forward | `159.99` | `13.26` | `2.55` | `1.33` | `1.07` | `1.01` |
| *(3,158 games)* | Unsolved `[count]` | `160.0 [3157]` | `13.7 [3045]` | `3.2 [2204]` | `2.2 [854]` | `1.9 [250]` | `1.5 [77]` |
| `entropy_candidates_only` | Carried-Forward | `90.15` | `6.57` | `1.84` | `1.20` | `1.05` | `1.01` |
| *(3,158 games)* | Unsolved `[count]` | `90.2 [3157]` | `6.8 [3021]` | `2.4 [1844]` | `2.1 [560]` | `1.9 [170]` | `1.7 [54]` |
| **`entropy`** | Carried-Forward | **`100.07`** | **`4.68`** | **`1.18`** | **`1.01`** | **`1.00`** | **`1.00`** |
| *(3,158 games)* | Unsolved `[count]` | **`100.1 [3158]`** | **`4.8 [3089]`** | **`1.3 [1676]`** | **`1.1 [179]`** | **`1.0 [7]`** | **`null [0]`** |

---

## 4. Experimental Evaluation: Original 2021 Dictionary (`--original`, 2,315 Answers)

Full summaries for the Original 2021 dictionary are saved in `results/benchmark_summary.json` and `results/tuning_dev.json`:

| Agent | Guess Pool | Opener | Dev (463) Avg | Test (1,852) Avg | All (2,315) Avg | All Win Rate (%) | Max Win / Losses | ms/guess |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_consistent` (10 seeds) | $C$ | *(random)* | `4.0607 ± 0.0419` | `4.0916 ± 0.0234` | `4.0854 ± 0.0173` | `98.15% ± 0.31%` | `6 / 42.9` | `0.002` |
| `letter_frequency` (direct sum) | $C$ | `slate` | `3.6285` | `3.6425` | `3.6397` | `99.40%` | `6 / 14` | `0.052` |
| `letter_frequency_unique` | $C$ | `slate` | `3.6479` | `3.6409` | `3.6423` | `99.35%` | `6 / 15` | `1.439` |
| `entropy_candidates_only` ($\varepsilon=0.01$) | $C$ | `raise` | `3.6199` | `3.5929` | `3.5983` | `99.52%` | `6 / 11` | `0.121` |
| **`entropy`** ($\varepsilon=0.01$) | **All $12,972$** | **`raise`** | **`3.4644`** | **`3.4730`** | **`3.4713`** | **`100.00%`** | **`6 / 0`** | **`4.031`** |

---

## 5. Interactive Assistant & Live NYT Wordle Comparison (`src/assistant.py`)

When running `python main.py assistant` (which defaults to the $3,158$-word Modern NYT dictionary) on the live NYT Wordle game for `KARAT` (`CRANE` `01100` $\to$ `TOILS` `10000` $\to$ `DUMPY` `00000` $\to$ `KARAT` `22222`):
- **With Modern NYT Dictionary (Default)**:
  - Turn 1 (`CRANE` $\to$ `.YY..`) reduces $|C|$ from $3,158 \to 80$.
  - Turn 2 (`TOILS` $\to$ `Y....`) reduces $|C|$ from $80 \to 4$: `KARAT, PARTY, RATTY, WARTY`.
  - On Turn 3, the assistant recommends `PARTY` ($H = 2.000\text{ bits}$, which splits all 4 candidates—including `KARAT`—into 4 singletons). When `DUMPY` (`00000`) is played instead, $|C|$ reduces from $4 \to 1$, identifying **`KARAT` as the sole remaining answer (`100% win`)** without needing any fallback.
- **With Original 2021 Dictionary (`--original`)**:
  - If `--original` is used on an NYT answer outside the 2021 list, `src/assistant.py` still maintains a secondary fallback over the $12,972$ allowed dictionary so the session never crashes when the 2,315 list hits `0`.

---

## 6. Reproducing the Experiments via CLI (`main.py`)

```powershell
# 1. Run all 12 unit and property test suites (both Modern NYT and Original editions)
.\.venv\Scripts\pytest.exe -v

# 2. Launch the Interactive Wordle Assistant (defaults to Modern NYT; add --original for 2021 list)
.\.venv\Scripts\python.exe main.py assistant
.\.venv\Scripts\python.exe main.py assistant --target karat

# 3. Simulate individual games in the terminal (Modern NYT default, or --original)
.\.venv\Scripts\python.exe main.py play --agent entropy --target karat,watch
.\.venv\Scripts\python.exe main.py play --original --agent entropy --target watch,apple

# 4. Tune epsilon on the 20% Development Split
.\.venv\Scripts\python.exe main.py tune
.\.venv\Scripts\python.exe main.py tune --original

# 5. Run the full benchmark suite and export JSON/JSONL logs
.\.venv\Scripts\python.exe main.py evaluate
.\.venv\Scripts\python.exe main.py evaluate --original
```
