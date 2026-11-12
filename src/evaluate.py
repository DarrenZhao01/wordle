"""Reproducible evaluation protocol, single-game runner, split slicing, and aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence

from src.agents import BaseAgent, EntropyAgent, LetterFrequencyAgent, RandomConsistentAgent
from src.dictionary import WordleDictionary, derive_game_seed
from src.engine import MAX_TURNS, WordleSimulator, format_pattern_ascii
from src.precompute import FeedbackCache


LOSS_PENALTY_GUESSES: int = 7


@dataclass(frozen=True)
class GameRecord:
    """Complete diagnostic record for a single simulated Wordle game."""

    target: str
    split: str
    agent_name: str
    agent_config: Dict[str, object]
    run_seed: int
    game_seed: int
    guesses: List[str]
    pattern_ids: List[int]
    pattern_ascii: List[str]
    candidates_before: List[int]
    candidates_after: List[int]
    turn_select_times_ms: List[float]
    used_opening_cache_turn1: bool
    solved: bool
    turns_used: int
    penalized_guesses: int
    total_select_time_ms: float


def play_single_game(
    agent: BaseAgent,
    target: str,
    dictionary: WordleDictionary,
    allowed_guesses_set: Optional[set] = None,
    run_seed: int = 0,
) -> GameRecord:
    """Run one complete game on target, starting with all 2,315 candidate answers."""
    cleaned_target = target.strip().lower()
    split_name = "dev" if cleaned_target in dictionary.dev_set else "test"
    game_seed = derive_game_seed(run_seed=run_seed, target_word=cleaned_target)

    # Invariant: agent always resets to all 2,315 candidate answers regardless of split
    agent.reset(seed=game_seed)
    if agent.num_candidates != dictionary.num_answers:
        raise RuntimeError(
            f"Split leakage check failed: expected {dictionary.num_answers} initial "
            f"candidates, got {agent.num_candidates}"
        )

    sim = WordleSimulator(
        target=cleaned_target,
        allowed_guesses_set=allowed_guesses_set,
        max_turns=MAX_TURNS,
    )

    guesses: List[str] = []
    pattern_ids: List[int] = []
    pattern_ascii: List[str] = []
    candidates_before: List[int] = []
    candidates_after: List[int] = []
    turn_select_times_ms: List[float] = []
    used_opening_cache_turn1 = False

    while not sim.game_over:
        c_before = agent.num_candidates
        t0 = time.perf_counter()
        guess = agent.select_guess(turns_remaining=sim.turns_remaining)
        select_ms = (time.perf_counter() - t0) * 1000.0

        if sim.turns_used == 0 and isinstance(agent, EntropyAgent):
            used_opening_cache_turn1 = bool(agent.last_used_opening_cache)

        step_res = sim.step(guess)
        c_after = agent.observe(guess, step_res.pattern_id)

        guesses.append(step_res.guess)
        pattern_ids.append(step_res.pattern_id)
        pattern_ascii.append(format_pattern_ascii(step_res.pattern_id))
        candidates_before.append(c_before)
        candidates_after.append(c_after)
        turn_select_times_ms.append(select_ms)

    penalized = sim.turns_used if sim.solved else LOSS_PENALTY_GUESSES

    return GameRecord(
        target=cleaned_target,
        split=split_name,
        agent_name=agent.name,
        agent_config=agent.config_dict(),
        run_seed=int(run_seed),
        game_seed=int(game_seed),
        guesses=guesses,
        pattern_ids=pattern_ids,
        pattern_ascii=pattern_ascii,
        candidates_before=candidates_before,
        candidates_after=candidates_after,
        turn_select_times_ms=turn_select_times_ms,
        used_opening_cache_turn1=used_opening_cache_turn1,
        solved=sim.solved,
        turns_used=sim.turns_used,
        penalized_guesses=penalized,
        total_select_time_ms=sum(turn_select_times_ms),
    )


def summarize_game_records(
    records: Sequence[GameRecord],
    split_label: str,
    cache_load_or_build_ms: Optional[float] = None,
    cache_loaded_from_disk: Optional[bool] = None,
) -> Dict[str, object]:
    """Compute exact evaluation metrics over a collection of GameRecords."""
    n = len(records)
    if n == 0:
        raise ValueError(f"Cannot summarize empty GameRecord sequence for {split_label}")

    won_records = [r for r in records if r.solved]
    wins = len(won_records)
    losses = n - wins
    win_rate_pct = (100.0 * wins) / float(n)

    avg_all = sum(r.penalized_guesses for r in records) / float(n)
    avg_won = (
        (sum(r.turns_used for r in won_records) / float(wins))
        if wins > 0
        else None
    )
    max_win_turns = max((r.turns_used for r in won_records), default=None)

    dist_counts: Dict[str, int] = {str(t): 0 for t in range(1, MAX_TURNS + 1)}
    dist_counts["loss_7"] = 0
    for r in records:
        if r.solved:
            dist_counts[str(r.turns_used)] += 1
        else:
            dist_counts["loss_7"] += 1

    dist_pct: Dict[str, float] = {
        k: round((100.0 * v) / float(n), 4) for k, v in dist_counts.items()
    }

    # Candidate-set trajectories after turn t in 1..6:
    # 1. Carried-forward: solved games carry forward 1 after the solving turn.
    # 2. Unsolved-after-turn: averaged strictly over games still unsolved after turn t.
    carried_forward_mean: Dict[str, float] = {}
    unsolved_after_turn_mean: Dict[str, Optional[float]] = {}
    unsolved_after_turn_count: Dict[str, int] = {}

    for t in range(1, MAX_TURNS + 1):
        cf_sum = 0
        unsolved_sum = 0
        unsolved_cnt = 0

        for r in records:
            if t <= r.turns_used:
                c_after_t = r.candidates_after[t - 1]
            else:
                # Game already solved on an earlier turn < t
                c_after_t = 1

            # If solved ON turn t, the game is solved after turn t (carries 1 in carried-forward)
            if r.solved and r.turns_used <= t:
                cf_sum += 1
            else:
                cf_sum += c_after_t
                unsolved_sum += c_after_t
                unsolved_cnt += 1

        carried_forward_mean[str(t)] = round(cf_sum / float(n), 4)
        unsolved_after_turn_count[str(t)] = unsolved_cnt
        unsolved_after_turn_mean[str(t)] = (
            round(unsolved_sum / float(unsolved_cnt), 4)
            if unsolved_cnt > 0
            else None
        )

    # Runtime breakdown
    all_turn_times = [ms for r in records for ms in r.turn_select_times_ms]
    turn1_times = [r.turn_select_times_ms[0] for r in records if r.turn_select_times_ms]
    later_times = [
        ms for r in records for ms in r.turn_select_times_ms[1:]
    ]

    opening_words = sorted({r.guesses[0] for r in records if r.guesses})

    return {
        "split": split_label,
        "num_games": n,
        "agent_name": records[0].agent_name,
        "agent_config": records[0].agent_config,
        "opening_words_used": opening_words,
        "avg_guesses_all_penalized7": round(avg_all, 6),
        "avg_guesses_won_only": round(avg_won, 6) if avg_won is not None else None,
        "win_rate_pct": round(win_rate_pct, 4),
        "wins": wins,
        "loss_count": losses,
        "max_win_turns": max_win_turns,
        "guess_distribution_counts": dist_counts,
        "guess_distribution_pct": dist_pct,
        "candidate_trajectory": {
            "carried_forward_mean": carried_forward_mean,
            "unsolved_after_turn_mean": unsolved_after_turn_mean,
            "unsolved_after_turn_count": unsolved_after_turn_count,
        },
        "runtime": {
            "total_turns": len(all_turn_times),
            "total_online_select_ms": round(sum(all_turn_times), 3),
            "mean_select_ms_per_guess": (
                round(sum(all_turn_times) / len(all_turn_times), 4)
                if all_turn_times
                else 0.0
            ),
            "mean_select_ms_turn1": (
                round(sum(turn1_times) / len(turn1_times), 4)
                if turn1_times
                else 0.0
            ),
            "mean_select_ms_turns2_to_6": (
                round(sum(later_times) / len(later_times), 4)
                if later_times
                else 0.0
            ),
            "opening_cache_used_turn1": any(r.used_opening_cache_turn1 for r in records),
            "cache_loaded_from_disk": cache_loaded_from_disk,
            "cache_load_or_build_ms": (
                round(cache_load_or_build_ms, 3)
                if cache_load_or_build_ms is not None
                else None
            ),
        },
    }


def slice_and_summarize_all_splits(
    all_records: Sequence[GameRecord],
    cache: FeedbackCache,
) -> Dict[str, Dict[str, object]]:
    """Slice canonical 2,315-word GameRecords into dev (463), test (1,852), and all (2,315) summaries."""
    dev_records = [r for r in all_records if r.split == "dev"]
    test_records = [r for r in all_records if r.split == "test"]

    out: Dict[str, Dict[str, object]] = {}
    if dev_records:
        out["dev"] = summarize_game_records(
            dev_records,
            split_label="dev",
            cache_load_or_build_ms=cache.load_or_build_time_ms,
            cache_loaded_from_disk=cache.loaded_from_disk,
        )
    if test_records:
        out["test"] = summarize_game_records(
            test_records,
            split_label="test",
            cache_load_or_build_ms=cache.load_or_build_time_ms,
            cache_loaded_from_disk=cache.loaded_from_disk,
        )
    if all_records:
        out["all"] = summarize_game_records(
            all_records,
            split_label="all",
            cache_load_or_build_ms=cache.load_or_build_time_ms,
            cache_loaded_from_disk=cache.loaded_from_disk,
        )
    return out


def _mean_std_min_max(values: Sequence[float]) -> Dict[str, float]:
    n = len(values)
    mean_val = sum(values) / float(n)
    var_val = sum((x - mean_val) ** 2 for x in values) / float(n) if n > 0 else 0.0
    return {
        "mean": round(mean_val, 6),
        "std": round(math.sqrt(var_val), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def evaluate_random_baseline_multi_seed(
    dictionary: WordleDictionary,
    cache: FeedbackCache,
    targets: Sequence[str],
    seeds: Sequence[int] = tuple(range(10)),
) -> Tuple[Dict[str, Dict[str, object]], List[GameRecord]]:
    """Evaluate RandomConsistentAgent across multiple run seeds, slicing dev/test/all from canonical records."""
    allowed_set = set(dictionary.allowed_guesses)
    agent = RandomConsistentAgent(dictionary, cache)

    all_seed_records: List[GameRecord] = []
    per_seed_summaries: Dict[str, List[Dict[str, object]]] = {
        "dev": [],
        "test": [],
        "all": [],
    }

    for run_seed in seeds:
        seed_records: List[GameRecord] = []
        for target in targets:
            rec = play_single_game(
                agent=agent,
                target=target,
                dictionary=dictionary,
                allowed_guesses_set=allowed_set,
                run_seed=run_seed,
            )
            seed_records.append(rec)
        all_seed_records.extend(seed_records)

        sliced = slice_and_summarize_all_splits(seed_records, cache=cache)
        for split_key, summary in sliced.items():
            per_seed_summaries[split_key].append(summary)

    # Pooled summary across all seeds + cross-seed variation stats
    pooled_by_split: Dict[str, Dict[str, object]] = {}
    for split_key in ("dev", "test", "all"):
        seed_list = per_seed_summaries[split_key]
        if not seed_list:
            continue
        if split_key == "all":
            split_records = all_seed_records
        else:
            split_records = [r for r in all_seed_records if r.split == split_key]

        pooled = summarize_game_records(
            split_records,
            split_label=split_key,
            cache_load_or_build_ms=cache.load_or_build_time_ms,
            cache_loaded_from_disk=cache.loaded_from_disk,
        )
        pooled["num_seeds"] = len(seeds)
        pooled["seeds"] = list(seeds)
        pooled["num_targets_per_seed"] = len(split_records) // len(seeds)
        pooled["seed_variation"] = {
            "avg_guesses_all_penalized7": _mean_std_min_max(
                [float(s["avg_guesses_all_penalized7"]) for s in seed_list]
            ),
            "avg_guesses_won_only": _mean_std_min_max(
                [
                    float(s["avg_guesses_won_only"])
                    for s in seed_list
                    if s["avg_guesses_won_only"] is not None
                ]
            ),
            "win_rate_pct": _mean_std_min_max(
                [float(s["win_rate_pct"]) for s in seed_list]
            ),
            "loss_count": _mean_std_min_max(
                [float(s["loss_count"]) for s in seed_list]
            ),
        }
        pooled_by_split[split_key] = pooled

    return pooled_by_split, all_seed_records


def evaluate_deterministic_agent(
    agent: BaseAgent,
    dictionary: WordleDictionary,
    cache: FeedbackCache,
    targets: Sequence[str],
) -> Tuple[Dict[str, Dict[str, object]], List[GameRecord]]:
    """Run a deterministic agent across targets and compute dev/test/all summaries from canonical records."""
    allowed_set = set(dictionary.allowed_guesses)
    records: List[GameRecord] = []
    for target in targets:
        rec = play_single_game(
            agent=agent,
            target=target,
            dictionary=dictionary,
            allowed_guesses_set=allowed_set,
            run_seed=0,
        )
        records.append(rec)

    summaries = slice_and_summarize_all_splits(records, cache=cache)
    return summaries, records


def tune_epsilon_on_dev(
    dictionary: WordleDictionary,
    cache: FeedbackCache,
    epsilons: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    allow_non_answers: bool = True,
) -> Dict[str, object]:
    """Tune near-tie answer bonus epsilon exclusively on the 463-word development split."""
    allowed_set = set(dictionary.allowed_guesses)
    grid_results: List[Dict[str, object]] = []

    for eps in epsilons:
        agent = EntropyAgent(
            dictionary=dictionary,
            cache=cache,
            epsilon=eps,
            allow_non_answers=allow_non_answers,
        )
        dev_records = [
            play_single_game(
                agent=agent,
                target=t,
                dictionary=dictionary,
                allowed_guesses_set=allowed_set,
                run_seed=0,
            )
            for t in dictionary.dev_targets
        ]
        summary = summarize_game_records(
            dev_records,
            split_label="dev",
            cache_load_or_build_ms=cache.load_or_build_time_ms,
            cache_loaded_from_disk=cache.loaded_from_disk,
        )
        summary["epsilon"] = float(eps)
        grid_results.append(summary)

    # Sort by primary metric: lowest avg_guesses_all_penalized7, then highest win_rate_pct,
    # then lowest avg_guesses_won_only, then smallest epsilon.
    best_entry = min(
        grid_results,
        key=lambda s: (
            float(s["avg_guesses_all_penalized7"]),
            -float(s["win_rate_pct"]),
            float(s["avg_guesses_won_only"] or 99.0),
            float(s["epsilon"]),
        ),
    )

    return {
        "split": "dev",
        "num_dev_targets": len(dictionary.dev_targets),
        "allow_non_answers": allow_non_answers,
        "best_epsilon": best_entry["epsilon"],
        "best_summary": best_entry,
        "grid_results": grid_results,
        "dictionary_metadata": dictionary.metadata_dict(),
    }


def save_game_records_jsonl(records: Iterable[GameRecord], path: Path) -> None:
    """Save per-game diagnostic records in JSON Lines format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

