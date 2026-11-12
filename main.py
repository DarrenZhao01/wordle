"""Unified terminal CLI for the Wordle Simulator, Baseline Agents, and Entropy Solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List, Sequence

from src.agents import BaseAgent, EntropyAgent, LetterFrequencyAgent, RandomConsistentAgent
from src.dictionary import DEFAULT_DATA_DIR, WordleDictionary, load_dictionary
from src.engine import format_guess_ansi, format_pattern_ascii
from src.evaluate import (
    GameRecord,
    evaluate_deterministic_agent,
    evaluate_random_baseline_multi_seed,
    play_single_game,
    save_game_records_jsonl,
    summarize_game_records,
    tune_epsilon_on_dev,
)
from src.precompute import FeedbackCache, load_or_build_cache


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _build_agent(
    agent_key: str,
    dictionary: WordleDictionary,
    cache: FeedbackCache,
    epsilon: float = 0.01,
    allow_non_answers: bool = True,
    unique_letters_only: bool = False,
    seed: int = 0,
) -> BaseAgent:
    key = agent_key.strip().lower()
    if key in ("random", "random_consistent"):
        return RandomConsistentAgent(dictionary, cache, default_seed=seed)
    if key in ("frequency", "letter_frequency"):
        return LetterFrequencyAgent(
            dictionary, cache, unique_letters_only=unique_letters_only
        )
    if key == "frequency_unique":
        return LetterFrequencyAgent(dictionary, cache, unique_letters_only=True)
    if key == "entropy":
        return EntropyAgent(
            dictionary,
            cache,
            epsilon=epsilon,
            allow_non_answers=allow_non_answers,
        )
    if key == "entropy_candidates_only":
        return EntropyAgent(
            dictionary,
            cache,
            epsilon=epsilon,
            allow_non_answers=False,
        )
    raise ValueError(f"Unknown agent type: {agent_key!r}")


def _print_game_record(record: GameRecord, dictionary: WordleDictionary, cache: FeedbackCache) -> None:
    status = f"SOLVED in {record.turns_used}/6" if record.solved else "LOSS (7 penalty)"
    print(
        f"\n=== Target: {record.target.upper()} ({record.split} split) | "
        f"Agent: {record.agent_name} | {status} ==="
    )
    print(f"Game Seed: {record.game_seed} (from run_seed={record.run_seed})")
    print("-" * 74)
    print(f"{'Turn':<5} {'Guess':<22} {'Pattern':<9} {'|C| Before -> After':<22} {'Time (ms)':<10}")
    print("-" * 74)

    for idx, (g, pid, pat_str, cb, ca, ms) in enumerate(
        zip(
            record.guesses,
            record.pattern_ids,
            record.pattern_ascii,
            record.candidates_before,
            record.candidates_after,
            record.turn_select_times_ms,
        ),
        start=1,
    ):
        ansi_tiles = format_guess_ansi(g, pid)
        c_trans = f"{cb:>4d} -> {ca:<4d}"
        print(f"{idx:<5} {ansi_tiles}   [{pat_str}]   {c_trans:<22} {ms:>8.3f}")

    print("-" * 74)


def cmd_play(args: argparse.Namespace) -> None:
    dictionary = load_dictionary(DEFAULT_DATA_DIR)
    cache = load_or_build_cache(dictionary, DEFAULT_DATA_DIR, force_rebuild=args.rebuild_cache)
    allowed_set = set(dictionary.allowed_guesses)

    CacheStatus = "disk cache" if cache.loaded_from_disk else "freshly built"
    print(
        f"Loaded {dictionary.num_answers} answers & {dictionary.num_guesses} allowed guesses "
        f"({CacheStatus} in {cache.load_or_build_time_ms:.1f} ms)."
    )

    agent = _build_agent(
        agent_key=args.agent,
        dictionary=dictionary,
        cache=cache,
        epsilon=args.epsilon,
        allow_non_answers=not args.candidates_only,
        unique_letters_only=args.unique_letters,
        seed=args.seed,
    )

    if args.target:
        targets = [t.strip().lower() for t in args.target.split(",") if t.strip()]
        for t in targets:
            if t not in dictionary.answer_to_idx:
                raise ValueError(f"Target word {t!r} is not in the 2,315 answer dictionary")
    else:
        if args.split == "dev":
            pool = list(dictionary.dev_targets)
        elif args.split == "test":
            pool = list(dictionary.test_targets)
        else:
            pool = list(dictionary.answers)
        rng = random.Random(args.seed)
        targets = rng.sample(pool, min(args.num_games, len(pool)))

    records: List[GameRecord] = []
    for target in targets:
        rec = play_single_game(
            agent=agent,
            target=target,
            dictionary=dictionary,
            allowed_guesses_set=allowed_set,
            run_seed=args.seed,
        )
        records.append(rec)
        if not args.quiet:
            _print_game_record(rec, dictionary, cache)

    if len(records) > 1:
        summary = summarize_game_records(
            records,
            split_label=args.split if not args.target else "custom",
            cache_load_or_build_ms=cache.load_or_build_time_ms,
            cache_loaded_from_disk=cache.loaded_from_disk,
        )
        print(
            f"\nSummary over {len(records)} games ({agent.name}): "
            f"Avg Guesses (All/Pen7) = {summary['avg_guesses_all_penalized7']:.4f} | "
            f"Avg Guesses (Won) = {summary['avg_guesses_won_only']:.4f} | "
            f"Win Rate = {summary['win_rate_pct']:.2f}% ({summary['wins']}/{len(records)}) | "
            f"Mean Select Time = {summary['runtime']['mean_select_ms_per_guess']:.3f} ms/guess"
        )


def cmd_tune(args: argparse.Namespace) -> None:
    dictionary = load_dictionary(DEFAULT_DATA_DIR)
    cache = load_or_build_cache(dictionary, DEFAULT_DATA_DIR, force_rebuild=args.rebuild_cache)

    epsilons = [float(x) for x in args.epsilons.split(",")]
    print(
        f"Tuning epsilon on 20% Development Split ({len(dictionary.dev_targets)} targets, "
        f"each initialized with all {dictionary.num_answers} candidate answers)..."
    )

    result = tune_epsilon_on_dev(
        dictionary=dictionary,
        cache=cache,
        epsilons=epsilons,
        allow_non_answers=not args.candidates_only,
    )

    print("\n" + "=" * 88)
    print(
        f"{'Epsilon':<10} {'Opener':<9} {'Avg (All/Pen7)':<16} {'Avg (Won)':<12} "
        f"{'Win Rate %':<12} {'Losses':<8} {'ms/guess':<10}"
    )
    print("-" * 88)
    for row in result["grid_results"]:
        opener = ",".join(row["opening_words_used"])
        print(
            f"{row['epsilon']:<10.4f} {opener:<9} "
            f"{row['avg_guesses_all_penalized7']:<16.4f} "
            f"{row['avg_guesses_won_only']:<12.4f} "
            f"{row['win_rate_pct']:<12.2f} "
            f"{row['loss_count']:<8d} "
            f"{row['runtime']['mean_select_ms_per_guess']:<10.3f}"
        )
    print("=" * 88)
    print(f"Selected Best Epsilon on Dev Split: epsilon* = {result['best_epsilon']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved tuning results to {out_path}")


def _print_split_comparison_table(
    split_key: str,
    agent_summaries: Sequence[Dict[str, object]],
) -> None:
    if not agent_summaries:
        return
    num_targets = agent_summaries[0].get(
        "num_targets_per_seed", agent_summaries[0]["num_games"]
    )
    print(f"\n=== Benchmark Summary — Split: {split_key.upper()} ({num_targets} targets) ===")
    print("-" * 108)
    print(
        f"{'Agent':<26} {'Opener':<8} {'Avg (All/Pen7)':<20} {'Avg (Won Only)':<16} "
        f"{'Win Rate %':<16} {'MaxWin/Loss':<12} {'ms/guess':<8}"
    )
    print("-" * 108)
    for s in agent_summaries:
        name = str(s["agent_name"])
        openers = s["opening_words_used"]
        opener_str = openers[0] if len(openers) == 1 else f"({len(openers)} words)"
        if "seed_variation" in s:
            sv = s["seed_variation"]
            avg_all_str = f"{sv['avg_guesses_all_penalized7']['mean']:.4f} ±{sv['avg_guesses_all_penalized7']['std']:.4f}"
            avg_won_str = f"{sv['avg_guesses_won_only']['mean']:.4f} ±{sv['avg_guesses_won_only']['std']:.4f}"
            win_str = f"{sv['win_rate_pct']['mean']:.2f}% ±{sv['win_rate_pct']['std']:.2f}%"
            max_loss_str = f"{s['max_win_turns']}/{sv['loss_count']['mean']:.1f}"
        else:
            avg_all_str = f"{s['avg_guesses_all_penalized7']:.4f}"
            avg_won_str = f"{s['avg_guesses_won_only']:.4f}"
            win_str = f"{s['win_rate_pct']:.2f}%"
            max_loss_str = f"{s['max_win_turns']}/{s['loss_count']}"
        ms_str = f"{s['runtime']['mean_select_ms_per_guess']:.3f}"
        print(
            f"{name:<26} {opener_str:<8} {avg_all_str:<20} {avg_won_str:<16} "
            f"{win_str:<16} {max_loss_str:<12} {ms_str:<8}"
        )
    print("-" * 108)

    # Print candidate trajectory (carried-forward and unsolved-only)
    print("Candidate Set Trajectory After Turn t (Carried-Forward Mean | Unsolved-Only Mean [count]):")
    for s in agent_summaries:
        cf = s["candidate_trajectory"]["carried_forward_mean"]
        uo = s["candidate_trajectory"]["unsolved_after_turn_mean"]
        uc = s["candidate_trajectory"]["unsolved_after_turn_count"]
        cf_str = "  ".join(f"t{t}:{cf[str(t)]:.2f}" for t in range(1, 7))
        uo_parts = []
        for t in range(1, 7):
            val = uo[str(t)]
            cnt = uc[str(t)]
            val_s = f"{val:.1f}" if val is not None else "null"
            uo_parts.append(f"t{t}:{val_s}[{cnt}]")
        uo_str = "  ".join(uo_parts)
        print(f"  {s['agent_name']:<24} CF: {cf_str}")
        print(f"  {'':<24} UO: {uo_str}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    dictionary = load_dictionary(DEFAULT_DATA_DIR)
    cache = load_or_build_cache(dictionary, DEFAULT_DATA_DIR, force_rebuild=args.rebuild_cache)

    targets = list(dictionary.answers)
    if args.limit is not None and args.limit > 0:
        targets = targets[: args.limit]

    epsilon = args.epsilon
    tuning_path = DEFAULT_RESULTS_DIR / "tuning_dev.json"
    if epsilon is None:
        if tuning_path.exists():
            try:
                tdata = json.loads(tuning_path.read_text(encoding="utf-8"))
                epsilon = float(tdata["best_epsilon"])
                print(f"Loaded tuned epsilon = {epsilon} from {tuning_path}")
            except (KeyError, ValueError, json.JSONDecodeError):
                epsilon = 0.01
        else:
            epsilon = 0.01

    agent_keys = [k.strip() for k in args.agents.split(",") if k.strip()]
    results_by_agent: Dict[str, Dict[str, Dict[str, object]]] = {}
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for key in agent_keys:
        if key in ("random", "random_consistent"):
            seeds = list(range(args.num_seeds))
            print(f"Evaluating RandomConsistentAgent across {len(seeds)} seeds on {len(targets)} targets...")
            summaries, records = evaluate_random_baseline_multi_seed(
                dictionary=dictionary,
                cache=cache,
                targets=targets,
                seeds=seeds,
            )
            results_by_agent["random_consistent"] = summaries
            if args.save_logs:
                save_game_records_jsonl(records, results_dir / "game_logs_random_consistent.jsonl")
        else:
            agent = _build_agent(
                agent_key=key,
                dictionary=dictionary,
                cache=cache,
                epsilon=epsilon,
            )
            print(f"Evaluating {agent.name} (config={agent.config_dict()}) on {len(targets)} targets...")
            summaries, records = evaluate_deterministic_agent(
                agent=agent,
                dictionary=dictionary,
                cache=cache,
                targets=targets,
            )
            results_by_agent[agent.name] = summaries
            if args.save_logs:
                save_game_records_jsonl(records, results_dir / f"game_logs_{agent.name}.jsonl")

    for split_key in ("dev", "test", "all"):
        split_list = [
            agent_dict[split_key]
            for agent_dict in results_by_agent.values()
            if split_key in agent_dict
        ]
        _print_split_comparison_table(split_key, split_list)

    full_payload = {
        "dictionary_metadata": dictionary.metadata_dict(),
        "cache_metadata": cache.metadata,
        "epsilon_used": epsilon,
        "num_targets_evaluated": len(targets),
        "agents": results_by_agent,
    }
    summary_path = results_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark summary to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wordle Agent & Simulator CLI Prototype (Weeks 1-4)"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force regeneration of pattern_matrix.npy and opening_entropies.npy",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # play subcommand
    p_play = sub.add_parser("play", help="Simulate one or more Wordle games in the terminal")
    p_play.add_argument(
        "--agent",
        default="entropy",
        choices=[
            "entropy",
            "entropy_candidates_only",
            "frequency",
            "frequency_unique",
            "random",
        ],
        help="Agent policy to run",
    )
    p_play.add_argument(
        "--target",
        type=str,
        default=None,
        help="Hidden target word(s), comma-separated (e.g., crane,apple). Omit to sample.",
    )
    p_play.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to sample when --target is omitted",
    )
    p_play.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="all",
        help="Split to sample targets from when --target is omitted",
    )
    p_play.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Near-tie candidate preference bonus for EntropyAgent",
    )
    p_play.add_argument(
        "--candidates-only",
        action="store_true",
        help="Restrict EntropyAgent guesses strictly to remaining candidate answers C_idx",
    )
    p_play.add_argument(
        "--unique-letters",
        action="store_true",
        help="Use duplicate-letter discount variant for LetterFrequencyAgent",
    )
    p_play.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Run seed for target sampling and RandomConsistentAgent",
    )
    p_play.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-game board output when playing multiple games",
    )
    p_play.set_defaults(func=cmd_play)

    # tune subcommand
    p_tune = sub.add_parser("tune", help="Tune EntropyAgent epsilon on the 20% dev split (463 words)")
    p_tune.add_argument(
        "--epsilons",
        type=str,
        default="0.0,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50",
        help="Comma-separated list of epsilon values to evaluate on dev split",
    )
    p_tune.add_argument(
        "--candidates-only",
        action="store_true",
        help="Tune with allow_non_answers=False (restrict to C_idx)",
    )
    p_tune.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "tuning_dev.json"),
        help="Output JSON path for dev tuning results",
    )
    p_tune.set_defaults(func=cmd_tune)

    # evaluate subcommand
    p_eval = sub.add_parser(
        "evaluate",
        help="Run benchmark evaluation and report dev, test, and all-word summaries",
    )
    p_eval.add_argument(
        "--agents",
        type=str,
        default="random,frequency,entropy,entropy_candidates_only",
        help="Comma-separated list of agents to evaluate",
    )
    p_eval.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon for EntropyAgent (loads from tuning_dev.json if omitted)",
    )
    p_eval.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds (0..N-1) for RandomConsistentAgent",
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of targets for quick smoke checks",
    )
    p_eval.add_argument(
        "--save-logs",
        action="store_true",
        default=True,
        help="Save per-game JSONL diagnostic logs (enabled by default)",
    )
    p_eval.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to write benchmark_summary.json and per-game JSONL logs",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

