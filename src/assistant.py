"""Interactive Wordle Assistant & Compact Explanation Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.agents import BaseAgent, EntropyAgent
from src.dictionary import WordleDictionary
from src.engine import (
    ALL_GREEN,
    MAX_TURNS,
    compute_feedback_scalar,
    format_guess_ansi,
    format_pattern_ascii,
    parse_feedback_string,
)
from src.precompute import (
    FeedbackCache,
    compute_entropies,
    compute_entropy_scalar_row,
    compute_pattern_matrix_batched,
)


@dataclass(frozen=True)
class GuessStats:
    """Compact diagnostic statistics for a candidate guess against the active candidate set."""

    word: str
    entropy_bits: float
    in_candidates: bool
    num_groups: int
    max_group_size: int


def compute_word_stats(
    word: str,
    c_idx: np.ndarray,
    dictionary: WordleDictionary,
    cache: FeedbackCache,
    expanded_candidates: Optional[Sequence[str]] = None,
) -> GuessStats:
    """Compute one-step Shannon entropy, candidate membership, and partition stats for word."""
    cleaned = word.strip().lower()
    if expanded_candidates is not None:
        if len(expanded_candidates) == 0:
            return GuessStats(cleaned, 0.0, False, 0, 0)
        row_patterns = np.array(
            [compute_feedback_scalar(cleaned, cand) for cand in expanded_candidates],
            dtype=np.uint8,
        )
        in_c = cleaned in set(expanded_candidates)
    else:
        g_idx = dictionary.guess_to_idx[cleaned]
        ans_idx = int(dictionary.guess_idx_to_answer_idx[g_idx])
        in_c = bool(ans_idx >= 0 and np.any(c_idx == ans_idx))
        if len(c_idx) == 0:
            return GuessStats(cleaned, 0.0, False, 0, 0)
        row_patterns = cache.pattern_matrix[g_idx, c_idx]

    entropy = compute_entropy_scalar_row(row_patterns)
    counts = np.bincount(row_patterns)
    nonzero = counts[counts > 0]
    num_groups = int(len(nonzero))
    max_group = int(np.max(nonzero)) if num_groups > 0 else 0

    return GuessStats(
        word=cleaned,
        entropy_bits=entropy,
        in_candidates=in_c,
        num_groups=num_groups,
        max_group_size=max_group,
    )


def select_guess_over_expanded_candidates(
    expanded_candidates: Sequence[str],
    dictionary: WordleDictionary,
    turns_remaining: int,
    epsilon: float = 0.01,
    allow_non_answers: bool = True,
) -> str:
    """Select the highest-scoring entropy guess when operating on expanded 12,972-pool candidates."""
    if len(expanded_candidates) == 1:
        return expanded_candidates[0]

    restrict_to_c = (turns_remaining == 1) or (not allow_non_answers)
    cand_set = set(expanded_candidates)

    if restrict_to_c:
        pool = list(expanded_candidates)
        sub = compute_pattern_matrix_batched(pool, expanded_candidates)
        entropies = compute_entropies(sub)
        best_idx = int(np.argmax(entropies))
        return pool[best_idx]

    sub = compute_pattern_matrix_batched(dictionary.allowed_guesses, expanded_candidates)
    entropies = compute_entropies(sub)
    in_c_mask = np.array(
        [w in cand_set for w in dictionary.allowed_guesses], dtype=bool
    )
    scores = entropies + (epsilon * in_c_mask.astype(np.float64))
    max_score = float(np.max(scores))
    near_max = scores >= (max_score - 1e-12)
    tied_in_c = near_max & in_c_mask
    if np.any(tied_in_c):
        best_idx = int(np.flatnonzero(tied_in_c)[0])
    else:
        best_idx = int(np.flatnonzero(near_max)[0])
    return dictionary.allowed_guesses[best_idx]


def format_candidate_preview(candidates: Sequence[str], max_show: int = 10) -> str:
    """Format up to max_show candidate words with a trailing count if more remain."""
    shown = list(candidates[:max_show])
    preview = ", ".join(w.upper() for w in shown)
    extra = len(candidates) - len(shown)
    if extra > 0:
        preview += f" ... (+{extra} more)"
    return preview


def run_interactive_assistant(
    agent: BaseAgent,
    dictionary: WordleDictionary,
    cache: FeedbackCache,
    target: Optional[str] = None,
    max_candidates_preview: int = 10,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> List[Tuple[str, int]]:
    """Run an interactive terminal assistant session (External Helper or Practice Mode)."""
    cleaned_target: Optional[str] = None
    if target is not None:
        cleaned_target = target.strip().lower()
        if cleaned_target not in dictionary.guess_to_idx:
            raise ValueError(
                f"Practice target {cleaned_target!r} is not in the allowed Wordle dictionary"
            )

    agent.reset(seed=0)
    # Also track consistent words across the full 12,972 allowed dictionary so NYT words
    # outside the original 2,315 list (e.g., KARAT) are handled seamlessly.
    expanded_pool: List[str] = list(dictionary.allowed_guesses)
    using_expanded_pool: bool = False

    # State stack for undo: (c_idx_copy, history_copy, guessed_words_copy, expanded_pool_copy, using_expanded)
    undo_stack: List[Tuple[np.ndarray, List[Tuple[str, int]], set, List[str], bool]] = []

    mode_banner = (
        "PRACTICE / COMPARISON MODE (Simulator auto-scores your guesses)"
        if cleaned_target is not None
        else "EXTERNAL WORDLE HELPER (Enter your guess + Wordle's 5-tile feedback)"
    )
    output_fn("\n" + "=" * 74)
    output_fn(f"Wordle Interactive Assistant ({agent.name}) — {mode_banner}")
    output_fn(
        "Controls: Press Enter to accept #1 recommendation | 'undo' to step back | 'q' to quit"
    )
    output_fn("=" * 74)

    eps = agent.epsilon if isinstance(agent, EntropyAgent) else 0.01
    allow_non = agent.allow_non_answers if isinstance(agent, EntropyAgent) else True

    while len(agent.history) < MAX_TURNS:
        turn_num = len(agent.history) + 1
        turns_remaining = MAX_TURNS - len(agent.history)

        if not using_expanded_pool:
            c_list = agent.remaining_candidates()
            pool_label = "|C|"
        else:
            c_list = expanded_pool
            pool_label = "|C_expanded (12,972 pool)|"

        num_c = len(c_list)
        output_fn(f"\n--- Turn {turn_num}/{MAX_TURNS} | Remaining Candidates ({pool_label}): {num_c} ---")
        output_fn(
            f"Candidates:  {format_candidate_preview(c_list, max_show=max_candidates_preview)}"
        )

        if not using_expanded_pool:
            rec_word = agent.select_guess(turns_remaining=turns_remaining)
            rec_stats = compute_word_stats(rec_word, agent.c_idx, dictionary, cache)
        else:
            rec_word = select_guess_over_expanded_candidates(
                expanded_candidates=expanded_pool,
                dictionary=dictionary,
                turns_remaining=turns_remaining,
                epsilon=eps,
                allow_non_answers=allow_non,
            )
            rec_stats = compute_word_stats(
                rec_word,
                agent.c_idx,
                dictionary,
                cache,
                expanded_candidates=expanded_pool,
            )

        rec_tag = "possible answer" if rec_stats.in_candidates else "exploratory guess"

        if num_c == 1:
            output_fn(
                f"Recommend:   {rec_word.upper()}  [Sole remaining answer — 100% win]"
            )
        else:
            output_fn(
                f"Recommend:   {rec_word.upper()}  "
                f"[H = {rec_stats.entropy_bits:.3f} bits | {rec_tag} | "
                f"splits {num_c} -> {rec_stats.num_groups} groups]"
            )

        # 1. Prompt for guess
        try:
            raw_guess = input_fn(
                f"Your guess [Enter='{rec_word.upper()}', 'undo', 'q']: "
            ).strip().lower()
        except EOFError:
            output_fn("\nExiting assistant.")
            break

        if raw_guess in ("q", "quit", "exit"):
            output_fn("Exiting assistant.")
            break

        if raw_guess in ("undo", "back"):
            if not undo_stack:
                output_fn("  [!] Nothing to undo (already at Turn 1).")
            else:
                prev_c, prev_hist, prev_guessed, prev_exp, prev_flag = undo_stack.pop()
                agent.c_idx = prev_c
                agent.history = prev_hist
                agent.guessed_words = prev_guessed
                expanded_pool = prev_exp
                using_expanded_pool = prev_flag
                active_n = len(expanded_pool) if using_expanded_pool else agent.num_candidates
                output_fn(f"  [Undo] Reverted to Turn {len(agent.history) + 1} (|C| = {active_n}).")
            continue

        chosen_guess = rec_word if raw_guess == "" else raw_guess
        if chosen_guess not in dictionary.guess_to_idx:
            output_fn(
                f"  [!] {chosen_guess.upper()!r} is not in the 12,972 allowed Wordle dictionary. Try again."
            )
            continue

        # Show compact comparison if user picked a different guess than #1
        if chosen_guess != rec_word:
            user_stats = compute_word_stats(
                chosen_guess,
                agent.c_idx,
                dictionary,
                cache,
                expanded_candidates=expanded_pool if using_expanded_pool else None,
            )
            u_tag = "possible answer" if user_stats.in_candidates else "exploratory"
            output_fn(
                f"  Comparison: Your guess {chosen_guess.upper()} "
                f"(H = {user_stats.entropy_bits:.3f} bits, {u_tag}, {user_stats.num_groups} groups) "
                f"vs #1 {rec_word.upper()} (H = {rec_stats.entropy_bits:.3f} bits, {rec_tag}, {rec_stats.num_groups} groups)"
            )

        # 2. Obtain or prompt for feedback
        if cleaned_target is not None:
            pattern_id = compute_feedback_scalar(chosen_guess, cleaned_target)
            next_expanded = [
                w for w in expanded_pool if compute_feedback_scalar(chosen_guess, w) == pattern_id
            ]
        else:
            while True:
                try:
                    raw_fb = input_fn(
                        f"Feedback for {chosen_guess.upper()} (G/2=Green, Y/1=Yellow, ./B/0=Gray, or 'back'/'q'): "
                    ).strip()
                except EOFError:
                    return list(agent.history)

                low_fb = raw_fb.lower()
                if low_fb in ("q", "quit", "exit"):
                    output_fn("Exiting assistant.")
                    return list(agent.history)
                if low_fb in ("back", "undo"):
                    pattern_id = -1
                    next_expanded = []
                    break

                try:
                    pattern_id = parse_feedback_string(raw_fb)
                except ValueError as exc:
                    output_fn(f"  [!] {exc}")
                    continue

                g_idx = dictionary.guess_to_idx[chosen_guess]
                matching_c = (
                    int(np.sum(cache.pattern_matrix[g_idx, agent.c_idx] == pattern_id))
                    if not using_expanded_pool
                    else 0
                )
                next_expanded = [
                    w for w in expanded_pool if compute_feedback_scalar(chosen_guess, w) == pattern_id
                ]

                if not using_expanded_pool and matching_c == 0 and len(next_expanded) == 0:
                    output_fn(
                        f"  [!] Feedback [{format_pattern_ascii(pattern_id)}] for {chosen_guess.upper()} "
                        f"leaves 0 possible answers in both the 2,315 list and the full 12,972 dictionary! "
                        f"Check for a typo or type 'back'."
                    )
                    continue

                if using_expanded_pool and len(next_expanded) == 0:
                    output_fn(
                        f"  [!] Feedback [{format_pattern_ascii(pattern_id)}] for {chosen_guess.upper()} "
                        f"leaves 0 possible words in the expanded 12,972 dictionary! "
                        f"Check for a typo or type 'back'."
                    )
                    continue
                break

            if pattern_id == -1:
                continue

        # Save state for undo
        undo_stack.append(
            (
                agent.c_idx.copy(),
                list(agent.history),
                set(agent.guessed_words),
                list(expanded_pool),
                using_expanded_pool,
            )
        )

        # Update candidate sets
        g_idx = dictionary.guess_to_idx[chosen_guess]
        matching_c = (
            int(np.sum(cache.pattern_matrix[g_idx, agent.c_idx] == pattern_id))
            if not using_expanded_pool
            else 0
        )

        if not using_expanded_pool and matching_c > 0:
            new_count = agent.observe(chosen_guess, pattern_id)
            expanded_pool = next_expanded
        else:
            # Target is outside the original 2,315 list (e.g., NYT expanded word like KARAT)
            if not using_expanded_pool:
                output_fn(
                    f"\n  [Note] 0 words remain in the original 2,315 answer list, "
                    f"but {len(next_expanded)} word(s) in the full 12,972 Wordle dictionary "
                    f"match all feedback so far (NYT Wordle now includes expanded-dictionary words)!"
                )
                using_expanded_pool = True
            agent.c_idx = np.empty(0, dtype=np.int32)
            agent.history.append((chosen_guess, int(pattern_id)))
            agent.guessed_words.add(chosen_guess)
            expanded_pool = next_expanded
            new_count = len(expanded_pool)

        # Render board history so far
        output_fn("\nBoard:")
        for idx, (h_guess, h_pid) in enumerate(agent.history, start=1):
            output_fn(
                f"  {idx}. {format_guess_ansi(h_guess, h_pid)}   [{format_pattern_ascii(h_pid)}]"
            )
        output_fn(f"Candidates reduced: {num_c} -> {new_count}")

        if pattern_id == ALL_GREEN:
            output_fn(f"\nSolved in {len(agent.history)}/{MAX_TURNS} guesses! ({chosen_guess.upper()})")
            break

        if len(agent.history) == MAX_TURNS:
            if cleaned_target is not None:
                output_fn(f"\nOut of guesses (6/6)! Hidden target was: {cleaned_target.upper()}")
            else:
                active_cands = expanded_pool if using_expanded_pool else agent.remaining_candidates()
                rem = format_candidate_preview(active_cands, max_show=10)
                output_fn(f"\nReached 6/6 guesses. Remaining candidates ({new_count}): {rem}")
            break

    return list(agent.history)
