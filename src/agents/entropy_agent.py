"""Information-Gain (Shannon Entropy) Solver with near-tie answer bonus and final-turn rule."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent
from src.dictionary import WordleDictionary
from src.precompute import FeedbackCache, compute_entropies_from_transposed


TIE_TOLERANCE: float = 1e-12


class EntropyAgent(BaseAgent):
    """Primary Shannon entropy solver over allowed guesses with epsilon answer preference."""

    def __init__(
        self,
        dictionary: WordleDictionary,
        cache: FeedbackCache,
        epsilon: float = 0.0,
        allow_non_answers: bool = True,
        use_opening_cache: bool = True,
    ) -> None:
        if epsilon < 0.0 or not np.isfinite(epsilon):
            raise ValueError(f"epsilon must be a finite non-negative float, got {epsilon}")
        mode_suffix = "" if allow_non_answers else "_candidates_only"
        super().__init__(
            dictionary=dictionary,
            cache=cache,
            name=f"entropy{mode_suffix}",
        )
        self.epsilon: float = float(epsilon)
        self.allow_non_answers: bool = bool(allow_non_answers)
        self.use_opening_cache: bool = bool(use_opening_cache)
        self.last_used_opening_cache: bool = False

    def _on_reset(self, seed: Optional[int] = None) -> None:
        self.last_used_opening_cache = False

    def evaluate_candidates(
        self, turns_remaining: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (guess_indices, entropies, scores, in_c_mask) for the active search pool."""
        is_opening_state = (
            len(self.history) == 0
            and len(self.c_idx) == self.dictionary.num_answers
        )
        c_guess_indices = self.dictionary.answer_idx_to_guess_idx[self.c_idx]

        # Final-turn rule (turns_remaining == 1) or candidate-only restriction:
        # Restrict candidate guesses strictly to the current remaining answers C_idx.
        restrict_to_c = (turns_remaining == 1) or (not self.allow_non_answers)

        if restrict_to_c:
            guess_indices = c_guess_indices
            if is_opening_state and self.use_opening_cache:
                entropies = self.cache.opening_entropies[guess_indices].copy()
                self.last_used_opening_cache = True
            else:
                sub_t = self.cache.pattern_matrix_t[np.ix_(self.c_idx, guess_indices)]
                entropies = compute_entropies_from_transposed(sub_t)
                self.last_used_opening_cache = False

            in_c_mask = np.ones(len(guess_indices), dtype=bool)
            scores = entropies.copy()
            return guess_indices, entropies, scores, in_c_mask

        # Full allowed dictionary evaluation (all 12,972 guesses)
        guess_indices = np.arange(self.dictionary.num_guesses, dtype=np.int32)
        if is_opening_state and self.use_opening_cache:
            entropies = self.cache.opening_entropies.copy()
            self.last_used_opening_cache = True
        else:
            sub_t = self.cache.pattern_matrix_t[self.c_idx, :]
            entropies = compute_entropies_from_transposed(sub_t)
            self.last_used_opening_cache = False

        in_c_mask = np.zeros(self.dictionary.num_guesses, dtype=bool)
        in_c_mask[c_guess_indices] = True

        scores = entropies + (self.epsilon * in_c_mask.astype(np.float64))
        return guess_indices, entropies, scores, in_c_mask

    def _select_guess_impl(self, turns_remaining: int) -> str:
        guess_indices, _, scores, in_c_mask = self.evaluate_candidates(
            turns_remaining=turns_remaining
        )
        max_score = float(np.max(scores))
        near_max_mask = scores >= (max_score - TIE_TOLERANCE)

        # Among exact/floating-point ties for the maximum score, prefer membership in C_idx,
        # then break remaining ties alphabetically (lowest index in sorted allowed_guesses).
        tied_in_c = near_max_mask & in_c_mask
        if np.any(tied_in_c):
            chosen_pos = int(np.flatnonzero(tied_in_c)[0])
        else:
            chosen_pos = int(np.flatnonzero(near_max_mask)[0])

        best_guess_idx = int(guess_indices[chosen_pos])
        return self.dictionary.allowed_guesses[best_guess_idx]

    def config_dict(self) -> Dict[str, object]:
        return {
            "agent_type": "entropy",
            "epsilon": self.epsilon,
            "allow_non_answers": self.allow_non_answers,
            "final_turn_rule": "restrict_to_C_idx_highest_entropy_alphabetical",
            "use_opening_cache": self.use_opening_cache,
            "tie_tolerance": TIE_TOLERANCE,
            "tie_break": "in_C_idx_then_alphabetical",
        }
