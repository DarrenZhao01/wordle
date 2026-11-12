"""Positional Letter-Frequency Baseline Agent over remaining possible answers C_idx."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.agents.base import BaseAgent
from src.dictionary import WordleDictionary
from src.precompute import FeedbackCache, words_to_char_codes


class LetterFrequencyAgent(BaseAgent):
    """Scores and chooses among remaining possible answers C_idx using positional frequencies."""

    def __init__(
        self,
        dictionary: WordleDictionary,
        cache: FeedbackCache,
        unique_letters_only: bool = False,
    ) -> None:
        name = (
            "letter_frequency_unique"
            if unique_letters_only
            else "letter_frequency"
        )
        super().__init__(dictionary=dictionary, cache=cache, name=name)
        self.unique_letters_only: bool = bool(unique_letters_only)
        # Pre-encode all 2,315 answers as (N_answers, 5) uint8 in 0..25
        self._answer_char_codes: np.ndarray = words_to_char_codes(dictionary.answers)

    def score_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (raw_integer_scores, normalized_scores) for each candidate in self.c_idx."""
        k = len(self.c_idx)
        c_codes = self._answer_char_codes[self.c_idx]  # shape (K, 5)

        # pos_counts[pos, ch] = number of remaining answers with letter ch at position pos
        pos_counts = np.zeros((5, 26), dtype=np.int32)
        for pos in range(5):
            pos_counts[pos] = np.bincount(c_codes[:, pos], minlength=26)

        if not self.unique_letters_only:
            # Direct positional frequency sum: sum_{i=0..4} count(i, g[i])
            raw_scores = np.zeros(k, dtype=np.int32)
            for pos in range(5):
                raw_scores += pos_counts[pos, c_codes[:, pos]]
        else:
            # Optional duplicate-discount variant: only highest positional count per distinct letter
            raw_scores = np.zeros(k, dtype=np.int32)
            for row_idx in range(k):
                word_codes = c_codes[row_idx]
                best_per_letter: Dict[int, int] = {}
                for pos in range(5):
                    ch = int(word_codes[pos])
                    val = int(pos_counts[pos, ch])
                    if ch not in best_per_letter or val > best_per_letter[ch]:
                        best_per_letter[ch] = val
                raw_scores[row_idx] = sum(best_per_letter.values())

        norm_scores = raw_scores.astype(np.float64) / float(k)
        return raw_scores, norm_scores

    def _select_guess_impl(self, turns_remaining: int) -> str:
        raw_scores, _ = self.score_candidates()
        # np.argmax returns the first index of the maximum, which breaks ties
        # alphabetically because self.c_idx and dictionary.answers are sorted.
        best_local_idx = int(np.argmax(raw_scores))
        best_ans_idx = int(self.c_idx[best_local_idx])
        return self.dictionary.answers[best_ans_idx]

    def config_dict(self) -> Dict[str, object]:
        return {
            "agent_type": "letter_frequency",
            "candidate_pool": "remaining_answers_C_idx",
            "unique_letters_only": self.unique_letters_only,
            "tie_break": "alphabetical",
        }

