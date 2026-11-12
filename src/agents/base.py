"""Base Agent interface and candidate answer tracking state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.dictionary import WordleDictionary
from src.engine import NUM_PATTERNS
from src.precompute import FeedbackCache


class BaseAgent(ABC):
    """Abstract base class for Wordle agents with explicit reset/select_guess/observe lifecycle."""

    def __init__(
        self,
        dictionary: WordleDictionary,
        cache: FeedbackCache,
        name: str,
    ) -> None:
        self.dictionary: WordleDictionary = dictionary
        self.cache: FeedbackCache = cache
        self.name: str = name

        self._all_answer_indices: np.ndarray = np.arange(
            dictionary.num_answers, dtype=np.int32
        )
        self.c_idx: np.ndarray = self._all_answer_indices.copy()
        self.history: List[Tuple[str, int]] = []
        self.guessed_words: Set[str] = set()
        self.current_seed: Optional[int] = None
        self._is_reset: bool = False

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset candidate answer state to all 2,315 possible answers before a new game."""
        self.c_idx = self._all_answer_indices.copy()
        self.history.clear()
        self.guessed_words.clear()
        self.current_seed = seed
        self._is_reset = True
        self._on_reset(seed=seed)

    def _on_reset(self, seed: Optional[int] = None) -> None:
        """Optional subclass hook invoked during reset()."""

    @property
    def num_candidates(self) -> int:
        """Return the number of remaining possible answers |C|."""
        return int(len(self.c_idx))

    def remaining_candidates(self) -> List[str]:
        """Return the sorted list of remaining possible answer strings."""
        return [self.dictionary.answers[int(i)] for i in self.c_idx]

    def observe(self, guess: str, pattern_id: int) -> int:
        """Filter remaining candidate answers C_idx using the observed feedback pattern_id."""
        if not self._is_reset:
            raise RuntimeError("Agent must be reset() before calling observe()")

        cleaned = guess.strip().lower()
        if cleaned not in self.dictionary.guess_to_idx:
            raise ValueError(f"Observed guess {cleaned!r} is not in the allowed dictionary")
        if not (0 <= int(pattern_id) < NUM_PATTERNS):
            raise ValueError(f"Observed pattern_id {pattern_id} must be in [0, 242]")

        g_idx = self.dictionary.guess_to_idx[cleaned]
        row_patterns = self.cache.pattern_matrix[g_idx, self.c_idx]
        self.c_idx = self.c_idx[row_patterns == int(pattern_id)]
        self.history.append((cleaned, int(pattern_id)))
        self.guessed_words.add(cleaned)

        if len(self.c_idx) == 0:
            raise RuntimeError(
                f"Diagnostic error: 0 candidate answers remain after observing "
                f"(guess={cleaned!r}, pattern_id={pattern_id}) on history={self.history}"
            )
        return int(len(self.c_idx))

    def select_guess(self, turns_remaining: int) -> str:
        """Select the next guess given the current candidate set C_idx and turns_remaining."""
        if not self._is_reset:
            raise RuntimeError("Agent must be reset() before calling select_guess()")
        if turns_remaining < 1:
            raise ValueError(f"turns_remaining must be >= 1, got {turns_remaining}")
        if len(self.c_idx) == 0:
            raise RuntimeError("Cannot select a guess when 0 candidate answers remain")

        # Universal immediate-win rule: if only 1 candidate answer remains, guess it.
        if len(self.c_idx) == 1:
            return self.dictionary.answers[int(self.c_idx[0])]

        return self._select_guess_impl(turns_remaining=turns_remaining)

    @abstractmethod
    def _select_guess_impl(self, turns_remaining: int) -> str:
        """Subclass policy when |C| > 1."""

    @abstractmethod
    def config_dict(self) -> Dict[str, object]:
        """Return serializable agent configuration parameters for experiment logs."""

