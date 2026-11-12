"""Random-Consistent Baseline Agent choosing uniformly among remaining possible answers."""

from __future__ import annotations

import random
from typing import Dict, Optional

from src.agents.base import BaseAgent
from src.dictionary import WordleDictionary
from src.precompute import FeedbackCache


class RandomConsistentAgent(BaseAgent):
    """Selects uniformly at random from the remaining possible answers C_idx."""

    def __init__(
        self,
        dictionary: WordleDictionary,
        cache: FeedbackCache,
        default_seed: int = 0,
    ) -> None:
        super().__init__(dictionary=dictionary, cache=cache, name="random_consistent")
        self.default_seed: int = int(default_seed)
        self._rng: random.Random = random.Random(self.default_seed)

    def _on_reset(self, seed: Optional[int] = None) -> None:
        effective_seed = self.default_seed if seed is None else int(seed)
        self.current_seed = effective_seed
        self._rng = random.Random(effective_seed)

    def _select_guess_impl(self, turns_remaining: int) -> str:
        chosen_pos = self._rng.randrange(len(self.c_idx))
        ans_idx = int(self.c_idx[chosen_pos])
        return self.dictionary.answers[ans_idx]

    def config_dict(self) -> Dict[str, object]:
        return {
            "agent_type": "random_consistent",
            "candidate_pool": "remaining_answers_C_idx",
            "default_seed": self.default_seed,
        }

