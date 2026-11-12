"""Core Wordle feedback engine (scalar reference) and Game Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


GRAY: int = 0
YELLOW: int = 1
GREEN: int = 2

POWERS_OF_3: Tuple[int, int, int, int, int] = (1, 3, 9, 27, 81)
NUM_PATTERNS: int = 243
ALL_GREEN: int = 242  # 2*1 + 2*3 + 2*9 + 2*27 + 2*81
MAX_TURNS: int = 6
ENCODING_VERSION: str = "base3_lsb_pos0_v1"


def encode_trits(trits: Sequence[int]) -> int:
    """Encode 5 feedback trits (each in {0, 1, 2}) into a pattern_id in [0, 242]."""
    if len(trits) != 5:
        raise ValueError(f"Expected 5 trits, got {len(trits)}")
    pattern_id = 0
    for i, t in enumerate(trits):
        if t not in (0, 1, 2):
            raise ValueError(f"Invalid trit {t} at position {i}; expected 0, 1, or 2")
        pattern_id += int(t) * POWERS_OF_3[i]
    return pattern_id


def decode_pattern(pattern_id: int) -> Tuple[int, int, int, int, int]:
    """Decode a pattern_id in [0, 242] into a 5-tuple of trits (t0, t1, t2, t3, t4)."""
    if not (0 <= int(pattern_id) < NUM_PATTERNS):
        raise ValueError(f"pattern_id must be in [0, 242], got {pattern_id}")
    val = int(pattern_id)
    t0 = val % 3
    val //= 3
    t1 = val % 3
    val //= 3
    t2 = val % 3
    val //= 3
    t3 = val % 3
    t4 = val // 3
    return (t0, t1, t2, t3, t4)


def compute_feedback_trits_scalar(guess: str, answer: str) -> Tuple[int, int, int, int, int]:
    """Two-pass scalar reference implementation of Wordle duplicate-letter feedback rules."""
    if len(guess) != 5 or len(answer) != 5:
        raise ValueError(f"Both guess ({guess!r}) and answer ({answer!r}) must be 5-letter strings")

    trits = [GRAY, GRAY, GRAY, GRAY, GRAY]
    unmatched_counts = [0] * 26

    # Pass 1: Identify exact positional matches (GREEN) and count unmatched answer letters.
    for i in range(5):
        g_ch = guess[i]
        a_ch = answer[i]
        if g_ch == a_ch:
            trits[i] = GREEN
        else:
            idx = ord(a_ch) - 97
            if not (0 <= idx < 26):
                raise ValueError(f"Non-lowercase ASCII character {a_ch!r} in answer {answer!r}")
            unmatched_counts[idx] += 1

    # Pass 2: Left-to-right assignment of YELLOW vs GRAY for non-green positions.
    for i in range(5):
        if trits[i] == GREEN:
            continue
        g_ch = guess[i]
        idx = ord(g_ch) - 97
        if not (0 <= idx < 26):
            raise ValueError(f"Non-lowercase ASCII character {g_ch!r} in guess {guess!r}")
        if unmatched_counts[idx] > 0:
            trits[i] = YELLOW
            unmatched_counts[idx] -= 1

    return (trits[0], trits[1], trits[2], trits[3], trits[4])


def compute_feedback_scalar(guess: str, answer: str) -> int:
    """Compute base-3 encoded pattern_id in [0, 242] using the two-pass scalar reference."""
    t0, t1, t2, t3, t4 = compute_feedback_trits_scalar(guess, answer)
    return t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4


def format_pattern_ascii(pattern_id: int) -> str:
    """Format a pattern_id as a 5-character ASCII string ('G'=Green, 'Y'=Yellow, '.'=Gray)."""
    symbols = {GRAY: ".", YELLOW: "Y", GREEN: "G"}
    return "".join(symbols[t] for t in decode_pattern(pattern_id))


def format_guess_ansi(guess: str, pattern_id: int) -> str:
    """Render a 5-letter guess with ANSI background colors matching its feedback pattern."""
    trits = decode_pattern(pattern_id)
    styles = {
        GRAY: "\033[1;37;40m",    # Bold white on dark gray/black
        YELLOW: "\033[1;30;43m",  # Bold black on yellow
        GREEN: "\033[1;30;42m",   # Bold black on green
    }
    reset = "\033[0m"
    tiles = [f"{styles[t]} {ch.upper()} {reset}" for ch, t in zip(guess, trits)]
    return " ".join(tiles)


@dataclass(frozen=True)
class StepResult:
    """Immutable outcome of a single simulator turn."""

    guess: str
    pattern_id: int
    turn_number: int
    turns_remaining: int
    solved: bool
    game_over: bool


class WordleSimulator:
    """Stateful Wordle game simulator that owns the hidden target and enforces turn limits."""

    def __init__(
        self,
        target: str,
        allowed_guesses_set: Optional[Set[str]] = None,
        max_turns: int = MAX_TURNS,
    ) -> None:
        cleaned_target = target.strip().lower()
        if len(cleaned_target) != 5 or not cleaned_target.isalpha() or not cleaned_target.isascii():
            raise ValueError(f"Invalid 5-letter target word: {target!r}")
        if allowed_guesses_set is not None and cleaned_target not in allowed_guesses_set:
            raise ValueError(f"Target word {cleaned_target!r} is not in the allowed dictionary")

        self._target: str = cleaned_target
        self._allowed_guesses_set: Optional[Set[str]] = allowed_guesses_set
        self.max_turns: int = int(max_turns)
        self.turns_used: int = 0
        self.solved: bool = False
        self.history: List[StepResult] = []

    @property
    def turns_remaining(self) -> int:
        return self.max_turns - self.turns_used

    @property
    def game_over(self) -> bool:
        return self.solved or (self.turns_used >= self.max_turns)

    def step(self, guess: str) -> StepResult:
        """Validate a guess, advance turn state, and return feedback without leaking target."""
        if self.game_over:
            raise RuntimeError(
                f"Cannot call step() after game has already ended "
                f"(solved={self.solved}, turns_used={self.turns_used}/{self.max_turns})"
            )

        cleaned_guess = guess.strip().lower()
        if len(cleaned_guess) != 5 or not cleaned_guess.isalpha() or not cleaned_guess.isascii():
            raise ValueError(f"Guess {guess!r} must be a 5-letter ASCII word")

        if self._allowed_guesses_set is not None and cleaned_guess not in self._allowed_guesses_set:
            raise ValueError(f"Guess {cleaned_guess!r} is outside the allowed dictionary")

        pattern_id = compute_feedback_scalar(cleaned_guess, self._target)
        self.turns_used += 1
        if pattern_id == ALL_GREEN:
            self.solved = True

        result = StepResult(
            guess=cleaned_guess,
            pattern_id=pattern_id,
            turn_number=self.turns_used,
            turns_remaining=self.turns_remaining,
            solved=self.solved,
            game_over=self.game_over,
        )
        self.history.append(result)
        return result

