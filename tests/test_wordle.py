"""Comprehensive unit and property tests for Wordle engine, cache, agents, and invariants."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest

from src.agents import EntropyAgent, LetterFrequencyAgent, RandomConsistentAgent
from src.dictionary import DEFAULT_DATA_DIR, derive_game_seed, load_dictionary
from src.engine import (
    ALL_GREEN,
    GRAY,
    GREEN,
    NUM_PATTERNS,
    WordleSimulator,
    YELLOW,
    compute_feedback_scalar,
    compute_feedback_trits_scalar,
    decode_pattern,
    encode_trits,
)
from src.precompute import (
    CACHE_META_FILENAME,
    compute_entropies,
    compute_entropy_scalar_row,
    is_cache_valid,
    load_or_build_cache,
)


@pytest.fixture(scope="session")
def dictionary():
    return load_dictionary(DEFAULT_DATA_DIR)


@pytest.fixture(scope="session")
def cache(dictionary):
    return load_or_build_cache(dictionary, DEFAULT_DATA_DIR)


def test_duplicate_letter_feedback_examples():
    """Verify independently specified duplicate-letter Wordle feedback examples."""
    # Example 1: Guess ALLEY against APPLE -> Green, Yellow, Gray, Yellow, Gray
    assert compute_feedback_trits_scalar("alley", "apple") == (
        GREEN,
        YELLOW,
        GRAY,
        YELLOW,
        GRAY,
    )

    # Example 2: Green match in Pass 1 consumes the single 'e' in STEAL before Pass 2
    assert compute_feedback_trits_scalar("speed", "steal") == (
        GREEN,
        GRAY,
        GREEN,
        GRAY,
        GRAY,
    )

    # Example 3: Only one 'e' in ABIDE; first 'e' in SPEED gets Yellow, second gets Gray
    assert compute_feedback_trits_scalar("speed", "abide") == (
        GRAY,
        GRAY,
        YELLOW,
        GRAY,
        YELLOW,
    )

    # Example 4: Two 'e's in ERASE; both 'e's in SPEED get Yellow
    assert compute_feedback_trits_scalar("speed", "erase") == (
        YELLOW,
        GRAY,
        YELLOW,
        YELLOW,
        GRAY,
    )

    # Example 5: Triple 'e' in EERIE against ELITE (two 'e's at 0 and 4)
    assert compute_feedback_trits_scalar("eerie", "elite") == (
        GREEN,
        GRAY,
        GRAY,
        YELLOW,
        GREEN,
    )


def test_encoding_decoding_and_identity(dictionary):
    """Verify pattern_id round-trip encoding and identical-word 242 invariant."""
    for pid in range(NUM_PATTERNS):
        trits = decode_pattern(pid)
        assert len(trits) == 5
        assert all(t in (GRAY, YELLOW, GREEN) for t in trits)
        assert encode_trits(trits) == pid

    for word in dictionary.answers[:200]:
        assert compute_feedback_scalar(word, word) == ALL_GREEN


def test_dictionary_and_splits(dictionary):
    """Verify dictionary sizes, sorting, and disjoint Dev/Test target splits."""
    assert dictionary.num_answers == 2315
    assert dictionary.num_guesses == 12972
    assert list(dictionary.answers) == sorted(dictionary.answers)
    assert list(dictionary.allowed_guesses) == sorted(dictionary.allowed_guesses)

    assert len(dictionary.dev_targets) == 463
    assert len(dictionary.test_targets) == 1852
    assert dictionary.dev_set.isdisjoint(dictionary.test_set)
    assert sorted(list(dictionary.dev_targets) + list(dictionary.test_targets)) == list(
        dictionary.answers
    )


def test_batched_matrix_and_opening_entropies_match_scalar_reference(dictionary, cache):
    """Verify cached pattern_matrix and opening_entropies against scalar reference functions."""
    rng = random.Random(12345)
    sample_g = rng.sample(range(dictionary.num_guesses), 60)
    sample_a = rng.sample(range(dictionary.num_answers), 60)

    # Also explicitly include words with repeated letters
    dup_guesses = [
        dictionary.guess_to_idx[w]
        for w in ("alley", "speed", "eerie", "sissy", "mamma", "fluff")
        if w in dictionary.guess_to_idx
    ]
    dup_answers = [
        dictionary.answer_to_idx[w]
        for w in ("apple", "steal", "abide", "erase", "elite", "vivid")
        if w in dictionary.answer_to_idx
    ]

    for g_idx in sample_g + dup_guesses:
        g_word = dictionary.allowed_guesses[g_idx]
        for a_idx in sample_a + dup_answers:
            a_word = dictionary.answers[a_idx]
            expected = compute_feedback_scalar(g_word, a_word)
            actual = int(cache.pattern_matrix[g_idx, a_idx])
            assert actual == expected, f"Mismatch for guess={g_word}, answer={a_word}"

    # Verify diagonal for all 2,315 answers in pattern_matrix is 242 (ALL_GREEN)
    ans_in_g = dictionary.answer_idx_to_guess_idx
    diag = cache.pattern_matrix[ans_in_g, np.arange(dictionary.num_answers)]
    assert np.all(diag == ALL_GREEN)

    # Verify opening entropies match scalar entropy reference
    for g_idx in sample_g[:25]:
        row = cache.pattern_matrix[g_idx, :]
        expected_h = compute_entropy_scalar_row(row)
        actual_h = float(cache.opening_entropies[g_idx])
        assert actual_h == pytest.approx(expected_h, abs=1e-12)


def test_cache_validation_rejects_mismatched_metadata(dictionary, tmp_path: Path):
    """Verify cache validator rejects tampered metadata hashes or encoding versions."""
    meta_path = DEFAULT_DATA_DIR / CACHE_META_FILENAME
    matrix_path = DEFAULT_DATA_DIR / "pattern_matrix.npy"
    entropies_path = DEFAULT_DATA_DIR / "opening_entropies.npy"

    valid, meta = is_cache_valid(dictionary, meta_path, matrix_path, entropies_path)
    assert valid is True
    assert meta is not None

    bad_meta_path = tmp_path / "bad_meta.json"
    tampered = dict(meta)
    tampered["answers_list_sha256"] = "0" * 64
    bad_meta_path.write_text(json.dumps(tampered), encoding="utf-8")

    valid_bad, _ = is_cache_valid(dictionary, bad_meta_path, matrix_path, entropies_path)
    assert valid_bad is False


def test_simulator_validation_and_lifecycle(dictionary):
    """Verify simulator rejects out-of-dictionary guesses and calls after game end."""
    allowed_set = set(dictionary.allowed_guesses)
    sim = WordleSimulator(target="crane", allowed_guesses_set=allowed_set)

    with pytest.raises(ValueError, match="outside the allowed dictionary"):
        sim.step("zzzzz")

    # Solving immediately stops the game
    res = sim.step("crane")
    assert res.solved is True
    assert res.game_over is True
    assert res.pattern_id == ALL_GREEN

    with pytest.raises(RuntimeError, match="already ended"):
        sim.step("slate")

    # 6 wrong guesses end the game in a loss
    sim_loss = WordleSimulator(target="crane", allowed_guesses_set=allowed_set)
    for turn in range(1, 7):
        r = sim_loss.step("aback")
        assert r.turn_number == turn
        assert r.solved is False
        assert r.game_over is (turn == 6)

    with pytest.raises(RuntimeError, match="already ended"):
        sim_loss.step("aback")


def test_candidate_tracking_and_agent_lifecycle_invariants(dictionary, cache):
    """Verify true answer survival, monotonic candidate reduction, reset, and empty-set error."""
    allowed_set = set(dictionary.allowed_guesses)
    agents = [
        RandomConsistentAgent(dictionary, cache, default_seed=7),
        LetterFrequencyAgent(dictionary, cache),
        EntropyAgent(dictionary, cache, epsilon=0.05),
    ]

    # Pick targets from both dev and test splits; every game MUST start with 2,315 candidates
    targets = [dictionary.dev_targets[0], dictionary.test_targets[0], "apple", "vivid"]

    for agent in agents:
        for target in targets:
            target_idx = dictionary.answer_to_idx[target]
            agent.reset(seed=derive_game_seed(0, target))
            assert agent.num_candidates == 2315
            assert len(agent.history) == 0

            sim = WordleSimulator(target=target, allowed_guesses_set=allowed_set)
            prev_count = agent.num_candidates

            while not sim.game_over:
                if agent.num_candidates == 1:
                    sole_answer = dictionary.answers[int(agent.c_idx[0])]
                    guess = agent.select_guess(turns_remaining=sim.turns_remaining)
                    assert guess == sole_answer
                else:
                    guess = agent.select_guess(turns_remaining=sim.turns_remaining)

                step_res = sim.step(guess)
                new_count = agent.observe(guess, step_res.pattern_id)

                assert 1 <= new_count <= prev_count
                assert target_idx in set(map(int, agent.c_idx))
                prev_count = new_count

    # Verify contradictory observation raises diagnostic RuntimeError
    agent = RandomConsistentAgent(dictionary, cache)
    agent.reset(seed=1)
    # Observing ALL_GREEN (242) for "crane" leaves only "crane"; then observing 0 (all gray)
    # for "crane" is impossible and must raise RuntimeError.
    assert agent.observe("crane", ALL_GREEN) == 1
    with pytest.raises(RuntimeError, match="0 candidate answers remain"):
        agent.observe("crane", 0)


def test_tiny_candidate_sets_and_final_turn_rule(dictionary, cache):
    """Verify SIGHT/FIGHT/MIGHT/LIGHT entropy, 2-candidate tie-breaks, and final-turn rule."""
    agent = EntropyAgent(dictionary, cache, epsilon=0.0, allow_non_answers=True)
    agent.reset()

    # Manually set remaining candidates to SIGHT, FIGHT, MIGHT, LIGHT
    four_words = ["fight", "light", "might", "sight"]
    agent.c_idx = np.array([dictionary.answer_to_idx[w] for w in four_words], dtype=np.int32)
    agent.history.append(("dummy", 0))  # Mark as non-opening turn

    # Compute entropy of guessing "sight" vs "filmy" on this 4-word set
    sight_g_idx = dictionary.guess_to_idx["sight"]
    filmy_g_idx = dictionary.guess_to_idx["filmy"]
    sub = cache.pattern_matrix[:, agent.c_idx]
    entropies = compute_entropies(sub)

    # "filmy" separates all 4 into singletons -> 2.0 bits
    assert entropies[filmy_g_idx] == pytest.approx(2.0, abs=1e-12)
    # "sight" yields 1 green and 3 identical patterns -> -(1/4 log2(1/4) + 3/4 log2(3/4)) = 0.811278...
    assert entropies[sight_g_idx] == pytest.approx(0.8112781244591328, abs=1e-12)

    # With turns_remaining = 2 and epsilon = 0.0, agent chooses a 2.0-bit exploratory word
    guess_t2 = agent.select_guess(turns_remaining=2)
    assert entropies[dictionary.guess_to_idx[guess_t2]] == pytest.approx(2.0, abs=1e-12)

    # With turns_remaining = 1 (final turn rule), agent MUST choose from four_words!
    guess_t1 = agent.select_guess(turns_remaining=1)
    assert guess_t1 in four_words

    # With 2 candidates remaining, both have entropy 1.0 bit; tie-break must pick a candidate in C
    two_words = ["fight", "sight"]
    agent.c_idx = np.array([dictionary.answer_to_idx[w] for w in two_words], dtype=np.int32)
    guess_two = agent.select_guess(turns_remaining=3)
    assert guess_two == "fight"


def test_opening_entropy_cache_switching_threshold(dictionary, cache):
    """Verify opening word switches from best non-answer ('soare') to best answer ('raise') at exact delta H."""
    h = cache.opening_entropies
    soare_idx = dictionary.guess_to_idx["soare"]
    raise_idx = dictionary.guess_to_idx["raise"]
    delta_h = float(h[soare_idx] - h[raise_idx])

    # Below threshold: selects 'soare'
    agent_low = EntropyAgent(dictionary, cache, epsilon=delta_h - 1e-6, allow_non_answers=True)
    agent_low.reset()
    assert agent_low.select_guess(turns_remaining=6) == "soare"
    assert agent_low.last_used_opening_cache is True

    # Above threshold: selects 'raise'
    agent_high = EntropyAgent(dictionary, cache, epsilon=delta_h + 1e-6, allow_non_answers=True)
    agent_high.reset()
    assert agent_high.select_guess(turns_remaining=6) == "raise"
    assert agent_high.last_used_opening_cache is True

    # Candidate-only mode (allow_non_answers=False): always selects 'raise' on Turn 1
    agent_cand = EntropyAgent(dictionary, cache, epsilon=0.0, allow_non_answers=False)
    agent_cand.reset()
    assert agent_cand.select_guess(turns_remaining=6) == "raise"


def test_evaluation_seed_invariance_and_trajectory_conventions(dictionary, cache):
    """Verify per-game seed determinism across evaluation subsets and trajectory aggregation rules."""
    from src.evaluate import play_single_game, summarize_game_records

    allowed_set = set(dictionary.allowed_guesses)
    rand_agent = RandomConsistentAgent(dictionary, cache)

    # Playing 'crane' standalone vs after other games with the same run_seed produces identical guesses
    rec1 = play_single_game(rand_agent, "crane", dictionary, allowed_set, run_seed=4)
    _ = play_single_game(rand_agent, "apple", dictionary, allowed_set, run_seed=4)
    rec2 = play_single_game(rand_agent, "crane", dictionary, allowed_set, run_seed=4)
    assert rec1.game_seed == rec2.game_seed
    assert rec1.guesses == rec2.guesses
    assert rec1.candidates_after == rec2.candidates_after

    # If all games in a slice solve by turn 3, unsolved_after_turn_mean for turns 3..6 must be None
    ent_agent = EntropyAgent(dictionary, cache, epsilon=0.01)
    rec_fast = play_single_game(ent_agent, "raise", dictionary, allowed_set, run_seed=0)
    assert rec_fast.solved is True and rec_fast.turns_used == 1
    summary = summarize_game_records([rec_fast], split_label="single")
    cf = summary["candidate_trajectory"]["carried_forward_mean"]
    uo = summary["candidate_trajectory"]["unsolved_after_turn_mean"]
    uc = summary["candidate_trajectory"]["unsolved_after_turn_count"]
    for t in range(1, 7):
        assert cf[str(t)] == 1.0
        assert uc[str(t)] == 0
        assert uo[str(t)] is None


def test_parse_feedback_string_and_interactive_assistant(dictionary, cache):
    """Verify flexible feedback parsing, 0-candidate typo guard, undo, and practice mode."""
    from src.assistant import run_interactive_assistant
    from src.engine import format_pattern_ascii, parse_feedback_string

    assert parse_feedback_string("GGGGG") == ALL_GREEN
    assert parse_feedback_string("22222") == ALL_GREEN
    assert parse_feedback_string(".....") == 0
    assert parse_feedback_string("b x 0 - _") == 0
    assert format_pattern_ascii(parse_feedback_string(".Y..G")) == ".Y..G"

    with pytest.raises(ValueError, match="5 positions"):
        parse_feedback_string("GGG")
    with pytest.raises(ValueError, match="Invalid feedback symbol"):
        parse_feedback_string("GGGZ.")

    # 1. Test External Helper mode with:
    #    - Turn 1: press Enter ('') to accept 'raise', enter '.Y..G' (41 candidates)
    #    - Turn 2: type 'undo' -> reverts to Turn 1 (2,315 candidates)!
    #    - Turn 1 again: accept 'raise', enter '.Y..G' (41 candidates)
    #    - Turn 2: enter 'plant', test an impossible feedback ('GGGG.' leaves 0 in both 2,315 and 12,972!),
    #      verify it gets rejected non-destructively, then enter 'YYY..' and solve 'apple'!
    agent = EntropyAgent(dictionary, cache, epsilon=0.01)
    inputs = iter(
        [
            "",        # Turn 1: accept #1 recommendation ('raise')
            ".Y..G",   # Feedback for 'raise' vs 'apple' -> 41 candidates remain
            "undo",    # Undo back to Turn 1 (2,315 candidates)
            "",        # Turn 1 again: accept 'raise'
            ".Y..G",   # Feedback '.Y..G' -> 41 candidates remain
            "plant",   # Turn 2: enter 'plant'
            "GGGYG",   # Mathematically impossible Wordle pattern (4 greens + 1 yellow -> 0 in 12,972!)
            "YYY..",   # Valid feedback for 'plant' vs 'apple' -> 2 candidates ('ample', 'apple')
            "apple",   # Turn 3: enter 'apple'
            "GGGGG",   # Solved!
        ]
    )
    outputs: list[str] = []
    history = run_interactive_assistant(
        agent=agent,
        dictionary=dictionary,
        cache=cache,
        target=None,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
    )
    assert len(history) == 3
    assert [g for g, _ in history] == ["raise", "plant", "apple"]
    full_out = "\n".join(outputs)
    assert "leaves 0 possible answers" in full_out
    assert "Solved in 3/6 guesses!" in full_out

    # 2. Test NYT expanded-dictionary answer ('karat', in 12,972 allowed_guesses but not original 2,315)
    agent_karat = EntropyAgent(dictionary, cache, epsilon=0.01)
    karat_inputs = iter(
        [
            "crane",
            "01100",
            "toils",
            "10000",
            "dumpy",
            "00000",
            "karat",
            "22222",
        ]
    )
    karat_outputs: list[str] = []
    hist_karat = run_interactive_assistant(
        agent=agent_karat,
        dictionary=dictionary,
        cache=cache,
        target=None,
        input_fn=lambda prompt: next(karat_inputs),
        output_fn=karat_outputs.append,
    )
    assert [g for g, _ in hist_karat] == ["crane", "toils", "dumpy", "karat"]
    karat_text = "\n".join(karat_outputs)
    assert "ATTAR, GARTH, JARTA, KARAT, RABAT, RATHA" in karat_text
    assert "Solved in 4/6 guesses! (KARAT)" in karat_text

    # 2. Test Practice Mode (target='watch') with Enter shortcut + custom guess comparison
    agent_practice = EntropyAgent(dictionary, cache, epsilon=0.01)
    practice_inputs = iter(["crane", "", "", "", ""])
    practice_outputs: list[str] = []
    hist_practice = run_interactive_assistant(
        agent=agent_practice,
        dictionary=dictionary,
        cache=cache,
        target="watch",
        input_fn=lambda prompt: next(practice_inputs),
        output_fn=practice_outputs.append,
    )
    assert hist_practice[-1][0] == "watch"
    assert hist_practice[-1][1] == ALL_GREEN
    assert any("Comparison: Your guess CRANE" in line for line in practice_outputs)


