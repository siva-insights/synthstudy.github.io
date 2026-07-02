"""Tests for SEDG backend core logic.

Covers: pid sorting, persona sampling, prompt construction,
answer parsing, context-window estimation, and timing history.
No network calls are made; load_dataset and the Ollama API are never invoked.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
from app import (
    _sort_by_pid,
    build_prompt,
    estimate_num_ctx,
    get_average_seconds_per_respondent,
    parse_answers,
    sample_personas,
    Question,
)


# ── _sort_by_pid ──────────────────────────────────────────────────────────────

class TestSortByPid:
    def _df(self, pids):
        return pd.DataFrame({"pid": pids, "persona_summary": ["x"] * len(pids)})

    def test_numeric_ids_sorted_as_integers(self):
        result = _sort_by_pid(self._df(["10", "2", "1"]))
        assert list(result["pid"]) == ["1", "2", "10"]

    def test_alphabetic_ids_sorted_as_strings(self):
        result = _sort_by_pid(self._df(["c", "a", "b"]))
        assert list(result["pid"]) == ["a", "b", "c"]

    def test_index_reset_after_sort(self):
        result = _sort_by_pid(self._df(["3", "1", "2"]))
        assert list(result.index) == [0, 1, 2]

    def test_single_row_unchanged(self):
        result = _sort_by_pid(self._df(["42"]))
        assert list(result["pid"]) == ["42"]


# ── sample_personas ───────────────────────────────────────────────────────────

class TestSamplePersonas:
    def _df(self, n):
        return pd.DataFrame({
            "pid": [str(i) for i in range(1, n + 1)],
            "persona_summary": [f"p{i}" for i in range(1, n + 1)],
        })

    def test_sequential_exact_count(self):
        result = sample_personas(self._df(5), 5, sequential=True)
        assert len(result) == 5
        assert list(result["pid"]) == ["1", "2", "3", "4", "5"]

    def test_sequential_cyclic_wrap(self):
        result = sample_personas(self._df(3), 7, sequential=True)
        assert len(result) == 7
        assert list(result["pid"]) == ["1", "2", "3", "1", "2", "3", "1"]

    def test_sequential_index_reset(self):
        result = sample_personas(self._df(2), 5, sequential=True)
        assert list(result.index) == list(range(5))

    def test_random_returns_correct_count(self):
        result = sample_personas(self._df(10), 6, sequential=False)
        assert len(result) == 6

    def test_random_with_replacement_when_pool_smaller(self):
        result = sample_personas(self._df(3), 10, sequential=False)
        assert len(result) == 10

    def test_random_pids_come_from_source(self):
        source_pids = {"1", "2", "3", "4", "5"}
        result = sample_personas(self._df(5), 5, sequential=False)
        assert set(result["pid"]).issubset(source_pids)


# ── estimate_num_ctx ──────────────────────────────────────────────────────────

class TestEstimateNumCtx:
    def test_returns_integer(self):
        assert isinstance(estimate_num_ctx("hello world"), int)

    def test_known_calculation(self):
        # 4 words: (4 + 500 + 3061) * 2 = 7130
        assert estimate_num_ctx("one two three four") == 7130

    def test_longer_prompt_yields_larger_ctx(self):
        short = estimate_num_ctx("word")
        long = estimate_num_ctx(" ".join(["word"] * 500))
        assert long > short


# ── parse_answers ─────────────────────────────────────────────────────────────

def _q(number, scale_type, scale_points, scale_start=1, max_words=0):
    return Question(
        question_number=number,
        question_text=f"Q{number} text",
        scale_type=scale_type,
        scale_points=scale_points,
        scale_start=scale_start,
        max_words=max_words,
    )


class TestParseAnswers:
    def test_discrete_valid(self):
        q = _q(1, "discrete", ["Low", "Medium", "High"])
        answers, validation, invalid = parse_answers("Q1=2", [q])
        assert answers["Q1"] == 2
        assert validation == "valid"
        assert invalid == []

    def test_discrete_out_of_range_is_invalid(self):
        q = _q(1, "discrete", ["A", "B"])
        _, validation, invalid = parse_answers("Q1=9", [q])
        assert validation == "invalid"
        assert "Q1" in invalid

    def test_discrete_missing_is_invalid(self):
        q = _q(1, "discrete", ["A", "B"])
        answers, validation, invalid = parse_answers("no answer here", [q])
        assert answers["Q1"] == ""
        assert "Q1" in invalid

    def test_continuous_float_valid(self):
        q = _q(1, "continuous", ["1", "2", "3", "4", "5"])
        answers, validation, _ = parse_answers("Q1=3.5", [q])
        assert answers["Q1"] == 3.5
        assert validation == "valid"

    def test_text_quoted(self):
        q = _q(1, "text", [], max_words=50)
        answers, validation, _ = parse_answers('Q1="Great product"', [q])
        assert answers["Q1"] == "Great product"
        assert validation == "valid"

    def test_boundary_min_and_max_are_valid(self):
        q = _q(1, "discrete", ["A", "B", "C", "D", "E"], scale_start=1)
        _, v1, _ = parse_answers("Q1=1", [q])
        _, v5, _ = parse_answers("Q1=5", [q])
        assert v1 == "valid"
        assert v5 == "valid"

    def test_multiple_questions_partial_invalid(self):
        q1 = _q(1, "discrete", ["Low", "High"])
        q2 = _q(2, "discrete", ["Low", "High"])
        answers, validation, invalid = parse_answers("Q1=1\nQ2=99", [q1, q2])
        assert answers["Q1"] == 1
        assert validation == "invalid"
        assert "Q2" in invalid
        assert "Q1" not in invalid


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def _questions(self):
        return [_q(1, "discrete", ["Low", "Medium", "High"])]

    def test_persona_appears_in_prompt(self):
        prompt = build_prompt("Experienced teacher", "Rate this {Q1}", self._questions())
        assert "Experienced teacher" in prompt

    def test_no_persona_excludes_persona_section(self):
        prompt = build_prompt("", "Rate {Q1}", self._questions(), include_persona=False)
        assert "Respondent persona" not in prompt

    def test_question_placeholder_replaced(self):
        prompt = build_prompt("persona", "Intro {Q1} outro", self._questions())
        assert "{Q1}" not in prompt
        assert "Q1" in prompt

    def test_answer_template_in_prompt(self):
        prompt = build_prompt("persona", "Rate {Q1}", self._questions())
        assert "Q1=?" in prompt

    def test_custom_instruction_replaces_default(self):
        custom = "Custom: {embedded_stimuli} / {answer_template}"
        prompt = build_prompt("p", "stimuli {Q1}", self._questions(), generic_instruction=custom)
        assert "Custom:" in prompt
        assert "simulating one synthetic survey respondent" not in prompt

    def test_stimuli_content_present(self):
        prompt = build_prompt("p", "Buy this widget. {Q1}", self._questions())
        assert "Buy this widget." in prompt


# ── history and timing average ─────────────────────────────────────────────────

class TestHistoryAverage:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "HISTORY_FILE", tmp_path / "no_file.json")
        assert get_average_seconds_per_respondent("llama3") is None

    def test_empty_history_returns_none(self, tmp_path, monkeypatch):
        hfile = tmp_path / "history.json"
        hfile.write_text("[]")
        monkeypatch.setattr(app, "HISTORY_FILE", hfile)
        assert get_average_seconds_per_respondent("llama3") is None

    def test_average_computed_correctly(self, tmp_path, monkeypatch):
        hfile = tmp_path / "history.json"
        hfile.write_text(json.dumps([
            {"model_name": "llama3", "seconds_taken": 10},
            {"model_name": "llama3", "seconds_taken": 20},
            {"model_name": "other",  "seconds_taken": 99},
        ]))
        monkeypatch.setattr(app, "HISTORY_FILE", hfile)
        assert get_average_seconds_per_respondent("llama3") == 15.0

    def test_unknown_model_returns_none(self, tmp_path, monkeypatch):
        hfile = tmp_path / "history.json"
        hfile.write_text(json.dumps([{"model_name": "llama3", "seconds_taken": 5}]))
        monkeypatch.setattr(app, "HISTORY_FILE", hfile)
        assert get_average_seconds_per_respondent("gpt4") is None
