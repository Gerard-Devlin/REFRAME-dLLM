import ast
import json
from pathlib import Path

import pytest

from fastv_dllm.score_math import extract_metric_ast, read_records, score_record, summarize


def fixture_source():
    return '''
raise AssertionError("must not execute module body")
import nonexistent_optional_math_verify
SUBSTITUTIONS = [(" ", "")]
REMOVED_EXPRESSIONS = ["units"]
class timeout:
    pass
def last_boxed_only_string(value):
    return value
def remove_boxed(value):
    return value
def is_equiv(first, second):
    return first == second
def get_unnormalized_answer(value):
    return value
def normalize_final_answer(value):
    return value.replace(" ", "")
def unrelated_side_effect():
    raise AssertionError("not a metric dependency")
'''


def test_ast_loads_only_official_primary_definitions():
    tree = extract_metric_ast(fixture_source())
    namespace = {}
    exec(compile(tree, "fixture", "exec"), namespace)
    assert namespace["is_equiv"]("1", "1")
    assert "unrelated_side_effect" not in namespace
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)


def test_reject_changed_or_missing_official_definitions():
    with pytest.raises(ValueError, match="missing"):
        extract_metric_ast(fixture_source().replace("def is_equiv", "def renamed_equiv"))
    with pytest.raises(ValueError, match="Duplicate"):
        extract_metric_ast(fixture_source() + "\nSUBSTITUTIONS = []\n")
    with pytest.raises((ValueError, TypeError)):
        extract_metric_ast(fixture_source().replace('REMOVED_EXPRESSIONS = ["units"]',
                                                  'REMOVED_EXPRESSIONS = dangerous()'))


def test_gold_rebuilt_and_correctness_not_reused():
    namespace = {}
    exec(compile(extract_metric_ast(fixture_source()), "fixture", "exec"), namespace)
    record = {"index": 3, "id": "algebra:3", "target": "stale gold", "flash_native": {
        "text": "1 2", "correct": False, "prediction": "stale prediction", "seconds": 1.0}}
    original = json.dumps(record)
    result = score_record(record, {"solution": "12", "answer": "stale gold"}, namespace)
    assert result["gold"] == "12"
    assert result["methods"]["flash_native"]["exact_match"]
    assert json.dumps(record) == original
    assert summarize([result])["flash_native"]["changed_from_provisional"] == 1


def test_equal_strings_still_use_official_symbolic_decision():
    namespace = {}
    exec(compile(extract_metric_ast(fixture_source()), "fixture", "exec"), namespace)
    seen = []
    def symbolic(first, second):
        seen.append((first, second))
        return False
    namespace["is_equiv"] = symbolic
    result = score_record({"id": "a", "index": 0, "flash_native": {"text": "not math"}},
                          {"solution": "not math"}, namespace)
    assert seen == [("notmath", "notmath")]
    assert not result["methods"]["flash_native"]["exact_match"]


def test_math_verify_secondary_does_not_replace_strict_primary():
    namespace = {}
    exec(compile(extract_metric_ast(fixture_source()), "fixture", "exec"), namespace)
    namespace["get_unnormalized_answer"] = lambda value: "[invalidanswer]"
    seen = []
    def parse(value):
        seen.append(value)
        return value
    namespace["math_verify_functions"] = (parse, lambda gold, generated: True)
    text = r"The answer is \boxed{42}."
    result = score_record({"id": "a", "index": 0, "flash_native": {"text": text}},
                          {"solution": "42"}, namespace)
    assert seen == ["42", text]
    assert not result["methods"]["flash_native"]["exact_match"]
    assert result["methods"]["flash_native"]["math_verify"]
    metrics = summarize([result])["flash_native"]
    assert metrics["exact_match"] == 0.0 and metrics["math_verify"] == 1.0
    assert metrics["valid_extraction_rate"] == 0.0


def test_duplicate_and_unknown_generation_ids_rejected(tmp_path):
    rows = [{"index": 0, "id": "a"}, {"index": 1, "id": "a"}]
    file = tmp_path / "rank_0.jsonl"
    file.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="Duplicate"):
        read_records(tmp_path, {"a": {}})
    file.write_text(json.dumps({"index": 0, "id": "unrecognized"}))
    with pytest.raises(ValueError, match="Unknown"):
        read_records(tmp_path, {"a": {}})


def test_shards_read_in_sample_order_without_modification(tmp_path):
    for rank, indices in enumerate(([0, 2], [1, 3])):
        (tmp_path / f"rank_{rank}.jsonl").write_text("\n".join(
            json.dumps({"index": index, "id": str(index)}) for index in indices))
    original = {file.name: file.read_bytes() for file in tmp_path.glob("rank_*")}
    rows, files = read_records(tmp_path, {str(index): {} for index in range(4)})
    assert [row["index"] for row in rows] == list(range(4))
    assert {file.name: file.read_bytes() for file in files} == original


def test_unequal_method_sets_rejected():
    with pytest.raises(ValueError, match="unequal"):
        summarize([{"methods": {"a": {}}}, {"methods": {"b": {}}}])


def test_official_source_when_available():
    """Integration: require real installed extraction/normalization semantics."""
    from fastv_dllm.score_math import installed_utils
    try:
        path = installed_utils()
    except RuntimeError:
        pytest.skip("lm_eval is not installed locally")
    if not path.is_file():
        pytest.skip("installed lm_eval has no Minerva task")
    import re
    from typing import Optional
    namespace = {"re": re, "Optional": Optional}
    exec(compile(extract_metric_ast(path.read_text(encoding="utf-8")), str(path), "exec"), namespace)
    extract = namespace["get_unnormalized_answer"]
    normalize = namespace["normalize_final_answer"]
    # Official Minerva requires the final-answer phrase. Never introduce a
    # boxed-answer fallback and silently call it the same metric.
    assert extract(r"The solution is \boxed{42}.") == "[invalidanswer]"
    assert normalize(extract("Final Answer: The final answer is $42$. I hope it is correct.")) == "42"
    assert normalize(r"\frac12") == r"\frac{1}{2}"
    assert normalize(r"3\text{ cm}") == "3"
