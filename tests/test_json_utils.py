"""Tests for JSON field optimization utilities.

Covers:
- ``json_safe_loads``: null, empty, malformed, already-deserialized,
  regular strings, and valid JSON.
- ``deserialize_row``: known JSON fields, arbitrary field preservation,
  extra fields, safety with missing keys.
- Backward compatibility: plain strings that look like JSON are deserialized;
  genuine text fields without JSON shape are left untouched.
"""

from myelin.core.json_utils import (
    JSON_DICT_FIELDS,
    JSON_LIST_FIELDS,
    deserialize_row,
    json_safe_loads,
)


class TestJsonSafeLoads:
    """Tests for the safe JSON loader."""

    def test_none(self):
        assert json_safe_loads(None) is None

    def test_empty_string(self):
        assert json_safe_loads("") == ""

    def test_whitespace_string(self):
        assert json_safe_loads("   ") == "   "

    def test_already_list(self):
        assert json_safe_loads([1, 2, 3]) == [1, 2, 3]

    def test_already_dict(self):
        assert json_safe_loads({"key": "val"}) == {"key": "val"}

    def test_valid_json_array(self):
        assert json_safe_loads("[1, 2, 3]") == [1, 2, 3]

    def test_valid_json_object(self):
        assert json_safe_loads('{"a": 1}') == {"a": 1}

    def test_json_null_literal(self):
        assert json_safe_loads("null") is None

    def test_json_true(self):
        assert json_safe_loads("true") is True

    def test_json_false(self):
        assert json_safe_loads("false") is False

    def test_malformed_json(self):
        """Malformed JSON returns the original string."""
        result = json_safe_loads("{bad json}")
        assert result == "{bad json}"

    def test_partial_malformed(self):
        """Truncated JSON returns the original string."""
        result = json_safe_loads('{"key": "val')
        assert result == '{"key": "val'

    def test_plain_string(self):
        """Non-JSON-shaped strings pass through unchanged."""
        result = json_safe_loads("hello world")
        assert result == "hello world"

    def test_number(self):
        assert json_safe_loads(42) == 42

    def test_float(self):
        assert json_safe_loads(3.14) == 3.14

    def test_bool(self):
        assert json_safe_loads(False) is False

    def test_empty_array(self):
        """Empty JSON array deserializes correctly."""
        assert json_safe_loads("[]") == []

    def test_empty_object(self):
        """Empty JSON object deserializes correctly."""
        assert json_safe_loads("{}") == {}

    def test_nested_json(self):
        assert json_safe_loads('{"items": [1, 2], "nested": {"a": 1}}') == {
            "items": [1, 2],
            "nested": {"a": 1},
        }


class TestDeserializeRow:
    """Tests for row projection with auto-deserialization."""

    def test_deserialize_list_fields(self):
        row = {
            "id": "abc",
            "tags": '["dev", "test"]',
            "access_times": "[1.0, 2.0]",
            "content_text": "plain text",
        }
        result = deserialize_row(row)
        assert result["tags"] == ["dev", "test"]
        assert result["access_times"] == [1.0, 2.0]
        assert result["content_text"] == "plain text"  # unchanged
        assert result["id"] == "abc"  # unchanged

    def test_deserialize_dict_fields(self):
        row = {
            "id": "abc",
            "input_context": '{"tool": "git", "args": ["pull"]}',
            "output_result": '{"exit_code": 0}',
        }
        result = deserialize_row(row)
        assert result["input_context"] == {"tool": "git", "args": ["pull"]}
        assert result["output_result"] == {"exit_code": 0}

    def test_null_fields_stay_none(self):
        """NULL fields from SQLite should remain None."""
        row = {"input_context": None, "output_result": None, "tags": None}
        result = deserialize_row(row)
        assert result["input_context"] is None
        assert result["output_result"] is None
        assert result["tags"] is None

    def test_already_deserialized(self):
        """Already-deserialized fields should not be re-processed."""
        row = {
            "tags": ["dev", "test"],
            "access_times": [1.0, 2.0],
            "input_context": {"tool": "git"},
        }
        result = deserialize_row(row)
        assert result["tags"] == ["dev", "test"]
        assert result["access_times"] == [1.0, 2.0]
        assert result["input_context"] == {"tool": "git"}

    def test_arbitrary_fields_preserved(self):
        """Unknown fields should pass through unchanged."""
        row = {
            "id": "abc",
            "custom_field": "some value",
            "unknown_json": "[1, 2, 3]",  # not in known sets — stays as string
        }
        result = deserialize_row(row)
        assert result["custom_field"] == "some value"
        # This field is not in JSON_LIST_FIELDS or JSON_DICT_FIELDS
        assert result["unknown_json"] == "[1, 2, 3]"

    def test_missing_fields_no_error(self):
        """Fields not present in the row should not cause errors."""
        row = {"id": "abc"}
        result = deserialize_row(row)
        assert result == {"id": "abc"}

    def test_empty_row(self):
        """An empty dict should produce an empty dict."""
        result = deserialize_row({})
        assert result == {}

    def test_malformed_json_field(self):
        """A malformed JSON string in a known field should return the original string."""
        row = {"tags": "{bad json}"}
        result = deserialize_row(row)
        assert result["tags"] == "{bad json}"

    def test_extra_list_fields(self):
        """Extra list fields provided by the caller should also be deserialized."""
        row = {"custom_list": '["a", "b"]', "tags": '["x"]'}
        result = deserialize_row(row, extra_list_fields={"custom_list"})
        assert result["custom_list"] == ["a", "b"]
        assert result["tags"] == ["x"]

    def test_extra_dict_fields(self):
        """Extra dict fields provided by the caller should also be deserialized."""
        row = {"custom_dict": '{"a": 1}', "input_context": None}
        result = deserialize_row(row, extra_dict_fields={"custom_dict"})
        assert result["custom_dict"] == {"a": 1}
        assert result["input_context"] is None

    def test_backward_compat_plain_string_unchanged(self):
        """Ensure existing non-JSON text fields are not affected."""
        row = {
            "id": "proc-1",
            "name": "Build and Deploy",
            "description": "A multi-step deployment procedure",
            "trigger_pattern": "build and deploy",
            "steps": '[{"order": 1, "description": "Run docker build"}]',
            "source_agent": "agent1",
            "status": "active",
        }
        result = deserialize_row(row)
        assert result["name"] == "Build and Deploy"
        assert result["description"] == "A multi-step deployment procedure"
        assert result["trigger_pattern"] == "build and deploy"
        assert result["source_agent"] == "agent1"
        assert result["status"] == "active"
        assert result["steps"] == [{"order": 1, "description": "Run docker build"}]

    def test_known_field_sets_not_empty(self):
        """The known JSON field sets should contain expected fields."""
        assert "access_times" in JSON_LIST_FIELDS
        assert "tags" in JSON_LIST_FIELDS
        assert "steps" in JSON_LIST_FIELDS
        assert "input_context" in JSON_DICT_FIELDS
        assert "output_result" in JSON_DICT_FIELDS
        assert "details" in JSON_DICT_FIELDS
