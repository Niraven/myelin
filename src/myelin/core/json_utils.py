"""JSON serialization utilities for Myelin.

Provides safe JSON handling for SQLite-backed storage:
- ``json_safe_loads``: deserialize any value (str, None, list, dict, malformed)
  without raising.
- ``deserialize_row``: project a DB row by auto-deserializing a set of known
  JSON text columns while preserving all other fields (arbitrary fields survive
  round-trips through the DB automatically because database.py serialises
  unknown list/dict values to JSON).

Design decisions
----------------
- **Arbitrary field preservation**: The DB ``insert`` and ``update`` helpers in
  ``database.py`` already call ``json.dumps`` on any ``list`` or ``dict`` value.
  When reading back, we only deserialize explicitly declared JSON fields so that
  genuine text fields (e.g. ``content_text``, ``action``) are never accidentally
  deserialized. All other fields pass through as-is — including columns that
  don't exist in the Pydantic model — preserving them for downstream consumers.
- **Null safety**: ``input_context``, ``output_result``, and similar nullable
  JSON fields store as ``NULL`` in SQLite when ``None``.  ``json_safe_loads``
  returns ``None`` for ``NULL``, preserving the round-trip.
- **Empty / malformed**: ``""`` and ``"[]"`` or ``"{}"`` are all valid JSON and
  deserialize correctly; genuinely corrupted data returns ``None`` instead of
  crashing the caller.
"""

from __future__ import annotations

import json
from typing import Any


def json_safe_loads(value: Any) -> Any:
    """Safely deserialise a potentially-JSON value.

    Handles all of the following without raising::

        None                 →  None
        ""                   →  ""             (identity — already a str)
        "[1, 2, 3]"          →  [1, 2, 3]
        "null"               →  None
        "garbage"            →  "garbage"      (identity fallback)
        [1, 2, 3]            →  [1, 2, 3]      (identity — already a list)
        {"a": 1}             →  {"a": 1}        (identity — already a dict)
    """
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value  # already deserialized
    if not isinstance(value, str):
        return value  # int, float, bool — pass through
    stripped = value.strip()
    if not stripped:
        return value  # empty string stays empty string
    # Quick check: is it JSON-shaped?
    if stripped[0] not in ("[", "{", '"') and stripped not in (
        "true",
        "false",
        "null",
    ):
        return value  # plain string, not JSON
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value  # malformed — return the original string


JSON_LIST_FIELDS: set[str] = {
    "access_times",
    "tags",
    "source_ids",
    "steps",
    "preconditions",
    "postconditions",
    "source_episodes",
    "component_procedures",
    "parent_procedures",
    "transferred_to",
    "evidence_episodes",
    "conditions",
    "exceptions",
    "semantic_source_ids",
    "episode_source_ids",
}

JSON_DICT_FIELDS: set[str] = {
    "input_context",
    "output_result",
    "details",
    "adaptation_details",
    "capabilities",
}


def deserialize_row(
    row: dict[str, Any],
    extra_list_fields: set[str] | None = None,
    extra_dict_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Project a DB row by deserializing known JSON text columns in-place.

    Returns the same dict (mutated) for convenience.  Fields not in either
    ``JSON_LIST_FIELDS`` or ``JSON_DICT_FIELDS`` are left untouched — this
    preserves arbitrary / unknown columns through round-trips.
    """
    list_fields = JSON_LIST_FIELDS | (extra_list_fields or set())
    dict_fields = JSON_DICT_FIELDS | (extra_dict_fields or set())

    for field in list_fields:
        if field in row:
            row[field] = json_safe_loads(row[field])

    for field in dict_fields:
        if field in row:
            row[field] = json_safe_loads(row[field])

    return row
