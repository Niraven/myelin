"""Tests for Database FTS5 query hardening and SQL injection guards."""

import pytest

from myelin.core.database import (
    Database,
    _contains_injection,
    _normalize_fts_token,
    _tokenize_fts_query,
    build_fts_where,
    escape_fts_query,
    validate_where_clause,
)


class TestFtsTokenNormalization:
    def test_normalize_basic_token(self):
        assert _normalize_fts_token("hello") == '"hello"'

    def test_normalize_unicode_nfkc(self):
        result = _normalize_fts_token("\uff34\uff28\uff25")  # fullwidth THE
        assert "THE" in result or "\uff34" not in result

    def test_normalize_quotes_escaped(self):
        result = _normalize_fts_token('say"hello')
        assert 'say""hello' in result

    def test_normalize_special_chars_preserved(self):
        assert _normalize_fts_token("hello-world") == '"hello-world"'


class TestInjectionDetection:
    @pytest.mark.parametrize("token", ["DROP", "DELETE", "UPDATE", "UNION", "CREATE", "xDROP"])
    def test_accepts_sql_words_as_literal_search_terms(self, token):
        assert _contains_injection(token) is False

    def test_rejects_semicolon(self):
        assert _contains_injection(";") is True

    def test_rejects_sql_comment(self):
        assert _contains_injection("--") is True

    def test_rejects_block_comment(self):
        assert _contains_injection("/*") is True

    def test_accepts_normal_query(self):
        assert _contains_injection("hello") is False

    def test_accepts_mixed_case(self):
        assert _contains_injection("download") is False


class TestTokenizeFtsQuery:
    def test_simple_query(self):
        assert _tokenize_fts_query("hello world") == ["hello", "world"]

    def test_short_tokens_dropped(self):
        assert _tokenize_fts_query("a bc def") == ["def"]

    def test_short_token_only(self):
        assert _tokenize_fts_query("a") == ["a"]

    def test_double_quotes(self):
        tokens = _tokenize_fts_query('hello "world"')
        assert "hello" in tokens

    def test_special_chars(self):
        tokens = _tokenize_fts_query("hello-world test_api")
        assert "hello-world" in tokens
        assert "test_api" in tokens

    def test_unicode_tokens(self):
        tokens = _tokenize_fts_query("café résumé")
        assert any("caf" in t for t in tokens)
        assert any("rés" in t for t in tokens)

    def test_tokens_not_yet_quoted(self):
        tokens = _tokenize_fts_query("hello world")
        for t in tokens:
            assert not t.startswith('"')
            assert not t.endswith('"')

    def test_empty_query_returns_empty_list(self):
        assert _tokenize_fts_query("") == []

    def test_punctuation_only(self):
        assert _tokenize_fts_query("!@#$%^&*()") == []


class TestBuildFtsWhere:
    def test_or_query(self):
        result = build_fts_where(["hello", "world"], operator="OR")
        assert result == '"hello" OR "world"'

    def test_and_query(self):
        result = build_fts_where(["hello", "world"], operator="AND")
        assert result == '"hello" AND "world"'

    def test_empty_tokens_returns_sentinel(self):
        assert build_fts_where([]) == '"__myelin_no_match__"'

    def test_single_token(self):
        assert build_fts_where(["hello"]) == '"hello"'

    def test_accepts_sql_keyword_as_literal_token(self):
        assert build_fts_where(["hello", "DROP"]) == '"hello" OR "DROP"'

    def test_rejects_injection_semicolon(self):
        with pytest.raises(ValueError, match="SQL injection"):
            build_fts_where([";"])

    def test_rejects_overflow(self):
        long_token = "a" * 250
        with pytest.raises(ValueError, match="too long"):
            build_fts_where([long_token])

    def test_max_len_custom(self):
        result = build_fts_where(["hello"], max_len=10)
        assert len(result) <= 10

    def test_unicode_space_is_normalized_inside_literal_token(self):
        assert build_fts_where(["DROP\u2000TABLE"]) == '"DROP TABLE"'


class TestEscapeFtsQuery:
    def test_preserves_backward_compat(self):
        result = escape_fts_query("hello world")
        assert '"hello"' in result and '"world"' in result

    def test_short_tokens_dropped(self):
        result = escape_fts_query("a bb ccc")
        assert '"ccc"' in result
        assert '"a"' not in result
        assert '"bb"' not in result

    def test_empty_returns_sentinel(self):
        assert escape_fts_query("") == '"__myelin_no_match__"'

    def test_punctuation_only_returns_sentinel(self):
        assert escape_fts_query("!@#") == '"__myelin_no_match__"'

    def test_sql_text_is_quoted_as_literal_search_terms(self):
        assert escape_fts_query("hello; DROP TABLE") == '"hello" OR "DROP" OR "TABLE"'

    def test_rejects_long_query(self):
        with pytest.raises(ValueError):
            escape_fts_query("word " * 50)

    def test_special_chars(self):
        result = escape_fts_query("test_api v2.0")
        assert '"test_api"' in result


class TestValidateWhereClause:
    def test_none_passes(self):
        validate_where_clause(None)

    def test_empty_passes(self):
        validate_where_clause("")

    def test_clean_where_passes(self):
        validate_where_clause("domain = ?")

    def test_rejects_semicolon(self):
        with pytest.raises(ValueError, match=";"):
            validate_where_clause("; DROP TABLE")

    def test_rejects_sql_comment(self):
        with pytest.raises(ValueError, match="--"):
            validate_where_clause("-- DROP TABLE")

    def test_rejects_drop_keyword(self):
        with pytest.raises(ValueError, match="DROP"):
            validate_where_clause("domain = ? DROP TABLE")

    def test_rejects_delete_keyword(self):
        with pytest.raises(ValueError, match="DELETE"):
            validate_where_clause("DELETE FROM episodes")


class TestFtsSearchIntegration:
    @pytest.fixture
    def db(self, tmp_path):
        d = Database(str(tmp_path / "test.db"))
        d.conn.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        d.conn.execute("CREATE TABLE test_data (id TEXT PRIMARY KEY, content TEXT)")
        d.conn.execute("INSERT INTO test_data VALUES ('e1', 'hello world')")
        d.conn.execute("INSERT INTO test_fts(rowid, content) VALUES (1, 'hello world')")
        d.conn.execute("INSERT INTO test_data VALUES ('e2', 'foo bar')")
        d.conn.execute("INSERT INTO test_fts(rowid, content) VALUES (2, 'foo bar')")
        d.commit()
        return d

    def test_valid_fts_search(self, db):
        results = db.fts_search("test_data", "test_fts", "hello")
        assert len(results) == 1

    def test_fts_search_with_where(self, db):
        results = db.fts_search("test_data", "test_fts", "hello", where="t.id = 'e1'")
        assert len(results) == 1

    def test_fts_search_rejects_injection_where(self, db):
        with pytest.raises(ValueError):
            db.fts_search("test_data", "test_fts", "hello", where="1=1; DROP TABLE test_data")

    def test_fts_search_treats_sql_keywords_as_data(self, db):
        assert db.fts_search("test_data", "test_fts", "create project note") == []
        assert db.fetchone("SELECT COUNT(*) AS count FROM test_data")["count"] == 2
