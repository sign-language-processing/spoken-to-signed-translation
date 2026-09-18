from unittest.mock import MagicMock

import pytest

from spoken_to_signed.gloss_to_pose.lookup.lookup import CoverageType, PoseLookup, PoseResult
from spoken_to_signed.gloss_to_pose.lookup.sql_lookup import SQLPoseLookup
from spoken_to_signed.text_to_gloss.types import GlossItem


def test_candidates_are_parameterized_and_match_lookup_forms(monkeypatch):
    lookup = SQLPoseLookup({"dbname": "test"})
    query = MagicMock(return_value=[{"path": "gs://sign-mt-poses/test/sign.pose", "words": "New York"}])
    monkeypatch.setattr(lookup, "query", query)
    rows = lookup.get_initial_candidates(
        [GlossItem("New York", "NEW YORK"), GlossItem(None, "BOOK+"), GlossItem("x'", "x'")],
        "en",
        "ase",
        "source_%'",
    )
    sql, params = query.call_args.args
    assert params[:3] == ("en", ["ase"], "source_%'")
    assert set(params[3]) == {"new york", "newyork", "book+", "book", "x'", "x"}
    assert params[3] == params[4]
    assert "source_%'" not in sql
    assert "x'" not in sql
    assert rows == query.return_value


def test_database_connection_is_readonly_and_closed(monkeypatch):
    import psycopg2

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [{"phrase": "book"}]
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(psycopg2, "connect", connect)
    lookup = SQLPoseLookup({"dsn": "postgresql://localhost/test"})
    assert lookup.query("SELECT %s", ("book",)) == [{"phrase": "book"}]
    connect.assert_called_once_with(dsn="postgresql://localhost/test")
    connection.set_session.assert_called_once_with(readonly=True, autocommit=True)
    cursor.execute.assert_called_once_with("SELECT %s", ("book",))
    connection.close.assert_called_once()
    cursor.execute.side_effect = psycopg2.OperationalError("connection failed")
    with pytest.raises(psycopg2.OperationalError, match="connection failed"):
        lookup.query("SELECT %s", ("book",))
    assert connection.close.call_count == 2


def test_missing_sign_uses_existing_backup_and_reports_coverage(monkeypatch):
    backup = MagicMock(spec=PoseLookup)
    result = PoseResult(pose=object(), coverage=CoverageType.FINGERSPELLING_BACKUP)
    backup.lookup.return_value = result
    lookup = SQLPoseLookup({}, backup=backup)
    monkeypatch.setattr(lookup, "query", MagicMock(return_value=[]))
    assert lookup.lookup_sequence([GlossItem(None, "Amit")], "en", "ase") == [result]
    backup.lookup.assert_called_once_with("Amit", "Amit", "en", "ase", None)
    assert lookup.last_coverage[0].coverage == CoverageType.FINGERSPELLING_BACKUP
    assert lookup.lookup_sequence([], "en", "ase") == []
    assert lookup.last_coverage == []


def test_no_backup_does_not_fingerspell(monkeypatch):
    lookup = SQLPoseLookup({})
    monkeypatch.setattr(lookup, "query", MagicMock(return_value=[]))
    with pytest.raises(Exception, match="No poses found"):
        lookup.lookup_sequence([GlossItem("Amit", "Amit")], "en", "ase")
