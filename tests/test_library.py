import json
import time

from live_lyrics import library as lib
from live_lyrics.library import Library


def test_record_play_appends_and_dedupes_quick_relocks(tmp_path):
    L = Library(str(tmp_path / "lib.json"))
    L.record_play("a|x", "X", "A", cover="http://img")
    L.record_play("a|x", "X", "A")                  # the same song re-locked seconds later: same play
    L.record_play("b|y", "Y", "B")
    plays = lib.read_history(L.history_path)
    assert [p["key"] for p in plays] == ["b|y", "a|x"]   # newest first
    assert plays[1]["cover"] == "http://img"


def test_record_play_counts_a_real_replay(tmp_path):
    L = Library(str(tmp_path / "lib.json"))
    L.record_play("a|x", "X", "A")
    L.last_play = ("a|x", time.time() - lib.REPLAY_GAP - 1)   # played again much later
    L.record_play("a|x", "X", "A")
    assert len(lib.read_history(L.history_path)) == 2


def test_read_history_skips_torn_lines_and_old_plays(tmp_path):
    p = tmp_path / "history.jsonl"
    old = json.dumps({"t": time.time() - 40 * 86400, "key": "old", "title": "O", "artist": "O"})
    good = json.dumps({"t": time.time(), "key": "new", "title": "N", "artist": "N"})
    p.write_text(old + "\n" + good + "\n" + '{"t": 123, "cut off')
    plays = lib.read_history(p)
    assert [q["key"] for q in plays] == ["new"]


def test_read_history_missing_file(tmp_path):
    assert lib.read_history(tmp_path / "nope.jsonl") == []
