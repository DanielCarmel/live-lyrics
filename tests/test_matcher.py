from live_lyrics import matcher
from live_lyrics.matcher import SongText


def test_norm_strips_accents_case_punctuation():
    assert matcher.norm("Héllo, WORLD!") == "hello world"
    assert matcher.norm("don't") == "dont"
    assert matcher.norm("a  b\tc") == "a b c"


def test_songtext_from_synced_and_plain():
    st = SongText.from_lyrics({"synced": [{"t": 0, "text": "Hello world"}, {"t": 4, "text": "Second line"}]})
    assert st.times == [0, 4] and st.words == ["hello", "world", "second", "line"]
    assert st.word_line == [0, 0, 1, 1]
    st = SongText.from_lyrics({"plain": ["One two", "three"]})
    assert st.times is None and st.words == ["one", "two", "three"]
    assert SongText.from_lyrics(None) is None
    assert SongText.from_lyrics({"track_duration": 200}) is None


LYR = {"plain": ["the quick brown fox jumps over the lazy dog tonight",
                 "a completely different second verse appears right here",
                 "and the chorus rises high above the silent crowd"]}


def test_identify_picks_the_right_song():
    a, b = SongText.from_lyrics(LYR), SongText.from_lyrics(
        {"plain": ["totally unrelated words about sunshine and summer rain falling slowly"]})
    found = matcher.identify("the chorus rises high above the silent crowd", {"a": a, "b": b})
    assert found and found[0] == "a"
    assert matcher.identify("xyzzy", {"a": a, "b": b}) is None  # too few words


def test_locate_finds_position_and_rejects_garbage():
    st = SongText.from_lyrics(LYR)
    loc = matcher.locate(["different", "second", "verse", "appears"], st, None)
    assert loc is not None
    assert st.word_line[loc[0]] == 1
    assert matcher.locate(["qqq", "www", "eee", "rrr"], st, None) is None


def test_script_language():
    assert matcher.script_language("שלום לכולם") == "he"
    assert matcher.script_language("hello there") is None
    assert matcher.script_language("12 34 !") is None
