from spoken_to_signed.text_to_gloss.simple import tokens_to_gloss
from spoken_to_signed.text_to_gloss.types import GlossItem


def test_tokens_to_gloss_preserves_caller_tokenization():
    tokens = [GlossItem("visit", "visit"), GlossItem("example.com", "example.com"), GlossItem("AT&T.", "at&t.")]

    result = tokens_to_gloss(tokens, language="en", signed_language="ase")

    assert result == [tokens]
    assert all(actual is original for actual, original in zip(result[0], tokens))
