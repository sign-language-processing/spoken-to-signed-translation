import sys

import pytest

from spoken_to_signed import bin
from spoken_to_signed.text_to_gloss.simple import tokens_to_gloss
from spoken_to_signed.text_to_gloss.types import GlossItem


def test_tokens_to_gloss_preserves_caller_tokenization():
    tokens = [GlossItem("visit", "visit"), GlossItem("example.com", "example.com"), GlossItem("AT&T.", "at&t.")]

    result = tokens_to_gloss(tokens, language="en", signed_language="ase")

    assert result == [tokens]
    assert all(actual is original for actual, original in zip(result[0], tokens))


@pytest.mark.parametrize("glosser", ["rules", "gpt"])
def test_cli_passes_the_target_language(monkeypatch, glosser):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "text_to_gloss",
            "--text",
            "Hello",
            "--glosser",
            glosser,
            "--spoken-language",
            "en",
            "--signed-language",
            "ase",
        ],
    )
    calls = []
    monkeypatch.setattr(bin, "_text_to_gloss", lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    bin.text_to_gloss()
    assert calls == [(("Hello", "en", glosser), {"signed_language": "ase"})]
