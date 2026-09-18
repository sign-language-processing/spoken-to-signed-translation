import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spoken_to_signed.text_to_gloss import gpt
from spoken_to_signed.text_to_gloss.types import GlossItem


@pytest.fixture
def client(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(gpt, "get_openai_client", lambda: mock)
    return mock


def respond(client, payload):
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def question():
    tokens = [GlossItem(word, word) for word in ["what", "is", "your", "name", "?"]]
    return tokens, [{"pos": pos} for pos in ["PRON", "AUX", "PRON", "NOUN", "PUNCT"]]


@pytest.mark.parametrize("pretokenized", [True, False])
def test_luna_request_parameters(client, monkeypatch, pretokenized):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    if pretokenized:
        respond(client, {"order": [2, 3, 0, 4]})
        tokens, metadata = question()
        gpt.tokens_to_gloss(tokens, "en", "ase", metadata=metadata)
    else:
        respond(client, ["YOUR/your NAME/name WHAT/What"])
        assert [token.gloss for token in gpt.text_to_gloss("What is your name?", "en", "ase")[0]] == [
            "YOUR",
            "NAME",
            "WHAT",
        ]
    call = client.chat.completions.create.call_args.kwargs
    assert call["model"] == "gpt-5.6-luna"
    assert call["reasoning_effort"] == "none"
    assert call["max_completion_tokens"] == (1024 if pretokenized else 500)
    assert "max_tokens" not in call
    assert "temperature" not in call


def test_index_selection_preserves_identity_and_sends_optional_indexes(client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-20b")
    tokens, metadata = question()
    respond(client, {"order": [2, 3, 0, 4]})
    [result] = gpt.tokens_to_gloss(tokens, "en", "ase", metadata=metadata)
    assert all(actual is tokens[i] for actual, i in zip(result, [2, 3, 0, 4]))
    call = client.chat.completions.create.call_args.kwargs
    assert call["model"] == "openai/gpt-oss-20b"
    assert json.loads(call["messages"][-1]["content"])["optional_indexes"] == [1]


def test_token_prompt_demonstrates_reordering_and_dropping_before_actual_input(client):
    tokens = [GlossItem("Hello", "hello"), GlossItem("!", "!")]
    respond(client, {"order": [0, 1]})
    gpt.tokens_to_gloss(tokens, "en", "ase")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    expected = [["your", "name", "What", "?"], ["Yesterday", "she", "bought", "red", "car", "in", "London", "."]]
    for offset, words in zip([1, 3], expected):
        example = json.loads(messages[offset]["content"])
        order = json.loads(messages[offset + 1]["content"])["order"]
        example_tokens = [GlossItem(t["word"], t["gloss"]) for t in example["tokens"]]
        required = set(range(len(example_tokens))) - set(example["optional_indexes"])
        [selected] = gpt._select_tokens(example_tokens, order, required)
        assert [token.word for token in selected] == words
    actual = json.loads(messages[-1]["content"])
    assert [token["word"] for token in actual["tokens"]] == ["Hello", "!"]
    assert actual["optional_indexes"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"order": [2, 3, False, 4]},
        {"order": [2, 3, 0.0, 4]},
        {"order": [2, 3, "0", 4]},
        {"order": [2, 3, -1, 4]},
        {"order": [2, 3, 5, 4]},
        {"order": [2, 3, 0, 0, 4]},
        {"order": [3, 0, 4]},
        {"order": []},
        {"order": "2,3,0,4"},
        {},
        [2, 3, 0, 4],
        None,
    ],
)
def test_rejects_invalid_or_meaning_losing_indexes(client, payload):
    tokens, metadata = question()
    respond(client, payload)
    with pytest.raises(ValueError, match="unique token indexes"):
        gpt.tokens_to_gloss(tokens, "en", "ase", metadata=metadata)


def test_without_metadata_no_omissions_are_authorized(client):
    tokens, _ = question()
    respond(client, {"order": [2, 3, 0, 4]})
    with pytest.raises(ValueError, match="non-optional"):
        gpt.tokens_to_gloss(tokens, "en", "ase")


@pytest.mark.parametrize("order", [[2, 3, 0, 1], [1, 0, 2, 3]])
def test_rejects_cross_sentence_reordering_and_moved_punctuation(client, order):
    tokens = [GlossItem(word, word) for word in ["hello", ".", "bye", "."]]
    respond(client, {"order": order})
    with pytest.raises(ValueError, match="boundaries|punctuation"):
        gpt.tokens_to_gloss(tokens, "en", "ase")


def test_multiple_sentences_remain_separate(client):
    tokens = [GlossItem(word, word) for word in ["hello", ".", "bye", "."]]
    respond(client, {"order": [0, 1, 2, 3]})
    assert gpt.tokens_to_gloss(tokens, "en", "ase") == [tokens[:2], tokens[2:]]


def test_empty_input_does_not_call_model_and_metadata_is_checked(client):
    assert gpt.tokens_to_gloss([], "en", "ase") == [[]]
    with pytest.raises(ValueError, match="one entry"):
        gpt.tokens_to_gloss(question()[0], "en", "ase", metadata=[])
    with pytest.raises(ValueError, match="one entry"):
        gpt.tokens_to_gloss([], "en", "ase", metadata=[{}])
    client.chat.completions.create.assert_not_called()
