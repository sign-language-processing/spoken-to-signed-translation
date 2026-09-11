# Text-to-Gloss

([Background](https://research.sign.mt/#text-to-gloss))


Each component implements `text_to_gloss`; reorder-only components can also accept pre-tokenized input:

```python
def text_to_gloss(text: str, language: str) -> List[Gloss]: ...


def tokens_to_gloss(tokens: Gloss, language: str, signed_language: str) -> List[Gloss]: ...
```

Both return sentences of `(word, gloss)` items. Token-based components return original item objects, without introducing
or duplicating tokens; grammar components may omit items as well as reorder them.

## `asl` token component

`asl.tokens_to_gloss` accepts existing English tokens and aligned `metadata` dictionaries containing `pos` and optionally
`morphology` (a list of feature dictionaries). It drops articles, present-tense copulas and present-tense support “do,”
and moves a leading WH word or short phrase after the remaining words: “what is your name?” → “your name what?”;
“how many books do you have?” → “you have how many books?”. Original items and multiword spans stay intact.
Without metadata it preserves the input. Relative/coordinated clauses and ambiguous multi-predicate sentences remain
in input order. These rules are not a complete ASL grammar; tense/aspect realization, spatial agreement and nonmanuals
need further work.

## `gpt` indexed token component

Install `spoken-to-signed[gpt]`. For a local OpenAI-compatible server, set:

```sh
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=local
export OPENAI_MODEL=openai/gpt-oss-20b
```

`gpt.tokens_to_gloss` accepts the same tokens and metadata. The model returns `{"order": [2, 3, 0, 4]}`:
indexes are zero-based input items, not retokenized words. Omitted indexes are dropped. Existing ASL rules identify
which omissions are allowed; without English/ASL metadata every token is required. Invalid indexes, duplication,
missing required tokens, and moves across sentence boundaries raise `ValueError`. The returned items are the original
objects, so WSD sense/entity links stay aligned. `OPENAI_MODEL` defaults to `gpt-4o-mini`.

This is experimental and opt-in. Weak-model outputs can satisfy the index contract but still have poor word order;
they are not validated ASL. See `benchmarks/token_gloss.py` for a small reproducible diagnostic, not an accuracy benchmark.

## `nmt` component

Using this component means that the spoken language text is translated into a sequence of sign language glosses with
a neural machine translation system.

Currently, the only language pair that is supported is German (DE) and German Sign Language (DGS), but the same system
can also be used to translate between DE and Swiss German Sign Language (DSGS).

We provide a trained model for this that is downloaded automatically from our public file server.

To reproduce the training of this model, follow some of the steps outlined in this repository:
https://github.com/bricksdont/easier-gloss-translation (see this repository for more explanation and documentation):

````bash
git clone https://github.com/bricksdont/easier-gloss-translation
cd easier-gloss-translation
````

````bash
./scripts/setup/create_venv.sh
````

````bash
./scripts/setup/install.sh
````

Then run **one experiment** defined in

````bash
./scripts/running/run_multilingual_models.sh
````

specifically, execute only the part "Multilingual 1: all German and DGS directions" in this script.
