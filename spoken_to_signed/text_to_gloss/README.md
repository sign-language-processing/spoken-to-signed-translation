# Text-to-Gloss

([Background](https://research.sign.mt/#text-to-gloss))


Each component implements `text_to_gloss`; reorder-only components can also accept pre-tokenized input:

```python
def text_to_gloss(text: str, language: str) -> List[Gloss]: ...


def tokens_to_gloss(tokens: Gloss, language: str, signed_language: str) -> List[Gloss]: ...
```

Both return sentences of `(word, gloss)` items. Token-based components return original item objects, without introducing
or duplicating tokens; grammar components may omit items as well as reorder them.

## `rules` and `gpt`

Use `rules` for rule-based glossing or `gpt` for model-based glossing, including English → ASL (`en` → `ase`).
The existing German/French text rules are unchanged. English text rules require the `spacy` extra.

Both accept pre-tokenized input via `tokens_to_gloss(tokens, language="en", signed_language="ase", metadata=metadata)`.
Each metadata entry supplies `pos` and optionally `morphology` (a list of feature dictionaries).
The English/ASL rules drop articles and present-tense support auxiliaries and move short WH phrases in simple questions.
They preserve unknown constructions and, without metadata, leave the tokens unchanged. Pre-tokenized rules currently
support only English → ASL; they never retokenize or split multiword items.

GPT chooses token indexes, with omissions limited to those allowed by the rules. Invalid indexes, duplicates,
missing required tokens and cross-sentence moves are rejected. This compares ordering strategies under the same
omission constraints, not unrestricted translation. Neither method implements full ASL grammar or nonmanuals.

Install the `gpt` extra and set `OPENAI_API_KEY`. `OPENAI_MODEL` defaults to `gpt-4o-mini`;
`OPENAI_BASE_URL` can point to a local OpenAI-compatible server (for example `http://localhost:1234/v1`).

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
