# Text-to-Gloss

([Background](https://research.sign.mt/#text-to-gloss))


Each component implements `text_to_gloss`; reorder-only components can also accept pre-tokenized input:

```python
def text_to_gloss(text: str, language: str) -> List[Gloss]: ...


def tokens_to_gloss(tokens: Gloss, language: str, signed_language: str) -> List[Gloss]: ...
```

Both return sentences of `(word, gloss)` items. `tokens_to_gloss` must preserve every item exactly once, changing only
their order.

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
