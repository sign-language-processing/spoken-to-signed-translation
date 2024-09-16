# Text-to-Gloss

([Background](https://research.sign.mt/#text-to-gloss))


Each file must implement a `text_to_gloss` function with the following signature:

```python
def text_to_gloss(text: str, language: str) -> List[Gloss]:
    ...
```

It should return a list of sentences, of tuples, each containing the original word and its gloss.

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
