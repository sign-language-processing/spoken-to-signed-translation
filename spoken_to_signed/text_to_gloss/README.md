# Text-to-Gloss

([Background](https://research.sign.mt/#text-to-gloss))


Each file must implement a `text_to_gloss` function with the following signature:

```python
def text_to_gloss(text: str, language: str) -> Gloss:
    ...
```

It should return a list of tuples, each containing the original word and its gloss.