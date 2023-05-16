# Text-to-Gloss

Each file must implement a `text_to_gloss` function with the following signature:

```python
def text_to_gloss(text: str, language: str) -> List[Tuple[str, str]]:
    ...
```

It should return a list of tuples, each containing the original word and its gloss.