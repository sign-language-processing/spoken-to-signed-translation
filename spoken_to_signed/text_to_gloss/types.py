from typing import NamedTuple


class GlossItem(NamedTuple):
    word: str | None
    gloss: str


Gloss = list[GlossItem]
