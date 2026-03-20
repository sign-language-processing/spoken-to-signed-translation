from typing import NamedTuple, Optional


class GlossItem(NamedTuple):
    word: Optional[str]
    gloss: str


Gloss = list[GlossItem]
