"""Tests for attach_svp, _to_infinitive, and expand_contractions_de in rules.py."""
from spoken_to_signed.text_to_gloss.rules import _to_infinitive, attach_svp, expand_contractions_de


class MockToken:
    """Minimal stand-in for a spaCy Token sufficient for attach_svp."""

    def __init__(self, text, pos_, dep_, lemma_, children=None):
        self.text = text
        self.pos_ = pos_
        self.dep_ = dep_
        self.lemma_ = lemma_
        self._children = children or []
        self.head = None

    @property
    def children(self):
        return iter(self._children)


# ---------------------------------------------------------------------------
# expand_contractions_de
# ---------------------------------------------------------------------------


class TestExpandContractionsDE:
    def test_lowercase_contraction_expanded(self):
        assert expand_contractions_de("Für die Falcon Bank wird's immer ungemütlicher") == (
            "Für die Falcon Bank wird es immer ungemütlicher"
        )

    def test_multiple_contractions(self):
        assert expand_contractions_de("gibt's und geht's") == "gibt es und geht es"

    def test_uppercase_possessive_untouched(self):
        assert expand_contractions_de("McDonald's Burger") == "McDonald's Burger"

    def test_mixed(self):
        assert expand_contractions_de("Anna's Katze gibt's nicht") == "Anna's Katze gibt es nicht"

    def test_no_contraction(self):
        assert expand_contractions_de("Kleine Kinder essen Pizza.") == "Kleine Kinder essen Pizza."


# ---------------------------------------------------------------------------
# _to_infinitive
# ---------------------------------------------------------------------------


class TestToInfinitive:
    def test_already_infinitive_en(self):
        assert _to_infinitive("machen") == "machen"

    def test_already_infinitive_en_uppercase(self):
        assert _to_infinitive("Gehen") == "gehen"

    def test_strip_st_suffix(self):
        # "Machst" (2nd person sg.) → "machen"
        assert _to_infinitive("Machst") == "machen"

    def test_strip_t_suffix(self):
        # "macht" (3rd person sg.) → "machen"
        assert _to_infinitive("macht") == "machen"

    def test_strip_e_suffix(self):
        # "mache" (1st person sg. / imperative) → "machen"
        assert _to_infinitive("mache") == "machen"

    def test_strip_est_suffix(self):
        # "machest" (archaic 2nd person sg.) → "machen"
        assert _to_infinitive("machest") == "machen"

    def test_stem_without_suffix_gets_en(self):
        # "zieh" has no conjugation suffix → "ziehen"
        assert _to_infinitive("zieh") == "ziehen"

    def test_n_ending_not_en_gets_en_appended(self):
        # "renn" (imperative) ends in "n" but not "en" → "rennen"
        assert _to_infinitive("renn") == "rennen"


# ---------------------------------------------------------------------------
# attach_svp
# ---------------------------------------------------------------------------


def _make_verb_with_svp(verb_text, verb_lemma, svp_text):
    """Build a pair of MockTokens representing a verb and its svp child."""
    verb = MockToken(verb_text, "VERB", "ROOT", verb_lemma)
    svp = MockToken(svp_text, "PART", "svp", svp_text)
    verb._children = [svp]
    svp.head = verb
    return verb, svp


class TestAttachSvp:
    def test_unknown_lemma_machst_aus(self):
        # spaCy failed to lemmatize "Machst" → lemma == word form → "ausmachen"
        verb, svp = _make_verb_with_svp("Machst", "Machst", "aus")
        attach_svp([verb, svp])
        assert verb.lemma_ == "ausmachen"

    def test_known_lemma_ziehen_an(self):
        # spaCy correctly returned "ziehen" → lemma != word form → unchanged before svp
        verb, svp = _make_verb_with_svp("ziehst", "ziehen", "an")
        attach_svp([verb, svp])
        assert verb.lemma_ == "anziehen"

    def test_imperative_unknown_lemma(self):
        # "Mache deine Hausaufgaben" — spaCy returns "Mache" as lemma
        verb = MockToken("Mache", "VERB", "ROOT", "Mache")
        attach_svp([verb])
        assert verb.lemma_ == "machen"

    def test_imperative_already_infinitive(self):
        # Verb in infinitive form used as imperative — spaCy returns correct lemma
        verb = MockToken("machen", "VERB", "ROOT", "machen")
        attach_svp([verb])
        assert verb.lemma_ == "machen"

    def test_verb_with_known_lemma_no_svp_is_unchanged(self):
        # spaCy knows the lemma → lemma != text → not touched
        verb = MockToken("wird", "VERB", "ROOT", "werden")
        attach_svp([verb])
        assert verb.lemma_ == "werden"

    def test_non_verb_token_is_unchanged(self):
        noun = MockToken("Licht", "NOUN", "obj", "Licht")
        attach_svp([noun])
        assert noun.lemma_ == "Licht"
