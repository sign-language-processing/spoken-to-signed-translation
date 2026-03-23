# originally written by Anne Goehring
# adapted by Mathias Müller
import re
import sys
from collections.abc import Iterator

from .common import load_spacy_model
from .types import Gloss, GlossItem

LANGUAGE_MODELS_RULES = {
    "de": ("de_core_news_lg", "de_core_news_md", "de_core_news_sm"),
    "fr": ("fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"),
}


def print_token(token):
    print(
        token.text,
        token.ent_type_,
        token.lemma_,
        token.pos_,
        token.tag_,
        token.dep_,
        token.head,
        token.morph,
        file=sys.stderr,
    )


def _to_infinitive(lemma: str) -> str:
    """Convert a verb lemma to infinitive form.

    When spaCy fails to lemmatize a verb it returns the word form itself (e.g.
    "Machst" for "machen").  Strip common German conjugation suffixes so that we
    reconstruct a reasonable infinitive rather than producing garbage like
    "machstn".  Lemmas that already end in "n" (i.e. are already in infinitive
    form) are returned unchanged.
    """
    lemma = lemma.lower()
    if lemma.endswith("en"):
        return lemma
    # Strip conjugation suffixes in order from longest to shortest so that
    # "machest" is handled before the shorter "est" branch would be tried.
    for suffix in ("est", "st", "et", "t", "e"):
        if lemma.endswith(suffix) and len(lemma) > len(suffix) + 1:
            lemma = lemma[: -len(suffix)]
            break
    if not lemma.endswith("en"):
        lemma += "en"
    return lemma


def attach_svp(tokens):
    for token in tokens:
        # When spaCy fails to lemmatize a verb it returns the word form itself;
        # apply heuristic infinitive recovery for all such verbs (covers both
        # separable-prefix constructions and imperatives like "Mache").
        if token.pos_ == "VERB":
            if token.lemma_.lower() == token.text.lower():
                token.lemma_ = _to_infinitive(token.lemma_)
        # prefix the separable verb particle to the (now corrected) lemma
        elif token.dep_ == "svp":
            token.head.lemma_ = token.lemma_ + token.head.lemma_


def get_clauses(tokens):
    # for token in tokens:
    #    print_token(token)
    def diff(l1, l2):
        return [x for x in l1 if x not in l2]

    verbs = [
        t
        for t in tokens
        if (t.pos_ == "VERB" and t.dep_ != "oc")
        or (t.pos_ == "AUX" and t.dep_ == "mo" and t.head.pos_ == "VERB")  # AUX in subclause
        or t.dep_ == "ROOT"  # ROOT to catch AUX in main clause
    ]
    subtrees = [[t for t in v.subtree] for v in verbs]
    clauses = [s for s in subtrees]
    subtrees.sort(key=len, reverse=True)
    new_clauses = []
    for clause in clauses:
        new_clause = clause
        for s in subtrees:
            if len(s) < len(new_clause):
                diff_clause = diff(new_clause, s)
                if diff_clause:
                    new_clause = diff_clause
        new_clauses.append(new_clause)

    return new_clauses


def reorder_sub_main(clauses):
    # find which clause is the subordinate
    # wenn KOUS ->cp-> benötigen ->mo-> Suchen: MAIN-SUBwenn to be reordered as SUBwenn-MAIN+dann?
    # Wenn KOUS ->cp-> benötigen ->re-> dann: already ordered as SUBwenn-MAINdann
    sub_clause = -1
    main_clause = -1
    main_verb = None

    for i, clause in enumerate(clauses):
        for token in clause:
            if token.tag_ == "KOUS" and token.dep_ == "cp" and token.head.dep_ == "mo":
                main_verb = token.head.head
                sub_clause = i

    if sub_clause >= 0:
        assert main_verb is not None

        # print(f"sub_clause: {clauses[sub_clause]}", file=sys.stderr)
        for j, clause in enumerate(clauses):
            if main_verb in clause:
                main_clause = j
        # if main_clause >= 0: # assert as there should be a main clause for the subordinate!
        #    print(f"main_clause: {clauses[main_clause]}", file=sys.stderr)

        if sub_clause > main_clause:
            # swap(clauses, sub_clause, main_clause)
            # TODO: instead of simply swapping them, should rather put the subclause in front of the corresponding main clause, in the case there are more than 2 clauses...
            clauses[sub_clause], clauses[main_clause] = clauses[main_clause], clauses[sub_clause]

    return clauses


def get_triplets(pairs, word_order="sov"):
    # pairs: [(s,v), (o,v)]
    triplets = []

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            # same verb
            if pairs[i][1] == pairs[j][1]:
                v = pairs[i][1]  # or pairs[j][1]
                if pairs[i][0].dep_ in {"sb", "nsubj"}:
                    s = pairs[i][0]
                    o = pairs[j][0]
                else:
                    s = pairs[j][0]
                    o = pairs[i][0]
                if word_order == "sov":
                    triplets.append((s, o, v))
                elif word_order == "svo":
                    triplets.append((s, v, o))
                elif word_order == "osv":
                    triplets.append((o, s, v))

    return triplets


def swap(tokens, token_a, token_b):
    # token_a(fter) should swap with token_b(efore) in the sequence of tokens
    # [ ...b...a...] => [...a...b...]
    new_tokens = []

    # move the verb
    if token_a.head == token_b:
        verb = token_b
        subtree = list(token_a.subtree)
        # print('move the verb after the subtree', file=sys.stderr)
        insubtree = False
        for t in tokens:
            if t == verb:
                continue
            if t in subtree:
                insubtree = True
            elif insubtree:
                new_tokens.append(verb)
                insubtree = False
            new_tokens.append(t)
        if insubtree:
            new_tokens.append(verb)

    elif token_b.head == token_a:
        verb = token_a
        subtree = list(token_b.subtree)
        put_a = False
        # print('move the verb before the subtree', file=sys.stderr)
        for t in tokens:
            if t == verb:
                continue
            if t in subtree and not put_a:
                new_tokens.append(verb)
                put_a = True
            new_tokens.append(t)
    else:
        # print('swap the subject and the object', file=sys.stderr)
        subtree_a = list(token_a.subtree)
        subtree_b = list(token_b.subtree)
        put_a = False
        for t in [t for t in tokens if t not in subtree_a]:
            if t in subtree_b and not put_a:
                new_tokens.extend(subtree_a)
                put_a = True
            new_tokens.append(t)

    return new_tokens


def reorder_svo_triplets(clause, word_order="sov"):
    pairs = []
    for token in clause:
        # print_token(token)
        if token.dep_ in {
            "sb",
            "oa",  # DE
            "nsubj",
            "obj",
            "obl:arg",  # FR
        }:
            pairs.append((token, token.head))

    # print(pairs, file=sys.stderr)
    reordered_triplets = get_triplets(pairs, word_order=word_order)
    # print(word_order, 'reordered:', reordered_triplets, file=sys.stderr)
    if reordered_triplets:
        # 6 possible (a,b,c)-triplet order
        (token_a, token_b, token_c) = reordered_triplets[0]
        if (token_a.i < token_b.i) and (token_b.i < token_c.i):
            # print('# 1,2,3 => no change', file=sys.stderr)
            pass
        elif (token_a.i < token_b.i) and (token_b.i > token_c.i):
            # print('# 1,3,2 => swap 3,2', file=sys.stderr)
            # print_token(token_b)
            # print_token(token_c)
            clause = swap(clause, token_b, token_c)
        elif (token_a.i > token_b.i) and (token_a.i < token_c.i):
            # print('# 2,1,3 => swap 2,1', file=sys.stderr)
            clause = swap(clause, token_a, token_b)
        elif (token_a.i < token_b.i) and (token_a.i > token_c.i):
            print("# 2,3,1 => put 1 before", file=sys.stderr)  # TODO
            pass
        elif (token_a.i > token_b.i) and (token_a.i > token_c.i):
            print("# 3,1,2 => put 3 after", file=sys.stderr)  # TODO
            pass
        elif (token_a.i > token_b.i) and (token_b.i > token_c.i):
            # print('# 3,2,1 => swap 3,1', file=sys.stderr)
            clause = swap(clause, token_a, token_c)

    return clause


def haben_main_verb(token):
    if token.lemma_ == "haben":
        # is there a dependent main verb?
        for c in token.children:
            if c.pos_ == "VERB" and c.dep_ == "oc":
                return False
        return True

    return False


def gloss_de_poss_pronoun(token):
    # DE: mein/dein/sein/ihr/Ihr/unser/euer
    pposat_map = {
        "M": "mein",
        "m": "mein",
        "D": "dein",
        "d": "dein",
        "S": "sein",
        "s": "sein",
        "i": "ihr",
        "I": "Ihr",
        "U": "unser",
        "u": "unser",
        "E": "euer",
        "e": "euer",
    }

    return pposat_map[token.text[0]] + "-IX"


def glossify(tokens) -> Iterator[GlossItem]:
    for t in tokens:
        # print_token(t)

        # default: lemmatize
        gloss = t.lemma_

        # Plural nouns with suffix "+"
        if t.tag_ == "NN" and "Number=Plur" in t.morph:
            gloss += "+"

        # word form for adverbs (as spacy DE models sometimes set a wrong lemma)
        elif t.pos_ == "ADV":
            gloss = t.text.lower()

        # mark German attributive possessive pronouns with "-IX" suffix, e.g. dein-IX
        elif t.tag_ == "PPOSAT":
            gloss = gloss_de_poss_pronoun(t)

        # lowercased word form for pronouns since the lemma sometimes looses the person information
        elif t.tag_ in [
            "PPER",
            "PRF",
            "PDS",  # DE, e.g. "Wir  ich PRON PPER ..."
            "PRON",
            "DET",  # FR, e.g. "sa  son DET DET ..."
        ]:
            gloss = t.text.lower() + "-IX"

        # DE "haben" as main verb should be glossed as "DA"
        elif haben_main_verb(t):
            gloss = "da"

        # other forms of "haben" and "sein" (auxiliary) should be skipped
        # FR: avons  avoir AUX AUX aux:tense
        elif (
            t.lemma_ in {"habe", "haben", "sein"}  # DE
            or (t.lemma_ == "avoir" and t.pos_ == "AUX")
        ):  # FR
            continue

        # # DE: lemma of NER-identified location entities preceded by preposition
        # if t.ent_type_ == "LOC" and t.head.pos_ == "ADP":
        #     glosses.append(t.head.text)

        yield GlossItem(word=t.text, gloss=gloss)


def clause_to_gloss(clause, lang: str, punctuation=False) -> list[GlossItem]:
    # Rule 1: Extract subject-verb-object triplets and reorder them
    clause = reorder_svo_triplets(clause)

    # Rule 2: Discard all tokens with unwanted PoS
    tokens = [
        t
        for t in clause
        if t.pos_ in {"NOUN", "VERB", "PROPN", "ADJ", "NUM", "AUX", "SCONJ", "X"}
        or (punctuation and t.pos_ == "PUNCT")
        or (t.pos_ == "ADV" and t.dep_ != "svp")
        or (t.pos_ == "PRON" and t.dep_ != "ep")
        or (t.dep_ == "ng")
        or (t.lemma_ == "kein")
        or (t.tag_ in {"PTKNEG", "KON", "PPOSAT"})  # TODO: "PDAT" e.g. gloss("dieses")=IX?
        or (t.tag_ == "DET" and "Poss=Yes" in t.morph)  # son  son DET DET det ami Number=Sing|Poss=Yes
        or (t.tag_ == "CCONJ" and (lang != "de" or t.lemma_.lower() != "und"))  # FR: mais
    ]

    # Apply punctuation as its own lemma
    if punctuation:
        for t in tokens:
            if t.pos_ == "PUNCT":
                t.lemma_ = t.text

    # Rule 3: Move adverbs to the start?
    # TODO: Move verb modifying adverbs before the verb in each clause
    adverbs = [t for t in tokens if t.pos_ == "ADV" and t.dep_ == "mo" and t.head.pos_ == "VERB"]
    tokens = [t for t in tokens if t not in adverbs]
    tokens = adverbs + tokens

    # Rule 4: Move location words to the start
    # TODO: move only if it modifies the verb?
    locations = [t for t in tokens if t.ent_type_ == "LOC"]
    tokens = [t for t in tokens if t not in locations]
    tokens = locations + tokens

    # # Rule 5: Move negation words to the end
    # negations = [t for t in tokens if t.dep_ == "ng"]
    # tokens = [t for t in tokens if t not in negations] + negations
    #
    # if len(tokens) > 0 and lang == "de":
    #     from spacy.tokens import Token
    #
    #     token = tokens[0]
    #     extra_token_id = len(token.doc)
    #
    #     neg_token = Token(token.vocab, token.doc, extra_token_id)
    #     neg_token.lemma_ = "<neg>"
    #     extra_token_id += 1
    #
    #     neg_close_token = Token(token.vocab, token.doc, extra_token_id)
    #     neg_close_token.lemma_ = "</neg>"
    #     extra_token_id += 1
    #
    #     for token in list(tokens):
    #         if token.dep_ == "ng":
    #             tokens.insert(0, neg_token)
    #             tokens.remove(token)
    #             tokens.append(neg_close_token)
    #         elif token.lemma_ == "kein":
    #             tokens.insert(tokens.index(token), neg_token)
    #             tokens.append(neg_close_token)

    # TODO: is compound splitting necessary? only taking the first noun loses information!
    # Rule 6: Replace compound nouns with the first noun
    for i, t in enumerate(tokens):
        if t.dep_ == "compound":
            tokens[i] = t.head

    # Rule 7: Glossify all tokens, i.e. lemmatize most tokens
    return list(glossify(tokens))


def expand_contractions_de(text: str) -> str:
    """Expand German verb contractions of the form "verb's" → "verb es".

    Lowercase "'s" is treated as a contraction of the pronoun "es" (e.g.
    "wird's" → "wird es", "gibt's" → "gibt es").  Uppercase "'s" is left
    untouched because it is almost certainly a possessive suffix on a proper
    noun or brand name (e.g. "McDonald's", "Anna's").

    Known limitation: a sentence-initial contracted verb (e.g. "Gibt's noch
    Kaffee?") starts with a capital letter and will not be expanded.
    """
    return re.sub(r"\b([a-z]\w*)'s\b", r"\1 es", text)


def text_to_gloss_given_spacy_model(text: str, spacy_model, lang: str = "de", punctuation=False) -> dict:
    if text.strip() == "":
        return {"glosses": [], "tokens": [], "gloss_string": ""}

    if lang == "de":
        text = expand_contractions_de(text)

    doc = spacy_model(text)

    if lang != "fr":
        # Rule 0: Attach separable verb particle to the verb lemma, but not for French
        attach_svp(doc)

    # split sentence into separate clauses
    clauses = get_clauses(doc)

    # reorder clauses
    clauses = reorder_sub_main(clauses)

    glossed_clauses: list[list[GlossItem]] = []

    for clause in clauses:
        items = clause_to_gloss(clause, lang, punctuation=punctuation)
        glossed_clauses.append(items)

    all_items = [item for clause_items in glossed_clauses for item in clause_items]

    # clause separator "|" and end of sentence "||"
    gloss_string = " | ".join([" ".join(item.gloss for item in clause_items) for clause_items in glossed_clauses])
    gloss_string += " ||"

    # Final Rule: Begin sequence with a capital
    gloss_string = gloss_string.title()

    return {
        "glosses": [item.gloss for item in all_items],
        "tokens": [item.word for item in all_items],
        "gloss_string": gloss_string,
    }


def text_to_gloss(text: str, language: str, punctuation=False, **unused_kwargs) -> list[Gloss]:
    if language not in LANGUAGE_MODELS_RULES:
        raise NotImplementedError(f"Don't know language '{language}'.")

    model_names = LANGUAGE_MODELS_RULES[language]

    spacy_model = load_spacy_model(model_names)
    output_dict = text_to_gloss_given_spacy_model(text, spacy_model=spacy_model, lang=language, punctuation=punctuation)

    glosses = output_dict["glosses"]
    tokens = output_dict["tokens"]

    return [[GlossItem(word=t, gloss=g) for t, g in zip(tokens, glosses)]]
