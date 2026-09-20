# Senses → ASL lexical plans

This is a deterministic **ordering and omission** experiment, not a complete ASL
translator. A source-word plan is not a native gloss transcription. A good index
sequence cannot supply classifiers, spatial agreement, role shift, nonmanuals or
lexical aspect. Those remain realization work; we do not silently discard their
English cues. Unknown words remain available for downstream fingerspelling.

## Component ownership

WSD exposes language-neutral `dep`, `head`, `ent_type`, morphology, inclusive
sentence boundaries, selected senses and linked entities. It does not gloss.
`senses_to_gloss(document, semantics=...)` is the public utility here; the HTTP
adapter only validates its contract and invokes it. WordNet supplies semantics.
No second tokenization, WSD model, dictionary database logic or LLM runs here.

The stages are explicit functions, not a rule-engine framework:

1. Validate sentence partitions and dependency trees.
2. Group WSD meanings into atomic candidates; retain original tokens and annotations.
3. Apply guarded omissions within each sentence.
4. Establish guarded location and temporal frames without splitting phrases.
5. Normalize simple question order and retain source alignment.
6. Omit standalone punctuation after interpreting it; retain sentence boundaries and question notes.

`changes` identifies each applied rule and its original token indexes; `notes`
reports missing semantic configuration, complex-clause fallback and unrealized
question nonmanuals. The original WSD document remains authoritative. Add a rule
as a small function in `asl.py`, with explicit preconditions and contrast tests.
The existing Python token/GPT paths remain for comparison, not HTTP selection.

## Rules and limits

| Rule | Preconditions | Deliberately preserved |
| --- | --- | --- |
| Article omission | A singleton a/an/the tagged DET, outside entities | Possessives, demonstratives, atomic meanings |
| Present copula omission | Present AUX be, dependency ROOT with a complement | Existential there, passive, progressive, ellipsis, past |
| Do-support omission | Present AUX do/does with syntactic negation or in a root question | Emphatic affirmative do, lexical do, past did |
| Infinitival to omission | Singleton PART/aux of a VERB/xcomp under want, need, like, try, plan, hope, decide or prefer | Recipient/direction, ellipsis, atomic spans; unresolved used-to, have-to, remember-to |
| Event-location frame | Exact selected event-location sense of at; single simple PP on an affirmative transitive root | Unknown/target senses, noun attachment, questions, auxiliaries, negation, focus, multiple PPs |
| Temporal frame first | Resolved temporal sense, root adverbial, whole contiguous phrase | Objects, durations, embedded clauses, protected entities |
| Temporal phrase extensions | on/at + DATE/TIME noun under a supported event predicate; or quantity + temporal unit + ago | Prepositions and ago retained; ambiguous PP predicates, for/in/since/until relations unchanged |
| Subject before auxiliary | Simple question with one unambiguous contiguous subject | Modal/tense/aspect auxiliaries themselves |
| WH-final | Small initial WH phrase in a simple question | Complex/embedded questions, uncertain structures |
| Default | Source SVO order | No blanket OSV or preposition/conjunction deletion |

Time-first and WH-final are **canonicalization choices**, not universal ASL
requirements. Bill Vicars documents substantial variation and warns against
mandatory time-fronting. Basic SVO is a reasonable default; topicalization needs
context, not just a permutation. See [time placement](https://www.lifeprint.com/asl101/topics/time-first.htm),
[SVO](https://www.lifeprint.com/asl101/topics/subject-verb-object-asl-sentence-structure.htm),
and [WH variation](https://www.lifeprint.com/asl101/lessons/lesson07.htm).

The omission policy is informed by [ASL grammar](https://www.lifeprint.com/asl101/pages-layout/grammar.htm)
and [be-verbs](https://www.lifeprint.com/asl101/pages-signs/b/be-verbs.htm).
Keeping auxiliaries in unresolved constructions is an engineering safeguard,
**not** a claim that ASL signs those English auxiliaries.
The new omissions are informed by [infinitives and incorporated prepositions](https://www.lifeprint.com/asl101/topics/preprositions-asl-preposition-incorporation-or-drop.htm).
Event-location fronting is a narrow canonicalization choice, not a claim that
all ASL locations come first. It retains the location phrase, omits only the
resolved event-locative marker, and keeps pronouns/possessives. The location's
spatial/nonmanual realization is still missing. Other senses of at remain.
The exact sense `wikidata-en-L3263-S2` was verified through the WordNet API:
"indicating a location for an event" (S1 is an action's target).

Temporal PP fronting additionally requires one of the supported root predicates
in `TEMPORAL_PP_PREDICATES`: parse attachment and a temporal noun alone cannot
distinguish "meet on Monday" from "reflect on Monday". The prepositions remain
because reordering does not license erasing their relation. The ago extension
preserves both quantity and past direction. Complex-clause reordering remains
deferred; guarded infinitive omission can apply locally without moving a clause.
[WH questions require nonmanual grammar](https://www.lifeprint.com/asl101/pages-layout/whfacialexpression.htm),
which this lexical plan does not render.

### WordNet classification

The adapter follows `hypernym` and `instance_hypernym` in the existing WordNet
API using **selected sense IDs**, never a new lookup of the surface word.
Temporal roots in OMW English 1.4 are `15113229-n` (time period), `15180528-n`
(point in time), `15154774-n` (time unit), and `15129927-n` (time-of-day reading),
with `omw-en-` prefixes. Clock times such as noon follow a measurement/reading
ancestry rather than the other three roots; this was verified through the API.

[Adverbs do not have noun-style hypernym chains](https://wordnet.princeton.edu/documentation/wninput5wn).
`TIME_ADVERBS` lists five exact
selected senses for yesterday, tomorrow, today and tonight, checked against the
local WordNet API. It is deliberately incomplete: unknown senses keep their
position. A title named Yesterday is not temporal just because of its spelling.
Time classification excludes linked entities and non-temporal NER categories,
so a name cannot become a temporal frame. This is not a global ban on moving
entities: location phrases and question subjects can move as whole units with
all their annotations intact. No omission rule may split or delete an entity.

Taxonomy is necessary but insufficient. spaCy parses both “I remember yesterday”
and “They left yesterday” with a bare NOUN temporal adverbial. We abstain from
moving bare temporal nouns without a direct object or phrase modifier. That
costs some preferred time-first outputs but avoids changing remembered content
into the time of remembering. Better predicate-role analysis belongs upstream.

Cache successful parent lookups (bounded to 8192), not network errors. The
service returns 503 for a configured WordNet outage. Without configuration, it
explicitly notes that semantic ordering was skipped. Pin WordNet resources with
the service. TODO: batch distinct-sense ancestry queries when WordNet exposes one.

## Evaluation protocol

`cases.json` was committed **before** implementing the rules: 24 development and
16 held-out sentences. `challenge.json` adds 20 cases after the first iteration:
negation, questions, relative clauses, coordination, conditionals, ellipsis,
quotations, names, time phrases and deliberately misleading temporal senses.
These are original engineering examples, not copied corpus annotations.

The metric compares retained source words (excluding standalone punctuation), order and sentence boundaries,
case-insensitively. Repeated words also have index/provenance unit tests. Some
expectations explicitly require conservative English-shaped output. Therefore
**a passing percentage is not translation accuracy or native-ASL acceptance**.
Unchanged complex cases count as preservation tests, not fluent translations.

There are two separate experiments:

- **Isolated rules:** real `en_core_web_lg` syntax plus supplied temporal senses.
  This deliberately isolates rule behavior, not WSD accuracy. Some supplied
  adverb senses accompany spaCy noun tags; the syntactic ambiguity guard still
  applies. Full upstream behavior is tested separately below.
- **Live integration:** the real WSD response, including its actual selected
  senses and entities, passed unchanged to the Docker HTTP service.

Historical results, before punctuation omission (spaCy 3.8.16, en_core_web_lg 3.8.0):

| Version | Development | Held-out | Challenge |
| --- | ---: | ---: | ---: |
| Released token rules | 20/24 | 12/16 | not measured |
| Syntax-aware omissions | 21/24 | not inspected | not measured |
| Semantic frames + simple questions + ambiguity guard | 24/24 | 15/16 | 20/20 |

The held-out miss is “They left yesterday”: it remains in source order. This is
a preferred-order mismatch, not proven ungrammatical ASL. The held-out reference
was not edited to turn that miss into a pass. No model training or model judge
was used. The implementer reviewed failures; independent Deaf-ASL review is
still required before making fluency claims.

### Historical live integration and manual judgment (before punctuation omission)

The patched native WSD service (`sign/Ettin-150m-WSD`, revision
`8751b577199d1bb95b74fa2457da7065d57100ae`) fed the Docker glosser unchanged.
It matched **39/40** original and **20/20** challenge ordering expectations.
The remaining mismatch is the same conservative bare-time-noun case. The service
was also checked over HTTP/2, including its `X-Model-Tag` header.

| Input | Returned source-word plan | Review |
| --- | --- | --- |
| What is your name? | your name What ? | Correct intended omission/order; question nonmanuals still missing |
| I deposited my paycheck at the bank yesterday. | yesterday I deposited my paycheck at bank . | Whole temporal frame moves; selected bank sense is the financial institution |
| We meet next Tuesday. | next Tuesday We meet . | Modifier and day move together |
| I remember yesterday. | I remember yesterday . | Correct conservative scope preservation |
| The book is read by the teacher. | book is read by teacher . | Safe preservation, not a completed ASL passive translation |
| My name is Amit. | My name Amit . | Order passes, but WSD incorrectly links Amit to Q1579 (Ganesha) |

The last case is an **upstream semantic error**. This glosser must not silently
replace a supplied entity with a guess. It is a known end-to-end risk: lookup
could select the wrong sign despite a perfect order score. Entity confidence /
abstention should be addressed in WSD before claiming production translation
quality. Complex clauses, retained auxiliaries and quotation preservation likewise
need further linguistic work; preservation is not naturalization.

Run from the repository with `.[spacy]` and `en_core_web_lg` installed:

```bash
PYTHONPATH=. python evaluation/asl/evaluate.py --baseline --split dev
PYTHONPATH=. python evaluation/asl/evaluate.py --wordnet-url http://localhost:8080
PYTHONPATH=. python evaluation/asl/evaluate.py --cases evaluation/asl/challenge.json \
  --wordnet-url http://localhost:8080
PYTHONPATH=. python evaluation/asl/evaluate.py --wsd-url http://localhost:8081 \
  --glosser-url http://localhost:8082 --output /tmp/asl-live.json
```

`--output` preserves complete WSD inputs and gloss responses for case-by-case
review. Without it, the runner prints compact results. Generated runs are not
committed. The unit suite checks invalid/cyclic/cross-sentence trees, atomic and
overlapping spans, semantic outages, entity protection and source preservation.

### Phrase-rule regression run

`phrases.json` contains 25 original positive/contrast cases for the new rules.
Run it with real spaCy syntax and the WordNet container:

```bash
PYTHONPATH=. python evaluation/asl/evaluate.py --cases evaluation/asl/phrases.json \
  --wordnet-url http://localhost:8080 --output /tmp/asl-phrases.json
```

With spaCy 3.8.16 / en_core_web_lg 3.8.0 and WordNet v1.8.0: 25/25 phrase
cases, 39/40 original cases (the same bare-yesterday abstention), 20/20 challenge
cases. One challenge expectation intentionally changed: "I want to buy a book"
now omits infinitival to. The original held-out references are unchanged.
These are engineering regressions, not native-ASL accuracy measurements.
Unit tests independently supply explicit trees to exercise guards and atomic
span/provenance invariants without adding a spaCy model to the runtime or CI.

## Datasets researched

Verification: the final run passed 163 spoken-to-signed tests and 38 targeted WSD
tests. Docker build, HTTP/2 POST, version headers and changed-file lint passed.
One intermediate run failed in the unchanged pose-coverage E2E test with a pose
buffer error; its isolated rerun and the full final rerun passed. That intermittent
pose test was not changed as part of this glossing experiment.

| Dataset | Useful for | Caveat |
| --- | --- | --- |
| [ASLLRP / NCSLGR](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html) | Genuine continuous ASL, manual gloss and nonmanual annotations; strongest next evaluation source | [Downloads require an account](https://dai.cs.rutgers.edu/dai/s/daioriginal); [terms](https://www.bu.edu/asllrp/dai-terms.html) restrict commercial use and redistribution without permission |
| [ASLG-PC12](https://achrafothman.net/site/english-asl-gloss-parallel-corpus-2012-aslg-pc12/) | Large English/gloss stress corpus | Rule-generated, so not independent linguistic gold; CC BY-NC 4.0 |
| [How2Sign](https://how2sign.github.io/) | ASL video and English translation review | Public download section provides videos/keypoints/translations, not a readily usable gloss-order gold set; CC BY-NC 4.0 |

No restricted corpus was copied into this repository and no corpus score is
claimed. Next: obtain an authorized ASLLRP subset and native-signer review, align
meaning units rather than English spellings, and score omission errors, temporal
scope, clause boundaries and nonmanual coverage separately. Corpus gloss labels
are sign identifiers, not necessarily the English lemmas used by this utility.
