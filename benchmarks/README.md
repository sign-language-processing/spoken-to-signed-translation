# Local indexed-glossing experiments

These 19 hand-authored examples/probes check a preferred token order and preservation of meaning-bearing items.
All cases were inspected during iteration: this is not held-out evaluation or an ASL accuracy measurement.
ASL permits multiple word orders, including [different WH placements](https://www.lifeprint.com/asl101/topics/wh-question-placement.htm).
Token indexes cannot represent nonmanual grammar, spatial agreement, or morphology realization.

Run from the checkout after installing the `dev` extra, with LM Studio serving `openai/gpt-oss-20b`:

```sh
PYTHONPATH=. python benchmarks/token_gloss.py --variants production --no-structured --repeats 2 --output /tmp/gloss.json
```

The JSON records the prompt, raw replies, token selection, and per-call duration. Invalid outputs are rejected using
the same index validator as production. No expected answer is sent to the model. These are short correctness probes,
not throughput measurements: model/cache warmup and concurrent local work were not controlled for timing comparisons.

## September 11, 2026 observations

| Approach | Preferred-order matches |
| --- | ---: |
| Deterministic ASL rules (after regression fixes) | 16 / 19 |
| Original keep-every-index instruction, forced JSON | 5 / 19 |
| Add eligible omissions, forced JSON | 11 / 19 |
| Add grammar guidance, forced JSON | 7 / 19 |
| Same guidance without forced JSON | 14 / 19 |
| Add phrase examples | 13 / 19 |
| Limit changes to explicit operations (selected prompt, two runs) | 31 / 38 (15/19, 16/19) |
| Further AUX-pattern clarification, two repetitions (rejected) | 28 / 38 |

The last clarification also produced four truncated replies at the diagnostic's 512-token output cap; it was reverted.
Both selected-prompt runs returned all 19 structurally valid selections; the differing orders show that temperature zero
and a fixed seed did not make this local serving configuration fully deterministic.
The runtime allows 1024 output tokens, but still rejects invalid/truncated output rather than silently shortening a sentence.
Forced JSON yielded cleaner syntax but worse orders on this local model, so it is not enabled in the glosser.

The model can handle WH phrases and time fronting that the cheap rules leave alone, but regresses other cases.
Keep model-assisted glossing opt-in. The reliable changes are the index contract and rule regressions: preserve
emphatic `do`, and do not move leading WH words across coordinated clauses. Dictionaries/fingerspelling remain downstream.
