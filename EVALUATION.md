# Evaluation results

Running record of evaluation runs across iterations. Raw run outputs live in
`data/processed/eval_runs/` and are **gitignored** (regenerable measurement
noise); this file is the version-controlled record of what each run showed and
what changed between them.

Design rationale for the pipeline itself is ADR-0011; the plan it implements is
`references/evaluation_brainstorm.md`.

## Where things live

| Thing | Path | Tracked |
|---|---|---|
| Question set (21 labelled questions) | `data/eval/questions.yaml` | yes |
| Replay fixtures | `data/eval/fixtures/` | yes |
| Record schema | `src/evaluation/schema.py` | yes |
| Runner + CLI | `src/evaluation/run_eval.py` | yes |
| Assertions + aggregation | `src/evaluation/metrics.py` | yes |
| Record/replay layer | `src/evaluation/replay.py` | yes |
| Langfuse sync (opt-in, **not yet wired**) | `src/evaluation/langfuse_sync.py` | yes |
| Raw run outputs | `data/processed/eval_runs/*.json` | no |
| This record | `EVALUATION.md` | yes |

## How to reproduce

```bash
# Full sweep: 21 questions x 2 arms x 3 reps = 126 runs, ~2 hours, sequential
PYTHONPATH=. python -u -m src.evaluation.run_eval --reps 3 --mode live

# Fast smoke: filter by question id or tag
PYTHONPATH=. python -u -m src.evaluation.run_eval --questions kb-net-income,fin-crypto --reps 1

# Record fixtures, then replay deterministically (content assertions on volatile questions)
PYTHONPATH=. python -u -m src.evaluation.run_eval --mode record --questions <id>
PYTHONPATH=. python -u -m src.evaluation.run_eval --mode replay --questions <id>
```

Sequential by necessity: parallel runs contend for Vertex quota and inflate the
latency being measured. Metrics can be recomputed over a stored run file with no
agent calls, so a metric bug does not cost another sweep.

---

## Run history

### 2026-07-28 - first baseline (pre-attribution-fix)

`20260728T134242Z_live_reps3_budgets0-1.json` - 126 runs, reps=3, live mode,
arms `critique_budget` 0 and 1. 5 runs lost to Vertex `429 RESOURCE_EXHAUSTED`.
66 of 121 scored runs clean.

| arm | runs | hard pass | regressions | known-defect hits |
|---|---|---|---|---|
| budget0 | 63 | 69% | 53 | 19 |
| budget1 | 63 | 71% | 50 | 17 |

Latency (median / IQR, seconds), segmented by cycle count:

| arm | cycles | n | median | IQR | max |
|---|---|---|---|---|---|
| budget0 | 1 | 63 | 18.1 | 38.2 | 298.1 |
| budget1 | 1 | 61 | 20.0 | 38.1 | 299.5 |
| budget1 | 2 | 2 | 184.5 | 83.4 | 212.3 |

Regressions by assertion: redundancy 39, routing 20, citation 14, decline 11,
content 4.

Failing 6/6 (fully reproducible, not noise):

- `web-ifc-parent` **routing** - calls the knowledge base for a public question.
  This is the phase 2 redundancy defect that commit `390f940` was believed to
  have fixed. It is not fixed.
- `fin-coverage-gap` **routing + decline** - on USD/PLN it falls back to web and
  produces a number instead of declining.
- `decline-future-figure` **routing 6/6, decline 5/6, content 4/6** - asked for
  FY2027 net income, returns the FY2024 figure `1,485`. Most serious finding: a
  plausible answer to a different question.
- `kb-charges-on-borrowings` **redundancy** - up to 8 KB calls for one figure.
- `web-current-event` **redundancy** - up to 10 web searches.
- `loop-multipart` **redundancy** - 5-6 calls against a max of 3.

Notes:

- The multi-cycle path fired (2 runs at `cycles=2`), so the per-cycle split and
  `critique_outcome` handling were exercised on real turns rather than only
  synthetically. Two runs is thin evidence.
- Two metric defects were found by checking suspicious numbers against stored
  answers, and fixed before these figures were trusted: the citation check only
  matched a literal `.pdf` filename (16 false positives per arm), and the
  latency label compared a single-tool verdict against a single-cycle median.
  Figures above are post-fix, recomputed from the stored records.

### 2026-07-29 - attribution fix (subset only)

`20260729T100106Z_live_reps2_budgets0-1.json` - 16 runs, reps=2, live, 4
questions (`web-wbg-president`, `fin-currency`, `delegate-news-topic`,
`kb-net-income`). **A subset, not comparable like-for-like with the baseline
above.**

| arm | runs | hard pass | regressions |
|---|---|---|---|
| budget0 | 8 | 100% | 0 |
| budget1 | 8 | 75% | 4 (routing + redundancy only) |

Latency: budget1 `cycles=1` median 12.4s (IQR 9.1); `cycles=2` median 104.2s.

What changed: web attribution was repaired at the tool boundary. Gemini's
`google_search` grounding never puts URLs in the model's text - they arrive as
structured `grounding_metadata`, and `AgentTool` forwards only text, so the
research agent had never been given a URL it could cite. `"(Google Search)"` was
the model naming the only source it could see. An `after_agent_callback` on the
web sub-agent now appends a `Sources:` block from the grounding chunks. The
financial half was a genuine prompt gap: `get_financial_data` already returned
its source URL and synthesis simply never used it.

Answers now carry real URLs, e.g.
`Source: [Yahoo Finance Currencies](https://finance.yahoo.com/markets/currencies/)`.

The seven `attribution-broken` known-defect markers were removed from the
question set: the claim they encoded ("no web trace has yet produced a real
URL") is no longer true, and a stale marker files a genuine future regression as
already-known.

**Outstanding:** the full baseline has not been re-run since this fix, so its
citation and latency columns are stale. Routing, redundancy and decline findings
are unaffected.

---

## Open defects the eval currently measures

| Defect | Evidence | Status |
|---|---|---|
| Answers a different question with a plausible figure | `decline-future-figure` returns FY2024 `1,485` for a FY2027 question | open, most serious |
| Falls back to web instead of declining | `fin-coverage-gap` routing + decline 6/6 | open |
| Public questions still hit the knowledge base | `web-ifc-parent` routing 6/6 | open, regression of a "fixed" defect |
| Redundant searching | up to 8 KB calls, 10 web searches on single questions | open |
| Premise-refuting search storm | `web-premise-refuting`, up to 12 searches | open, known defect |
| Decline padding | `decline-headcount-by-country` pads to 366 words | open, known defect |
| No turn timeout in production code | eval harness bounds runs at 300s; the agent itself does not | open |
| Loop improvement unmeasured | wasted-cycle detection is deterministic only; "did cycle N+1 improve" needs Tier 3 | not built |
