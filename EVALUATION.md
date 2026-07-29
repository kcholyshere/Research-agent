# Evaluation results
Version-controlled record of what each evaluation run showed. Design rationale is
ADR-0011 (what the pipeline measures) and ADR-0012 (how a sweep is scheduled).

## Latest baseline - 2026-07-29 (post-fix)
248 runs, live, concurrency 4. `20260729T163541Z_live_reps4_budgets0-1.json`.

| arm | runs | scored | hard pass | prev | cycles=2 |
|---|---|---|---|---|---|
| budget0 | 124 | 124 | 76% | 81% | 0 |
| budget1 | 124 | 124 | 81% | 81% | 4 |

Failing assertions against the same-day pre-fix baseline, both scored with the
corrected metrics (ADR-0013), so this is like-for-like:

| assertion | before | after |
|---|---|---|
| redundancy | 59 | 66 |
| routing | 35 | 42 |
| decline | 31 | 32 |
| citation | 7 | 7 |
| content | 6 | 5 |
| total | 138 | 152 |

Latency, timed phase: budget0 median 9.1s, budget1 10.7s - both inside the 15s
target. Concurrent phase median 22.9s / 23.1s, down from 25.5s / 28.5s.
Reliability improved outright: 0 errored, 0 timed out, 0 rate-limited, 248/248
scored, against 4 lost runs before.

### The tool-call cap traded one defect for another
Per-tool call volume fell exactly as intended - `kb-charges-on-borrowings` 6.5
to 4.1 mean calls, `decline-auditor-fee` 11.4 to 6.6 (and 181-290s to ~80s),
`web-premise-refuting` 5.0 to 3.9. No run now exceeds 8 tool calls; the old
worst was 12.

But assertions got worse, for two reasons that are the same reason:

1. The cap is per tool (3), while the eval's `max_tool_calls` bounds the turn
   in total (1 to 4). A turn making 3 knowledge-base calls and 3 web calls is
   within the cap and still fails redundancy.
2. Refusing a `search_documents` call pushes the agent to the web rather than
   to an answer. `decline-segment-margin` and `decline-headcount-by-country`
   now go to web on 8 of 8 runs, `decline-auditor-fee` 7 of 8. The refusal text
   says in terms "do not substitute a different source"; it is ignored, which
   is the same finding that motivated the cap in the first place.

So the cap bounded the search storm and redirected it across tools instead of
stopping it. Net: cheaper and faster turns, more assertion failures. The fix
for this is a per-turn total ceiling rather than a per-tool one - what the
eval actually asserts - not a firmer refusal message.

## Where things live
| Thing | Path | Tracked |
|---|---|---|
| Question set (31 labelled questions) | `data/eval/questions.yaml` | yes |
| Replay fixtures (all 15 volatile questions) | `data/eval/fixtures/` | yes |
| Record schema | `src/evaluation/schema.py` | yes |
| Runner + CLI | `src/evaluation/run_eval.py` | yes |
| Assertions + aggregation | `src/evaluation/metrics.py` | yes |
| Record/replay layer | `src/evaluation/replay.py` | yes |
| Langfuse sync (opt-in, `--sync-langfuse`) | `src/evaluation/langfuse_sync.py` | yes |
| Raw run outputs | `data/processed/eval_runs/*.json` | no |

## What the 31 questions cover
Each question is tagged with what it exists to test. Counts overlap - a question
can be both `kb` and `critique_loop`.

| Category | n | What it asserts |
|---|---|---|
| `kb` | 14 | Answerable from the IFC corpus alone; the web must not be consulted |
| `web` | 7 | Public knowledge the corpus does not hold; the KB must not be consulted |
| `financial` | 5 | Live prices via the MCP tool - stocks, crypto, currencies, plus its coverage gap |
| `multi_source` | 5 | Genuinely needs two or three tools; consulting only one is the failure |
| `decline` | 5 | Structurally unanswerable - the agent must say so plainly and stop |
| `critique_loop` | 5 | Aimed at the phase 4 loop rather than routing: does a second cycle earn its budget |
| `redundancy_check` | 2 | Sounds like it belongs to the corpus and does not - catches the reflex KB call |
| `phase5_candidate` | 2 | News-shaped; routes to web today, flips to the A2A News Agent when it exists |
| `premise_refuting` | 1 | The question's premise is false; the agent must conclude that, not keep searching |

Three tags cut across the above rather than describing a question's subject:
`volatile` (15) means the answer moves with the news or market, so content is
asserted only in replay against a fixture; `latency_target` (6) marks the runs
timed in isolation; `known_defect` (2) marks a question expected to fail today
against a logged defect, reported apart from fresh regressions.

## Open defects
As of the 2026-07-29 baseline. Update this table and the date with each sweep.

| Defect | Evidence | Status |
|---|---|---|
| Falls back to web instead of declining, then answers from the wrong source | 6 questions; routing 35 + decline 37. `decline-fy25-commitments` returns FY24 `31,654` for an FY25 question | open, most serious - one bug behind two assertion keys |
| Redundant searching | `kb-charges-on-borrowings` up to 10 KB calls for one figure | open |
| KB answers carry no citation | 40 hits, incl. `kb-net-income` - the attribution fix covered web/financial, not the corpus | open, regression of a "fixed" defect |
| Premise-refuting search storm | `web-premise-refuting`, up to 12 searches | open, known defect |
| No turn timeout in production code | the eval harness bounds runs at 300s; the agent does not | open |
| Loop improvement unmeasured | wasted-cycle detection is deterministic only; "did cycle N+1 improve" needs Tier 3 | not built |

<details>
<summary>Run history</summary>

### 2026-07-29 - full baseline on the 31-question set
See "Latest baseline" above. First sweep after the question-set expansion and the
two-phase concurrent scheduler (ADR-0012); 4x the throughput per run of the
2026-07-28 sweep. Confirmed `6346b44`'s figure-substitution fix was
period-specific rather than general: `decline-fy25-commitments` reproduces it.

### 2026-07-29 - attribution fix (subset, 16 runs)
`20260729T100106Z_live_reps2_budgets0-1.json`, 4 questions. Not comparable
like-for-like with a full sweep.

Gemini's `google_search` grounding never puts URLs in the model's text - they
arrive as structured `grounding_metadata`, and `AgentTool` forwards only text, so
the research agent had never been given a URL it could cite. `"(Google Search)"`
was the model naming the only source it could see. An `after_agent_callback` on
the web sub-agent now appends a `Sources:` block from the grounding chunks. The
financial half was a prompt gap: `get_financial_data` already returned its source
URL and synthesis never used it. The seven `attribution-broken` known-defect
markers were removed, since a stale marker files a real future regression as
already-known.

### 2026-07-28 - first baseline (126 runs, pre-attribution-fix)
`20260728T134242Z_live_reps3_budgets0-1.json`. budget0 69% hard pass, budget1
71%. Latency `cycles=1` median 18.1s / 20.0s. Regressions: redundancy 39, routing
20, citation 14, decline 11, content 4.

Two metric defects were found by checking suspicious numbers against stored
answers, and fixed before these figures were trusted: the citation check only
matched a literal `.pdf` filename (16 false positives per arm), and the latency
label compared a single-tool verdict against a single-cycle median. Figures above
are post-fix.

</details>
