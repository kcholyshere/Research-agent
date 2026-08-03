# Evaluation results
Version-controlled record of what each evaluation run showed. Design rationale is
ADR-0011 (what the pipeline measures) and ADR-0012 (how a sweep is scheduled).

## Latest baseline - 2026-08-03
248 runs, live, concurrency 4. `20260803T092025Z_live_reps4_budgets0-1.json`.
Three same-question sweeps, all scored with the corrected metrics (ADR-0013):

| assertion | no cap | cap 3/tool | cap 5/turn | checked | pass |
|---|---|---|---|---|---|
| redundancy | 59 | 66 | 73 | 248 | 71% |
| routing | 35 | 42 | 37 | 248 | 85% |
| decline | 31 | 32 | 33 | 48 | 31% |
| citation | 7 | 7 | 9 | 248 | 96% |
| content | 6 | 5 | 5 | 88 | 94% |
| total | 138 | 152 | 157 | 883 | 82% |

Hard pass: 81/81, then 76/81, then 77/79. Reliability is unchanged and good
throughout - 248/248 scored, no timeouts, errors or quota rejections.

### The cap has now been tried twice and lost twice
Counting the turn instead of the tool did what it was meant to: routing 42 to
37, unexpected-web on `decline-segment-margin` 8/8 to 2/8, worst run 8 calls
to 6. Closing the sideways exit was right.

The number was wrong, and it was wrong in the obvious direction. Per tool the
allowance was 3; per turn it is 5. Most of the set is single-tool with
`max_tool_calls: 2`, so those questions had their allowance raised from 3 to
5 and used it - `decline-segment-margin` and `decline-auditor-fee` at 6.0 mean
calls, `kb-charges-on-borrowings` 5.9. 54 of 73 redundancy failures sit on
`max_tool_calls: 2` questions, where a ceiling of 5 barely binds at all.

The finding underneath both attempts is ADR-0015's: a ceiling bounds the worst
case and cannot produce efficiency. It stops the storm; it does not make the
agent plan. Whatever the ceiling is, the agent spends up to it.

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
## 2026-08-03 - post-phase-5/6 sweep (140 runs, budget 0 only)
First sweep with the phase 5 News Agent and the phase 6 Canvas tool in place.
35 questions x 1 arm x 4 reps, live, concurrency 4, 0 timeouts and 0 errors.
Single arm by choice: the budget-0 vs budget-1 comparison was the agreed first
thing to cut for time, and budget 0 is directly comparable to the budget-0 half
of the four earlier sweeps.

| assertion | failing | checked | pass |
|---|---|---|---|
| routing | 13 | 140 | 91% |
| redundancy | 47 | 140 | 66% |
| citation | 3 | 136 | 98% |
| decline | 8 | 20 | 60% |
| content | 4 | 52 | 92% |
| artefact | 4 | 12 | 67% |
| total | 79 | 500 | 84% |

Hard pass rate (routing plus redundancy) 79%.

### What landed
Phase 5 routing works: both delegation questions go to `get_latest_news` with
one call each, and routing improved to 91% - the best it has measured on this
project.

The over-triggering guard fired **zero times across all 140 runs**. That is the
result worth keeping from this sweep: `check_artefact` scores "no artefact was
produced" on every question that did not ask for one, and no non-artefact
question produced a document. The gate on the synthesise step holds, so phase 6
did not leak into the prose path.

`canvas-not-requested` also spent exactly the same three `search_documents`
calls as `kb-net-income`, which is the control confirming a deliverable request
does not increase search volume.

### One real new defect: HTML artefacts never get rendered
`canvas-html-multi-source` failed `artefact` 4 of 4 and `citation` 3 of 4.
Routing was correct every time (knowledge base plus financial), so the research
worked; the finalise step simply did not fire. The model writes the HTML inline
in its reply instead of calling `create_canvas`, and the stored answers show it
reasoning about whether to - one begins "Wait, can we use CSS inside the
`create_canvas` tool for HTML?", another "Wait, let's complete the HTML tags and
avoid cutting off".

Two things to fix, and they are separable:
- Step 4 of the planner instruction is permissive about HTML in a way it is not
  about markdown or code, and the model reads authoring markup as its own job.
  Markdown and code artefacts rendered fine, so this is specific to the format
  the model believes it can write itself.
- Chain-of-thought is leaking into the answer text on this question. That is not
  a Canvas defect and would be worth a look on its own.

### Redundancy is unchanged, as expected
66% pass, and the failures are the same questions as every previous sweep
(`kb-charges-on-borrowings` at 5 calls, `decline-segment-margin` at 6,
`web-premise-refuting` at 6). ADR-0015 predicted exactly this: a ceiling bounds
the worst case and cannot produce efficiency. The three Canvas questions fail
redundancy for the same reason the rest of the knowledge-base set does, not for
anything phase 6 introduced - the control above is what establishes that.

### Note on comparing against earlier runs
Re-scoring the 2026-08-03 09:20 sweep with today's question set now shows
routing at 80% rather than 87%, because the two delegation questions expect
`[news_agent]` and that run predates the tool. That is a label change, not a
regression - a pre-phase-5 run cannot satisfy a post-phase-5 label. Compare
those two questions only within their own era.
