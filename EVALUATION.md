# Evaluation results
Version-controlled record of what each evaluation run showed. Design rationale is
ADR-0011 (what the pipeline measures) and ADR-0012 (how a sweep is scheduled).

## Read this before quoting a web-route number
Every figure below was measured while the web route went through Gemini's
`google_search` grounding. On 2026-08-10 that provider was replaced by the
Tavily Search API (ADR-0028), which changes what a web search returns, how its
citations are shaped, and how many model calls a web turn costs. Nothing here
is retracted - it is an accurate record of the system as it then was - but no
web-route latency, redundancy or citation number below describes the current
system, and only a fresh sweep would make one current again.

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
As of the 2026-08-03 20:20 sweep. Update this table and the date with each sweep.

| Defect | Evidence | Status |
|---|---|---|
| Falls back to web instead of declining, then answers from the wrong source | decline 40% pass (24/40 failing); `decline-headcount-by-country`, `decline-future-figure` and `fin-coverage-gap` all reach for the web | open, most serious - one bug behind two assertion keys |
| Redundant searching | 69% pass; `kb-charges-on-borrowings` and `canvas-code-artefact` both hit 6 calls against a max of 2 | open |
| Premise-refuting search storm | `web-premise-refuting`, up to 6 searches | open, known defect |
| Critique diagnoses correctly but does not repair | `loop-multipart-mixed` named the missing figure, ran a second cycle, still lacked it - at 3x the latency | open, new - see the Critic Agent section below |
| News Agent picked for non-news questions | `multi-web-and-financial` delegates on 8 of 8 runs | open, but may be a label decision rather than a defect |
| A2A timeout raises an unhandled AttributeError | `A2AClientTimeoutError` has no `status_code`; 1 run of 280 died on it | open, in the client library's error path |
| No turn timeout in production code | the eval harness bounds runs at 300s; the agent does not | open |
| Loop improvement barely measurable | 2 second cycles in 140 budget-1 runs, so "did cycle N+1 improve" has almost nothing to score | open |

Closed since the last table: KB answers now carry citations (page numbers and the
source filename), so citation passes at 98% across 271 checks. That row had been
carried as an open regression and this sweep does not reproduce it.

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
### 2026-08-03, later: the HTML artefact defect, fixed and re-measured
The defect above was reproduced first rather than patched from the summary, and
the mode was not "forgot to call the tool" but "did the tool's job itself". Of
four pre-fix runs, three hand-wrote a styled page into the answer - their own CSS
classes, `<div class="card">`, a `<style>` block - and one narrated a review of
its own stylesheet ("there is a minor bug in the CSS of Section 4, `</>` instead
of `</style>`"). Routing was correct on all four, so nothing failed but the
artefact assertion. The answers were also truncated mid-markup, because authoring
a whole HTML document runs into `max_output_tokens`.

The cause was ambiguous wording, not model waywardness. The tool described
`output_format: "html"` as "a styled standalone page", which reads as *you supply
one* rather than *this tool produces one*, and the planner instruction repeated
the phrasing. On that reading, hand-writing the markup is the obedient
interpretation. The fix removes the ambiguity rather than adding a fourth
prohibition to an instruction that already carried three.

| | artefact produced |
|---|---|
| before the fix | 1 of 4 runs |
| after the fix | 5 of 5 completed runs |

Every post-fix run called `create_canvas` exactly once and returned a 4.6-5.3 KB
HTML artefact with a short covering note, which is the intended shape. Two reps
across those batches died on OAuth transport errors from a tethered connection
and are excluded as environment rather than code.

The chain-of-thought leakage noted as a separate defect turned out to be the same
one. It appears only in the pre-fix run, where the model was reviewing its own
CSS; all five post-fix answers are clean of it. It was a symptom of the model
doing Canvas's job, not an independent defect, and needed no separate fix.

`check_artefact` now also fails any answer containing raw block-level or styling
markup, on every question rather than only artefact ones - writing a document into
the reply is a defect regardless of what was asked. Inline emphasis tags are
excluded deliberately, since they appear in quoted source text and would accuse
correct answers. Calibrated against the 248 stored answers from the 09:20 sweep:
zero false positives.

## 2026-08-03, later still - the phase 4 budget sweep, and what the loop actually does
The first sweep aimed at phase 4 rather than at routing.
`20260803T195834Z_live_reps4_budgets0-1.json`: the 5 `critique_loop` questions
plus `kb-net-income` and `kb-cur-definition` as controls, 2 arms x 4 reps = 56
runs, live, concurrency 4. 56/56 scored, 0 timeouts, 0 errors. Scoped rather
than full because the hypothesis was narrow: does cycle 2 earn its latency.

Every earlier sweep measured phase 4 by accident at best. The 2026-08-03 09:20
run reached `cycles >= 2` on 3 of its 124 budget-1 runs, and the post-phase-5/6
sweep was budget 0 only, which short-circuits the loop by design (ADR-0010). So
the project's architectural centrepiece had effectively no measurement.

### The headline: the loop ran, and chose to stop, every single time
All 28 budget-1 runs finished at `cycles=1`. Not one second cycle, including on
the five questions written specifically to provoke one.

This is not the loop being skipped. The two arms separate cleanly on
`critique_outcome`, which is what distinguishes the two explanations:

| arm | critique outcome | n |
|---|---|---|
| budget 0 | `skipped` (short-circuit before the LLM call) | 28/28 |
| budget 1 | `exit` (critique ran, reviewed, called `exit_loop`) | 28/28 |

Follow-up questions raised across all 56 runs: **zero**. The budget-0 column is
also the confirmation that ADR-0010's short-circuit still fires exactly as
specified - budget 0 reproduces the pre-phase-4 path with no critique call at all.

### What the loop costs when it decides to do nothing
Median latency per question, arm against arm:

| question | budget 0 | budget 1 | delta |
|---|---|---|---|
| loop-single-fact | 10.0s | 11.1s | +1.1s |
| loop-multipart-mixed | 18.7s | 21.6s | +2.9s |
| loop-definitional-single | 7.0s | 9.9s | +2.9s |
| kb-net-income | 8.4s | 11.5s | +3.1s |
| kb-cur-definition | 12.8s | 15.9s | +3.1s |
| loop-comparison | 25.4s | 28.8s | +3.4s |
| loop-multipart | 24.1s | 29.0s | +4.9s |

About +3.1s median, and it is close to a flat cost rather than a proportional
one - the critique call is one bounded LLM round trip whatever the question. On
the cheapest question that is +37%; on the most expensive, +20%.

The controls behave identically to the loop questions, which is the useful part:
budget 1 changes nothing about the research cycle, so the arms differ only in
whether a critique call happens after it. The routing and redundancy differences
between arms (91% against 89% hard pass) are therefore pure run-to-run variance
by construction, and they usefully calibrate the noise floor at n=28.

### Is the critique right to exit?
On this evidence, yes, and the one case that looked like a counter-example was an
instrument defect rather than a missed gap.

`loop-comparison` failed `content` on 3 of 4 budget-1 runs, which looked exactly
like the critique agent reading an incomplete answer and passing it: the question
has two halves, and the disbursements half appeared to be missing. It was not.
Every run answered both halves correctly, in millions - `$19,147m` against
`$18,689m` - while `must_contain` demanded `19.1` and `18.7`, the same two
figures in billions. The one run that passed did so only because it volunteered a
"(or ~$19.1 billion)" gloss alongside the millions figure.

The question was asserting commitments in millions and disbursements in billions,
so the agent had to write both renderings of the same number to pass. Fixed to
`19,147`/`18,689`, making all four figures consistent with the units the corpus
tables carry. Re-scored against the stored answers rather than re-run, since only
the assertion changed:

| | content failures |
|---|---|
| before the fix | 4 of 56 runs |
| after the fix | 0 of 56 runs |

That is the third instrument defect found by checking a suspicious number against
a stored answer rather than trusting the aggregate (after the citation/latency
pair in ADR-0013 and the decline threshold earlier today). The pattern is now
established enough to state as a rule: an assertion that fails on a question the
agent visibly got right is an assertion bug until proven otherwise.

### What this does and does not establish
It establishes that the loop terminates, that termination is overwhelmingly the
outcome, that the short-circuit works, and that the cost of a no-op critique pass
is about 3s.

It does not establish that the loop improves answers, because on this set it
never got the chance to try. `check_wasted_cycle` was applicable to 3 runs out of
248 in the 09:20 sweep and 0 out of 56 here - the detector for a loop burning
budget without progress has almost nothing to score. "Did cycle N+1 improve on
cycle N" remains unmeasured, and the reason is no longer that the metric is
missing but that cycle N+1 is.

The honest presentation framing: phase 4 currently buys insurance rather than
measured improvement. The instruction makes termination the default and requires
the model to justify continuing (ADR-0010), and it turns out that under that
instruction the model essentially never justifies continuing. Whether that is
correctly calibrated or too conservative is the open question, and answering it
needs questions with a deliberately unanswerable half rather than merely a
multi-part one.

## 2026-08-03, 20:20 - the final pre-presentation baseline (280 runs, both arms)
`20260803T202002Z_live_reps4_budgets0-1.json`. All 35 questions x 2 arms x 4
reps, live, concurrency 4, with the News Agent served on :8001 throughout. 279 of
280 scored; the one loss is analysed below rather than swept up.

| assertion | failing | checked | pass |
|---|---|---|---|
| artefact | 0 | 24 | 100% |
| citation | 5 | 271 | 98% |
| content | 11 | 104 | 89% |
| decline | 24 | 40 | 40% |
| redundancy | 86 | 279 | 69% |
| routing | 44 | 279 | 84% |
| wasted_cycle | 0 | 2 | 100% |
| total | 170 | 999 | 83% |

Hard pass 76% at budget 0 and 77% at budget 1 - the arms are indistinguishable,
as they were in the scoped sweep, and for the same structural reason: budget 1
changes nothing before the critique call.

### The Critic Agent, measured across the whole question set
The scoped sweep earlier tonight could only say the loop never continued on seven
knowledge-base questions. The full set is what shows how it behaves where gaps
genuinely exist.

| arm | critique outcome | n |
|---|---|---|
| budget 0 | `skipped` (short-circuit, no LLM call) | 139/139 |
| budget 1 | `exit` (reviewed, then stopped) | 121 |
| budget 1 | `skipped` (financial-only short-circuit) | 19 |
| budget 1 | `continue` (raised a follow-up) | 2 |

Three things are worth separating here.

**The short-circuits both work exactly as designed.** Budget 0 never reaches an
LLM call, so the pre-phase-4 path is reproduced precisely (ADR-0010). And the 19
`skipped` at budget 1 are the financial rule firing: a cycle whose only tool call
was `get_financial_data` has no citation to omit and no sub-question left, so it
escalates without a critique call at all. That is 19 model calls not made, on the
route that was already closest to its latency target.

**When it does review, it stops 121 times out of 123 - about 98%.** Two runs in
140 raised a follow-up. Both diagnoses were legitimate: `web-current-event` was
asked for the decision plus its citations, and `loop-multipart-mixed` was asked
for the LTF projects count its draft had not covered.

**The repair is where it breaks, and this is the finding of the sweep.**
`loop-multipart-mixed` asserts the figure `365`. Seven of its eight runs produced
it in a single cycle. The eighth is the run the critique caught - it correctly
named the missing figure, the second cycle ran another `search_documents`, and
the final answer still did not contain `365`:

| loop-multipart-mixed | cycles | contains `365` | latency |
|---|---|---|---|
| budget 0, reps 1-4 | 1 | yes (4/4) | 14.5-19.4s |
| budget 1, reps 1, 2, 4 | 1 | yes (3/3) | 21.3-23.9s |
| budget 1, rep 3 | 2 | **no** | **56.7s** |

So the one run that needed refinement is the one that ended up worst, at roughly
three times the median latency. The critique agent's diagnosis was right and its
repair did not land. That is a much more specific claim than "the loop rarely
runs", and it points somewhere different: the follow-up reaches the planner and
produces a search, but nothing carries the retrieved figure into the final
answer. Improving the loop means fixing repair, not tuning when it triggers.

The other continuation, `web-current-event`, grew its draft from 2,313 to 3,788
characters over 89.5s. Whether that is an improvement is not assertable - the
question is `volatile`, so content is only scored in replay.

**Cost.** Median latency 23.4s at budget 0 against 25.7s at budget 1 across all
questions (means 24.7s and 28.1s). About +2.3s median, consistent with the +3.1s
the scoped sweep measured on a narrower, cheaper set.

### The News Agent is being picked for questions that did not ask for news
`multi-web-and-financial` delegated to `news_agent` on **8 of 8 runs**, and three
other questions did so once each. The question is "What is the current price of
Bitcoin, and what has the European Central Bank most recently said about
regulating crypto assets?", labelled `[financial, web]` before phase 5 existed.

This needs a decision rather than a fix. "What has X most recently said" is
news-shaped, and delegating it to a specialist news agent is defensible - the
label predates the tool, exactly like the delegation questions did in the other
direction after phase 5 landed. Adding `news_agent` to that question's
`expected_routes` would recover 8 of the 44 routing failures, which is why it
should be an explicit call and not a quiet edit.

### One run lost, and it is worth naming
`delegate-news-topic` budget 0 rep 2 died with
`AttributeError("'A2AClientTimeoutError' object has no attribute 'status_code'")`
- an unhandled path inside the A2A client's own error handling, where a timeout
is passed to code expecting an HTTP response. ADR-0018 accepted losing the clean
unreachable-service error as the cost of addressing the remote as an agent rather
than an endpoint; this is that cost showing up concretely. The other seven
delegation runs routed correctly.

### What has not moved
Redundancy (69%) and decline (40%) are where they have been all along, on the
same questions. ADR-0015's conclusion continues to hold: a ceiling bounds the
worst case and cannot produce efficiency. Decline remains the most serious open
defect and the one with the clearest business consequence - answering from the
wrong source is worse than saying nothing.

## 2026-08-06 - the declared-plan sweep (288 runs, both arms)
`data/processed/eval_runs/20260806T200001Z_live_reps4_budgets0-1.json`. 36 questions x 2 critique budgets x 4 reps, concurrency 3, live. 288 runs, 288 scored, 0 timed out, 0 rate limited, 0 errored.

Four mechanisms landed between this sweep and the last one: `report_gap` and its rewritten decline assertion (ADR-0023), the `declare_plan` gate (ADR-0024), grounding-redirect resolution, and a critique rule that stops the critic chasing a fact the draft already declined.

### Read the hard pass rate carefully, or not at all
Hard pass is 78% at budget 0 and 75% at budget 1, against 84% recorded on 2026-08-03. That is not a regression and the two numbers are not comparable: the decline assertion was rewritten to be strictly stronger, a new question was added, and a new assertion (`critique_calibration`) was added that currently fails by construction.

The comparison that means something is the old sweep rescored on tonight's instrument, so both columns are measured the same way:

| assertion | 2026-08-03, rescored | 2026-08-06 |
|---|---|---|
| routing | 87% | **92%** |
| decline | 0% | **68%** |
| citation | 98% | 99% |
| content | 89% | 91% |
| redundancy | 69% | **61%** |
| artefact | 100% | 100% |

Decline's 0% in the left column is an artefact and should not be read as a 68-point gain: `report_gap` did not exist when those runs were recorded, so every one of them fails the first of the new check's three conditions automatically. The honest decline progression is the one measured on the same five questions under the same instrument: 45% before any behaviour change, 62% after the `declare_plan` gate, 68% here.

### Routing is the best it has ever been, and the mechanism is identifiable
92% is the highest routing figure this project has recorded, up from 87% on the same instrument and 91% on the looser one. `declare_plan` is the only change that could have moved it, and its effect is visible per question: `multi-kb-and-web` leaked to `news_agent` on 1 of 8 runs before and 0 of 8 now, and the decline questions stop reaching for the web once their declared source comes back empty. `fin-coverage-gap` went from 50% to 100% redundancy at 2.2 evidence calls down to 1.0 - it now consults the financial tool once, finds no coverage, and stops.

### Redundancy regressed, and the cause is the new mechanism
69% to 61%, and this is a real cost rather than a measurement artefact. Mean evidence calls per turn barely moved (2.59 to 2.77), so the regression is concentrated rather than general:

| question | old | new | old calls | new calls |
|---|---|---|---|---|
| `kb-cur-definition` | 100% | 0% | 1.5 | 3.5 |
| `canvas-not-requested` | 100% | 38% | 1.8 | 2.8 |
| `multi-kb-and-web` | 50% | 12% | 3.8 | 5.1 |

The obvious explanation is wrong and was checked rather than assumed. The theory was that `declare_plan` closes the sideways exit, so search pressure that used to leak out as a wrong-source call now stays on the right source as extra calls - the mirror image of what ADR-0015 saw when it moved from per-tool to per-turn budgets. `kb-cur-definition` refutes it: it routed to `knowledge_base` on 8 of 8 runs both before and after, so there was no routing failure to convert. It simply searches more, going from 1-2 calls to 3-5.

The actual cause is decomposition. `declare_plan` asks the planner to enumerate the distinct facts a question needs, and a fact that has been written down as its own line invites its own search. A definitional question whose answer is "Capital Required divided by Capital Available" now declares two facts and searches for each, where before it asked once. The mechanism improved which source gets consulted and made the agent more granular about what it asks that source for.

This is the agent being wasteful rather than the label being wrong: `kb-cur-definition` allows 2 calls for a single definition, and 3-5 is not defensible. The fix belongs in how coarsely the instruction asks for facts to be declared, not in raising the bound.

### What has not moved
`critique_calibration` fails 0 of 4, exactly as `loop-half-absent` predicted when it was added: the critic exits at cycle 1 because the draft explicitly declines the absent half, which the critic's own rules correctly treat as addressed. The assertion's premise - that a good critic keeps searching for an absent fact - is now in tension with the decided design, in which a correctly declined fact is answered rather than missing. The field needs redefining or retiring; it is not measuring what it was added to measure.

Over-searching a correctly declared single source is untouched, as ADR-0024 said it would be. That remains ADR-0015's territory and ADR-0015's stated limit.

## 2026-08-07 - the audit sweep: the fact-level plan gate, measured
288 runs, 36 questions, 2 arms, 4 reps, live mode. The first sweep after clearing the 2026-08-06 codebase audit (ADR-0026, ADR-0027).

Hypothesis stated before the run: routing holds at or above 92%, redundancy stays flat near 61% because the gate constrains which tool rather than how many times, decline holds at or above 68%, declaration uptake stays at 100% now the naming trap is closed, and the new `fact` argument produces near-zero refusals.

| metric | 2026-08-06 | this sweep | denominator |
|---|---|---|---|
| routing | 92% | **93%** | 268/288 |
| redundancy | 61% | 61% | 177/288 |
| decline | 68% | 68% | 27/40 |
| content | - | 90% | 123/136 |

Every prediction held. Hard pass is 76% at budget 0 and 78% at budget 1, against 78% and 75% on 2026-08-06 - the arms flipped and the spread is inside run-to-run variance. Do not compare either figure to the 83-84% recorded on 2026-08-03; `EVALUATION.md`'s own note on that sweep explains why the instrument changed underneath it.

### The number worth reading first: zero
Across 288 runs there was not one gate refusal. No `fact_not_in_declared_plan`, no `tool_not_the_declared_source_for_this_fact`, no rejected declaration. That was the real risk in ADR-0027 and the reason the sweep was run before merging: a required argument the model has to populate by quoting its own earlier output back verbatim is exactly the kind of mechanism that works in a smoke test and falls over at scale. It did not. The model declares a plan, then quotes each fact back exactly, on every turn.

Read that carefully, though, because it cuts both ways. Zero refusals also means the gate refused nothing, so this sweep provides no positive evidence that the fact-level check would catch a real mis-binding - only that it costs nothing when the model behaves. The routing gain from 92% to 93% is one percentage point on 288 runs and is not, on its own, evidence of anything.

### Redundancy is exactly where it was, as predicted
61%, unmoved. This was the explicit prediction rather than a disappointment: the gate constrains WHICH tool serves a fact, and redundancy is about HOW MANY times. `loop-half-absent`, `decline-segment-margin` and `decline-auditor-fee` each spend six calls on 8 of 8 runs - five allowed plus one refused by the numeric ceiling. Nothing in ADR-0027 touches that, and the open TODO naming declaration granularity as the cause still stands.

The 111 redundancy failures dominate the regression list and are the single largest lever left on the hard pass rate. They are not new.

### What this does not tell us
The sweep cannot distinguish "the fact gate is correct" from "the fact gate is inert", because the model never triggered it. Deciding between those needs an adversarial question - one whose correct plan names two sources for two facts and where answering one from the other is tempting - and the question set has no such case by construction. That is worth adding before the mechanism is credited with anything.
