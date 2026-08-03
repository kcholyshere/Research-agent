"""The Canvas Tool (phase 6): turns collected research into a finished artefact.

## What it is, and what makes it different from the other three tools

The Document Search, Web Search and Financial Data tools are all *evidence
gathering* - they answer "what is true". Canvas is the first tool here that
produces rather than retrieves, and that difference is structural rather than
cosmetic: it is terminal (nothing is planned after it), it never contributes a
fact, and it must be exempt from every bound and metric that exists to police
search volume. Two places enforce that and both are load-bearing:

- `src/research_agent/tool_budget.py` exempts it from `MAX_TOOL_CALLS_PER_TURN`.
  A report turn is plausibly KB search + reformulation + web + Canvas, which
  sits at the ceiling; if Canvas were the call that got refused, the artefact
  would silently not exist and the agent would answer in prose instead. That
  failure is invisible from the outside - the turn still produces an answer.
- `src/evaluation/schema.py` lists it in `OUTPUT_TOOLS`, so it does not count
  towards `tools_called` and therefore cannot fail `check_redundancy`. Without
  that, adding Canvas would push every artefact question one call over its
  `max_tool_calls` and read as a redundancy regression across the board.

## Why the signature is flat primitives

`title: str`, `output_format: str`, `section_headings: list[str]`,
`section_bodies: list[str]`, `citations: list[str]`, `language: str`. No nested
models, no dicts. ADK auto-wraps plain functions and derives the tool's
declared schema from the annotations, and what google-adk 2.5.0 does with a
nested Pydantic model or a `list[dict]` in that position is not something this
project has verified (CLAUDE.md is explicit that ADK APIs get verified against
the installed package, never assumed). Flat primitives are known to work, so
the structure is carried by two parallel lists instead of a list of objects.

The cost is real and worth naming: parallel lists can arrive at different
lengths, which a list of objects could not express. That is why the length
check below is an error return rather than an assertion - a mismatch is a
model mistake, and the model needs to be told what it did so it can retry,
not handed a stack trace.

## Why Pydantic and Jinja2 rather than f-strings

Phase 6's requirements name both as options. They pull their weight here for
different reasons.

Pydantic validates the *structure* at the boundary, so a malformed artefact
request produces a legible message the model can act on ("output_format must
be one of markdown, html, code") instead of a half-written file. It is doing
what the docstring cannot: the docstring persuades, the validator enforces -
the same split as prompt-versus-config in ADR-0009 and prompt-versus-ceiling
in tool_budget.py, which is by now the recurring lesson of this project.

Jinja2 keeps the three output formats as templates rather than as branching
string concatenation, so adding a fourth is a new template rather than a new
code path, and the HTML escaping that stops a citation containing `<` from
producing broken markup is the template engine's job rather than ours. Note
`autoescape` is per-format on purpose (see `_ENV`): escaping is correct for
HTML and actively wrong for Markdown, where `&` and `<` are legal text.

Note for maintenance: `src/evaluation/schema.py`'s module docstring said "if
phase 6's Canvas work brings Pydantic in for artefact validation, revisit"
whether the eval records should use it too. It should not - that reasoning
(written once, read once, no untrusted input) still holds for those records.
Canvas is the opposite case: its input comes from an LLM, which is exactly the
untrusted-input condition that makes validation worth its weight.

## Where the artefact goes

Both returned inline AND written to disk, which is deliberate and not
redundancy. Three consumers want different things:

- the *model* needs the content back to talk about what it produced;
- the *evaluation* needs it inside the research agent's own final response,
  because that is the only thing a run record captures (CLAUDE.md's rule about
  reading a turn's answer) - an artefact that lived only on disk would be
  invisible to every assertion;
- a *human* wants a file they can open.

Returning a path alone would satisfy only the third. It would also break the
critique loop in a way that is easy to miss: `critique_agent` reads
`draft_answer`, so if the answer became "I have written the report to X" the
critique would be reviewing a pointer, find no substance in it, and raise
follow-ups on every single artefact turn - burning refinement cycles on a
draft that was already finished.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import jinja2
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src import config

# Output formats. A Literal rather than a free string so Pydantic itself
# produces the "permitted: markdown, html, code" message, instead of this
# module hand-rolling one that can drift from the templates below.
OutputFormat = Literal["markdown", "html", "code"]

_EXTENSIONS: dict[str, str] = {"markdown": "md", "html": "html", "code": "txt"}

# Artefacts land under data/processed/, alongside the FAISS index and the
# stored evaluation runs - all three are build outputs rather than sources.
ARTEFACT_DIR = config.PROJECT_ROOT / "data" / "processed" / "artefacts"

# A cap on artefact size, for the same reason agent.py caps max_output_tokens:
# a generation loop is bounded in code, never by asking. 200 sections would be
# a runaway rather than a document.
MAX_SECTIONS = 40


class CanvasRequest(BaseModel):
    """A validated artefact request.

    Not part of the tool signature - `create_canvas` takes flat primitives and
    constructs this internally (see the module docstring for why). This model
    is where the structural rules live so they are enforced in one place and
    reported in the model's own terms.
    """

    title: str = Field(min_length=1)
    output_format: OutputFormat
    section_headings: list[str]
    section_bodies: list[str]
    citations: list[str] = Field(default_factory=list)
    language: str = ""

    @field_validator("section_headings", "section_bodies")
    @classmethod
    def _not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one section is required")
        if len(v) > MAX_SECTIONS:
            raise ValueError(f"at most {MAX_SECTIONS} sections are allowed, got {len(v)}")
        return v

    @model_validator(mode="after")
    def _parallel_lengths(self) -> CanvasRequest:
        if len(self.section_headings) != len(self.section_bodies):
            raise ValueError(
                f"section_headings and section_bodies must be the same length - "
                f"got {len(self.section_headings)} heading(s) and "
                f"{len(self.section_bodies)} body/bodies. They are parallel lists: "
                f"the Nth heading titles the Nth body."
            )
        return self

    @model_validator(mode="after")
    def _code_needs_language(self) -> CanvasRequest:
        if self.output_format == "code" and not self.language.strip():
            raise ValueError("output_format 'code' requires `language` (e.g. 'python', 'sql')")
        return self

    @property
    def sections(self) -> list[tuple[str, str]]:
        return list(zip(self.section_headings, self.section_bodies))


_MARKDOWN_TEMPLATE = """\
# {{ title }}
{% for heading, body in sections %}
## {{ heading }}
{{ body }}

{% endfor %}
{% if citations %}
## Sources
{% for c in citations %}
- {{ c }}
{% endfor %}

{% endif %}
*Generated by the research agent on {{ generated_at }}.*
"""

# The blank-line placement in the Markdown template is exact in both
# directions, and the first draft got it wrong in two ways worth recording
# because both produce *silently* malformed output rather than an error.
#
# No blank line directly AFTER a heading, per this repo's markdown rule in
# CLAUDE.md - an artefact this agent generates is a file in this project's
# house style, so the rule applies to generated markdown as much as to
# hand-written markdown.
#
# But a blank line BEFORE each heading and before the footer is mandatory, for
# rendering rather than style. Without it the trailing "*Generated by...*" line
# sits flush against the last "- citation" bullet, where markdown reads it as a
# lazy continuation of that list item: the footer disappears into the final
# bullet instead of standing alone. Body text running straight into the next
# `##` is the milder cousin of the same problem - CommonMark permits an ATX
# heading to interrupt a paragraph, but not every renderer does.
#
# `trim_blocks=True` on the environment is what makes this readable: it eats
# the newline that follows each `{% %}` tag, so the blank lines that appear in
# the template are exactly the blank lines that appear in the output.

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         line-height: 1.6; max-width: 44rem; margin: 0 auto; padding: 2rem 1.25rem; }
  h1 { font-size: 1.75rem; line-height: 1.25; margin-bottom: 1.5rem; }
  h2 { font-size: 1.15rem; margin-top: 2rem; }
  footer { margin-top: 3rem; font-size: 0.85rem; opacity: 0.7; }
  ul { padding-left: 1.25rem; }
  li { margin: 0.35rem 0; overflow-wrap: anywhere; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
{% for heading, body in sections %}
<section>
<h2>{{ heading }}</h2>
<p>{{ body }}</p>
</section>
{% endfor %}
{% if citations %}
<h2>Sources</h2>
<ul>
{% for c in citations %}
<li>{{ c }}</li>
{% endfor %}
</ul>
{% endif %}
<footer>Generated by the research agent on {{ generated_at }}.</footer>
</body>
</html>
"""

_CODE_TEMPLATE = """\
{{ comment }} {{ title }}
{{ comment }} Generated by the research agent on {{ generated_at }}.
{% for heading, body in sections %}
{{ comment }} --- {{ heading }} ---
{{ body }}

{% endfor %}
{% if citations %}
{{ comment }} Sources:
{% for c in citations %}
{{ comment }}   {{ c }}
{% endfor %}
{% endif %}
"""

# The `{%- %}` whitespace-stripping form was used here in the first draft and
# was wrong. `{%-` strips the preceding newline as well as the indentation, so
# every citation comment collapsed onto a single line with the "Sources:"
# header - "# Sources:#   page 63#   page 64" - producing one very long comment
# instead of a list. Caught on the first real code artefact (2026-08-03), not
# by the unit checks, because the round-trip tests asserted on substrings and a
# jammed line still contains every substring.
#
# Plain `{% %}` plus the environment's trim_blocks is the correct tool: it eats
# the newline that FOLLOWS a tag, which is what makes the template's own line
# breaks survive into the output. Same reasoning as the Markdown template
# above, and the same failure mode - silently malformed output rather than an
# error.

# Line-comment token per language, for the code template. Defaults to "#",
# which covers Python/shell/Ruby/YAML; the C family and SQL differ.
_COMMENT_TOKENS: dict[str, str] = {
    "javascript": "//", "typescript": "//", "java": "//", "c": "//", "cpp": "//",
    "c++": "//", "csharp": "//", "c#": "//", "go": "//", "rust": "//",
    "kotlin": "//", "swift": "//", "scala": "//", "php": "//",
    "sql": "--", "haskell": "--", "lua": "--",
}

# autoescape per format, not globally. Escaping is required for HTML and wrong
# for Markdown and code, where `&`, `<` and `>` are ordinary characters - a
# globally-escaped Markdown artefact would render "&amp;" as literal text.
_ENV = jinja2.Environment(
    loader=jinja2.DictLoader(
        {"markdown": _MARKDOWN_TEMPLATE, "html": _HTML_TEMPLATE, "code": _CODE_TEMPLATE}
    ),
    autoescape=jinja2.select_autoescape(enabled_extensions=("html",), default=False),
    trim_blocks=True,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)
# select_autoescape keys off a template *name* looking like a filename, and
# these are named "html"/"markdown"/"code" with no extension - so it would
# return False for all three. Set it explicitly on the HTML template instead
# of renaming the templates, which would make the format-to-template mapping
# indirect for no gain.
_ENV.autoescape = False
_HTML_ENV = _ENV.overlay(autoescape=True)


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (s[:60] or "artefact").rstrip("-")


def create_canvas(
    title: str,
    output_format: str,
    section_headings: list[str],
    section_bodies: list[str],
    citations: list[str],
    language: str = "",
) -> dict[str, Any]:
    """Render collected research into a finished artefact: a report, document, or code file.

    Call this ONLY when the user asked for a deliverable rather than an answer -
    a report, a document, a write-up, a summary document, a code file, a
    template. A plain question ("what was IFC's net income in FY24?") is
    answered directly and must NOT go through this tool. This is the last step
    of a turn: gather every fact you need with the research tools first, then
    call this once with the finished content. It creates nothing new and
    retrieves nothing - it only formats what you already have, so any fact not
    already retrieved will simply be absent from the artefact.

    Args:
        title: The artefact's title, e.g. "IFC FY24 Financial Performance".
        output_format: One of "markdown" (a report or document), "html" (a
            styled standalone page), or "code" (a commented source file).
        section_headings: One heading per section, in order.
        section_bodies: One body per section, in the SAME order and the SAME
            number as section_headings - the Nth heading titles the Nth body.
            Each body is the finished prose (or code) for that section, with
            its facts already cited inline as you would in a normal answer.
        citations: Every source used, as URLs or document-and-page references.
            These are collected into a Sources section at the end. Pass an
            empty list only if the artefact genuinely rests on no sources.
        language: Required when output_format is "code" - the language, e.g.
            "python", "sql", "javascript". Ignored for the other formats.

    Returns:
        On success, a dict with "status": "ok", the full rendered "artefact"
        text, the "path" it was written to, its "format" and "word_count".
        On invalid input, a dict with "status": "error" and a "detail" naming
        what to fix - correct it and call again.
    """
    try:
        request = CanvasRequest(
            title=title,
            output_format=output_format,  # type: ignore[arg-type]
            section_headings=section_headings,
            section_bodies=section_bodies,
            citations=citations,
            language=language,
        )
    except ValidationError as exc:
        # Flattened to one line per problem: the model reads this as a tool
        # response and Pydantic's default multi-line rendering (with "For
        # further information visit https://errors.pydantic.dev/...") is noise
        # in that context.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'request'}: {err['msg']}"
            for err in exc.errors()
        )
        return {
            "status": "error",
            "detail": f"The artefact request was not valid - {problems}. Fix and call create_canvas again.",
        }

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    env = _HTML_ENV if request.output_format == "html" else _ENV
    rendered = env.get_template(request.output_format).render(
        title=request.title,
        sections=request.sections,
        citations=request.citations,
        generated_at=generated_at,
        comment=_COMMENT_TOKENS.get(request.language.strip().lower(), "#"),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTEFACT_DIR / f"{stamp}_{_slug(request.title)}.{_EXTENSIONS[request.output_format]}"
    try:
        ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        written = str(path)
    except OSError as exc:
        # A disk failure must not lose the artefact: it is already rendered,
        # and the content matters more than the file. Return it with the write
        # failure noted rather than turning a successful render into an error
        # the model will retry - retrying would regenerate the whole document
        # for a problem no rewording can fix.
        written = f"(not written to disk: {exc})"

    return {
        "status": "ok",
        "format": request.output_format,
        "path": written,
        "word_count": len(rendered.split()),
        "artefact": rendered,
    }
