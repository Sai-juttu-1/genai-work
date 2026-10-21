"""Session 17 helpers - evaluation: a judge call, a parsed verdict, a pass-rate comparison, and a human spot-check.

Five ideas live here:
  1. The judge call: one model call per answer, grading all four rubric criteria at once, strict JSON out.
  2. A parsed verdict: four fixed keys, two allowed values, nothing else - never a free-text "mostly fine, I think".
  3. Two prompts, the same golden set, the same judge: v1 is Session 14's baseline, v2 is YOUR Session 16 files.
  4. An order guard: for every ticket, which prompt is judged first is shuffled, so no timing or caching quirk
     favours one prompt over the other.
  5. A human spot-check: you grade a handful of tickets yourself and compare with the judge, so the judge's
     verdicts are evidence you have checked, not blind trust.

Takes a Lab (module02/reasoning_utils.py) for both the "answerer" calls and the "judge" calls - they can be the
same Lab (same model playing both roles, today's default) or two different ones (a stronger judge, a limit named
on the guardrails slide but not exercised in this notebook)."""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

import jinja2

from evaluation_tasks import DEFAULT_FORMAT_RULE, GOLDEN20, GUIDE, JUDGE_SYSTEM, JUDGE_USER, RUBRIC, RUBRIC_IDS, V1_INSTRUCTION
from prompt_utils import CheckResult
from structured_utils import PromptFileError, SUPPORT_DIR, SYSTEM_FILE, USER_FILE, load_prompt_file, pick_examples, structured_maker

REPORT_PATH = Path("experiments") / "s17_eval.md"
S17_CACHE = Path.home() / ".genai_course" / "s17_cache.json"


# ---------------------------------------------------------------- templates: the same strict-undefined engine as Session 16
def make_env() -> jinja2.Environment:
    return jinja2.Environment(undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True,
                              keep_trailing_newline=False, autoescape=False)


def render(template_text: str, **variables) -> str:
    return make_env().from_string(template_text).render(**variables).strip()


_JUDGE_TAG_RE = re.compile(r"<(\s*/?\s*(?:ticket|answer|rubric|rulebook)\b)", re.IGNORECASE)


def escape_judge_tags(text: str) -> str:
    """A ticket or a model's answer must not be able to close the judge's <ticket>/<answer>/<rubric>/<rulebook> tags."""
    return _JUDGE_TAG_RE.sub(r"&lt;\1", text)


# ---------------------------------------------------------------- v1 and v2: two ways to answer the same ticket
def v1_messages(ticket: str) -> list:
    """Session 14's baseline: one line of instruction, the ticket pasted in, nothing else."""
    return [{"role": "system", "content": V1_INSTRUCTION}, {"role": "user", "content": ticket}]


def v2_messages_from_files(repo_path, ticket: str, k_per_label: int = 2) -> list:
    """Session 16's structured prompt, loaded from the student's OWN work repository right now."""
    folder = Path(repo_path) / SUPPORT_DIR
    system = load_prompt_file(folder / SYSTEM_FILE)
    template = load_prompt_file(folder / USER_FILE)
    maker = structured_maker(system.body, template.body, pick_examples(k_per_label=k_per_label))
    return maker(ticket)


def answer_with(lab, messages: list, max_tokens: int = 1500) -> dict:
    """One model call that answers a ticket (not a classifier label - a full customer-support reply).

    max_tokens is a safety ceiling, not a target: reasoning models (like the course's gpt-oss-20b) spend hidden
    thinking tokens from the same budget, and a tight ceiling leaves an empty reply."""
    r = lab.chat(messages, max_tokens=max_tokens)
    return {"text": r["text"].strip(), "truncated": r["truncated"], "tokens": r["prompt_tokens"] + r["completion_tokens"],
            "seconds": r["seconds"], "cached": r["cached"]}


# ---------------------------------------------------------------- the judge: one call, four verdicts
class JudgeParseError(ValueError):
    """The judge's reply was not the exact JSON shape the rubric needs. The message says what was wrong."""


def judge_messages(ticket: str, truth: str, answer: str, *, rubric=RUBRIC, guide: str = GUIDE,
                   format_rule: str = DEFAULT_FORMAT_RULE) -> list:
    user = render(JUDGE_USER, ticket=escape_judge_tags(ticket), truth=truth, answer=escape_judge_tags(answer),
                 guide=guide, format_rule=format_rule)
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str, ids=RUBRIC_IDS) -> dict:
    """{'c1': True, 'c2': False, ...} from the judge's reply. True = PASS. Raises JudgeParseError on anything else:
    missing keys, extra keys, a value that is not PASS/FAIL, or text that is not valid JSON at all."""
    m = _JSON_RE.search(text or "")
    if not m:
        raise JudgeParseError(f"no JSON object found in the judge's reply: {text[:120]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"the judge's reply is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise JudgeParseError(f"the judge's JSON is not an object: {obj!r}")
    missing = [i for i in ids if i not in obj]
    extra = [k for k in obj if k not in ids]
    if missing or extra:
        raise JudgeParseError(f"expected keys {ids}, got {sorted(obj)} (missing {missing}, extra {extra})")
    out = {}
    for i in ids:
        v = str(obj[i]).strip().upper()
        if v not in ("PASS", "FAIL"):
            raise JudgeParseError(f"criterion {i} is {obj[i]!r}, must be PASS or FAIL")
        out[i] = v == "PASS"
    return out


def judge_answer(lab, ticket: str, truth: str, answer: str, *, rubric=RUBRIC, guide: str = GUIDE,
                 format_rule: str = DEFAULT_FORMAT_RULE, max_tokens: int = 1500) -> dict:
    """One ticket, one answer, graded. 'verdicts' is None (not raised) when the judge's reply could not be parsed,
    so one bad judge call does not stop the whole run - it shows up as a row to look at, not a crash."""
    r = lab.chat(judge_messages(ticket, truth, answer, rubric=rubric, guide=guide, format_rule=format_rule), max_tokens=max_tokens)
    try:
        verdicts = parse_verdict(r["text"], ids=[c["id"] for c in rubric])
        error = None
    except JudgeParseError as e:
        verdicts, error = None, str(e)
    return {"verdicts": verdicts, "all_pass": bool(verdicts) and all(verdicts.values()), "error": error,
            "raw": r["text"].strip(), "truncated": r["truncated"], "tokens": r["prompt_tokens"] + r["completion_tokens"]}


# ---------------------------------------------------------------- the order guard: who gets judged first, per ticket
def shuffle_order(items, seed: int = 0) -> list:
    """One deterministic coin flip per ticket (same every run, no hidden randomness): True means 'judge v1 first
    for this ticket', False means v2 first. Guards against a timing or caching quirk always favouring one prompt."""
    return [int(hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest(), 16) % 2 == 0 for text, _ in items]


# ---------------------------------------------------------------- scoring one prompt version over the golden set
def score_prompt(answer_lab, judge_lab, make_answer_messages, items=GOLDEN20, *, rubric=RUBRIC, guide: str = GUIDE,
                 format_rule: str = DEFAULT_FORMAT_RULE, max_answer_tokens: int = 1500) -> dict:
    """Answer every ticket with ONE prompt version, then judge every answer against the rubric."""
    rows = []
    for text, truth in items:
        ans = answer_with(answer_lab, make_answer_messages(text), max_tokens=max_answer_tokens)
        verdict = judge_answer(judge_lab, text, truth, ans["text"], rubric=rubric, guide=guide, format_rule=format_rule)
        rows.append({"ticket": text, "truth": truth, "answer": ans["text"], "answer_truncated": ans["truncated"],
                    "verdicts": verdict["verdicts"], "all_pass": verdict["all_pass"], "judge_error": verdict["error"],
                    "judge_raw": verdict["raw"]})
    return pass_rates(rows, rubric)


def pass_rates(rows: list, rubric=RUBRIC) -> dict:
    ids = [c["id"] for c in rubric]
    graded = [r for r in rows if r["verdicts"] is not None]
    per = {i: sum(1 for r in graded if r["verdicts"][i]) for i in ids}
    return {"rows": rows, "n": len(rows), "graded": len(graded), "per_criterion": per,
            "all_pass": sum(1 for r in graded if r["all_pass"])}


def compare_prompts(answer_lab, judge_lab, v1_maker, v2_maker, items=GOLDEN20, *, rubric=RUBRIC, guide: str = GUIDE,
                    format_rule: str = DEFAULT_FORMAT_RULE, seed: int = 0, max_answer_tokens: int = 1500) -> dict:
    """v1 and v2, the same twenty tickets, the same judge. The order the two are answered in is shuffled per
    ticket (shuffle_order); the result is the same pass rates either way - this guards against, not relies on, order."""
    first_is_v1 = shuffle_order(items, seed=seed)
    v1_rows, v2_rows = [None] * len(items), [None] * len(items)
    for idx, ((text, truth), v1_first) in enumerate(zip(items, first_is_v1)):
        order = (("v1", v1_maker), ("v2", v2_maker)) if v1_first else (("v2", v2_maker), ("v1", v1_maker))
        graded = {}
        for name, maker in order:
            ans = answer_with(answer_lab, maker(text), max_tokens=max_answer_tokens)
            verdict = judge_answer(judge_lab, text, truth, ans["text"], rubric=rubric, guide=guide, format_rule=format_rule)
            graded[name] = {"ticket": text, "truth": truth, "answer": ans["text"], "answer_truncated": ans["truncated"],
                           "verdicts": verdict["verdicts"], "all_pass": verdict["all_pass"], "judge_error": verdict["error"],
                           "judge_raw": verdict["raw"]}
        v1_rows[idx], v2_rows[idx] = graded["v1"], graded["v2"]
    return {"v1": pass_rates(v1_rows, rubric), "v2": pass_rates(v2_rows, rubric), "order": first_is_v1}


def pct(n: int, total: int) -> str:
    return f"{n}/{total} = {round(100 * n / total) if total else 0}%"


# ---------------------------------------------------------------- the human spot-check
def spot_check_items(items=GOLDEN20, k: int = 5, seed: int = 0) -> list:
    """k tickets chosen the same deterministic way as pick_examples: same every run, no hidden randomness."""
    ranked = sorted(items, key=lambda it: hashlib.sha256(f"{seed}|{it[0]}".encode("utf-8")).hexdigest())
    return ranked[:k]


def agreement(human: dict, judge: dict) -> dict:
    """human and judge: {criterion_id: True/False, ...} for ONE ticket's ONE answer. Returns which ids agree."""
    ids = sorted(set(human) & set(judge))
    agree = [i for i in ids if human[i] == judge[i]]
    return {"ids": ids, "agree": agree, "disagree": [i for i in ids if i not in agree], "n": len(ids), "agreed": len(agree)}


def agreement_rate(pairs: list) -> dict:
    """pairs: [(human_verdicts, judge_verdicts), ...], one per spot-checked ticket. Agreement is counted
    criterion by criterion (k tickets x 4 criteria), not ticket by ticket."""
    total = agreed = 0
    for human, judge in pairs:
        a = agreement(human, judge)
        total += a["n"]
        agreed += a["agreed"]
    return {"agreed": agreed, "total": total, "rate": pct(agreed, total)}


# ---------------------------------------------------------------- the report you commit
def eval_markdown(v1: dict, v2: dict, model: str, judge_model: str, *, rubric=RUBRIC, spot_check: dict = None,
                  today: str = None) -> str:
    today = today or datetime.date.today().isoformat()
    lines = ["# Session 17: prompt evaluation - v1 baseline vs v2 structured", "",
             f"Answering model: `{model}` · Judge model: `{judge_model}` · Date: {today}", "",
             "| Criterion | v1 (S14 baseline) | v2 (S16 structured) |", "|---|---|---|"]
    for c in rubric:
        lines.append(f"| {c['name']} | {pct(v1['per_criterion'][c['id']], v1['n'])} | {pct(v2['per_criterion'][c['id']], v2['n'])} |")
    lines.append(f"| **All four criteria** | **{pct(v1['all_pass'], v1['n'])}** | **{pct(v2['all_pass'], v2['n'])}** |")
    lines += ["", f"Tickets graded: v1 {v1['graded']}/{v1['n']} · v2 {v2['graded']}/{v2['n']} (a ticket the judge could not "
             "parse is not counted, and is listed below)."]
    unparsed = [r["ticket"][:70] for r in v1["rows"] + v2["rows"] if r["verdicts"] is None]
    if unparsed:
        lines += ["", "Tickets the judge's reply could not be parsed for:", ""] + [f"- {t}" for t in unparsed]
    if spot_check is not None:
        lines += ["", f"Human spot-check: judge and human agreed on {spot_check['rate']} of graded criteria "
                 f"({spot_check['agreed']}/{spot_check['total']})."]
    return "\n".join(lines) + "\n"


def save_eval_report(repo_path, text: str) -> Path:
    target = Path(repo_path) / REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


_TABLE_RE = re.compile(r"\|\s*\*\*All four criteria\*\*\s*\|\s*\*\*(\d+)/(\d+) = \d+%\*\*\s*\|\s*\*\*(\d+)/(\d+) = \d+%\*\*\s*\|")
_SPOT_RE = re.compile(r"Human spot-check: judge and human agreed on \d+/(\d+) = \d+% of graded criteria \((\d+)/(\d+)\)")


def parse_eval_report(text: str):
    """(v1_all_pass, v1_n, v2_all_pass, v2_n) from the 'All four criteria' row, or None if the report does not have it."""
    m = _TABLE_RE.search(text or "")
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def parse_spot_check(text: str):
    """(agreed, total) from the report's spot-check line, or None if it is missing."""
    m = _SPOT_RE.search(text or "")
    if not m:
        return None
    return int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------- is my work complete? (offline: no model call)
def check_s17_work(repo_path, min_spot_check: int = 5) -> list:
    """Plain-English checks on your work repository for Session 17. Each failure carries the fix."""
    repo = Path(repo_path)

    def ck(name, ok, fix="", info=""):
        return CheckResult(name, ok, info if ok else fix)
    if not repo.exists():
        return [ck("work repo folder exists", False, f"{repo} not found. Fix WORK_REPO.")]
    out = [ck("work repo folder exists", True)]

    try:
        load_prompt_file(repo / SUPPORT_DIR / SYSTEM_FILE)
        load_prompt_file(repo / SUPPORT_DIR / USER_FILE)
        out.append(ck("Session 16 prompt files are still there (system.md, user.j2)", True))
    except PromptFileError as e:
        out.append(ck("Session 16 prompt files are still there (system.md, user.j2)", False,
                      f"{e} Session 17 grades v2 from these same files - do not delete or move them."))

    report_file = repo / REPORT_PATH
    if not report_file.exists():
        out.append(ck("evaluation report exists", False, f"{REPORT_PATH} not found. Run notebook section 12 with SAVE_REPORT = True."))
        return out
    out.append(ck("evaluation report exists", True))
    text = report_file.read_text(encoding="utf-8")
    parsed = parse_eval_report(text)
    if parsed is None:
        out.append(ck("report states pass rates for both prompts", False,
                      "No 'All four criteria' row found. Re-save the report from section 12."))
        return out
    v1_pass, v1_n, v2_pass, v2_n = parsed
    out.append(ck("report states pass rates for both prompts", True))
    out.append(ck("report covers all twenty tickets for both prompts", v1_n == v2_n == len(GOLDEN20),
                  f"v1 counted {v1_n}, v2 counted {v2_n}, golden set is {len(GOLDEN20)}"))
    spot = parse_spot_check(text)
    if spot is None:
        out.append(ck(f"human spot-check of at least {min_spot_check} tickets is recorded", False,
                      "No spot-check line found. Run notebook section 13 (grade a handful yourself) before saving the report."))
    else:
        agreed, total = spot
        n_tickets = total // len(RUBRIC_IDS) if total else 0
        out.append(ck(f"human spot-check of at least {min_spot_check} tickets is recorded", n_tickets >= min_spot_check,
                      f"only {n_tickets} tickets recorded", info=f"{n_tickets} tickets, {agreed}/{total} criteria agreed"))
    return out
