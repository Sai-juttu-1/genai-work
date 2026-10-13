"""Instructor tool: who pushed the Session 16 work? A system prompt file, a ticket template and a held-out report of at least 16/20.

    uv run python module02/check_s16_submissions.py submissions.txt

submissions.txt is the same file as for Sessions 14 and 15 (one GitHub repository URL per line, or a CSV exported from your form).
Repositories must be PUBLIC. It reads three files straight from GitHub (branch main, then master) and writes s16_report.csv.
It does NOT call a model: the accuracy it reports is the one in the student's own report, so a pre-run of their files is a separate step."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_s15_submissions import fetch_from_branches  # noqa: E402
from check_submissions import fetch_text, parse_repo, read_lines  # noqa: E402
from structured_utils import parse_heldout_report, render_check  # noqa: E402

BASE = "prompts/support_classifier"


def _split_front(text: str) -> str:
    """The prompt body without the optional '---' block."""
    t = text.lstrip("\ufeff").replace("\r\n", "\n")
    if t.startswith("---"):
        parts = t.split("\n---", 1)
        if len(parts) == 2:
            return parts[1].lstrip("-").strip()
    return t.strip()


def check_repo_s16(line: str, fetch=fetch_text, need: int = 16, n: int = 20) -> dict:
    parsed = parse_repo(line)
    if not parsed:
        return {"repo": line.strip(), "status": "NOT A GITHUB URL", "accuracy": "", "renders": ""}
    owner, repo = parsed
    system = fetch_from_branches(owner, repo, f"{BASE}/system.md", fetch)
    template = fetch_from_branches(owner, repo, f"{BASE}/user.j2", fetch)
    report = fetch_from_branches(owner, repo, "experiments/s16_heldout.md", fetch)
    problems, renders, accuracy = [], "", ""
    if system is None:
        problems.append("NO system.md")
    elif len(_split_front(system)) < 400:
        problems.append("SYSTEM PROMPT TOO THIN")
    if template is None:
        problems.append("NO user.j2")
    elif system is not None:
        try:
            render_check(_split_front(system), _split_front(template))
            renders = "yes"
        except Exception as e:  # noqa: BLE001 - any template error is a finding, not a crash
            renders = "no"
            problems.append(f"TEMPLATE DOES NOT RENDER ({type(e).__name__})")
        if "<ticket" not in template or "</ticket" not in template:
            problems.append("TICKET NOT DELIMITED")
    if report is None:
        problems.append("NO held-out report")
    else:
        got = parse_heldout_report(report)
        if got is None:
            problems.append("REPORT HAS NO ACCURACY LINE")
        else:
            right, total, rows = got
            accuracy = f"{right}/{total}"
            if total != n or rows != n:
                problems.append(f"REPORT COVERS {rows} TICKETS")
            if right < need:
                problems.append(f"BELOW {need}/{n}")
    return {"repo": f"{owner}/{repo}", "status": "PASS" if not problems else "; ".join(problems), "accuracy": accuracy, "renders": renders}


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    results = [check_repo_s16(line) for line in read_lines(Path(argv[1]))]
    print(f"{'repository':42} {'held-out':>8} {'renders':>8}  status")
    for r in results:
        print(f"{r['repo'][:42]:42} {r['accuracy']:>8} {r['renders']:>8}  {r['status']}")
    print(f"\n{sum(r['status'] == 'PASS' for r in results)}/{len(results)} pass")
    with open("s16_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "status", "accuracy", "renders"])
        w.writeheader()
        w.writerows(results)
    print("wrote s16_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
