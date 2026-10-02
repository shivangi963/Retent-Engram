"""
frontend/components/parsers.py
==============================
Turns the raw text returned by the local LLM into structured pieces
the UI can render. Shared by the Dashboard preview and the Study page.
"""
import re


def plain(text: str) -> str:
    """Strip markdown emphasis and bullet characters."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.replace("•", "").replace("*", "").strip()


def parse_flashcard(content: str) -> tuple:
    """Returns (question, answer) from 'QUESTION: ... ANSWER: ...' text."""
    q_lines, a_lines, section = [], [], None
    for line in content.split("\n"):
        s = line.strip()
        up = s.upper()
        if up.startswith("QUESTION:"):
            section = "Q"
            if s[9:].strip():
                q_lines.append(s[9:].strip())
        elif up.startswith("ANSWER:"):
            section = "A"
            if s[7:].strip():
                a_lines.append(s[7:].strip())
        elif s and section == "Q":
            q_lines.append(s)
        elif s and section == "A":
            a_lines.append(s)
    question = " ".join(q_lines) if q_lines else content.strip()
    answer = " ".join(a_lines) if a_lines else ""
    return question, answer


def summary_body(content: str) -> str:
    """Summary text without the 'SUMMARY:' header line."""
    lines = [l for l in content.split("\n") if not l.strip().upper().startswith("SUMMARY:")]
    return "\n".join(lines).strip()


def summary_blurb(content: str, max_chars: int = 230) -> str:
    """One short paragraph from a summary (for dashboard previews)."""
    bullets = [plain(l) for l in summary_body(content).split("\n") if l.strip()]
    text = " ".join(bullets)
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"


def parse_quiz(content: str) -> list:
    """Returns [{question, options, correct, explanation}]."""
    questions = []
    for block in re.split(r"Q\d+:", content)[1:]:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines:
            continue
        q = {"question": lines[0], "options": [], "correct": "", "explanation": ""}
        for line in lines[1:]:
            if re.match(r"^[A-D]\)", line):
                q["options"].append(line)
            elif line.upper().startswith("CORRECT:"):
                q["correct"] = line.split(":", 1)[1].strip().upper()[:1]
            elif line.upper().startswith("EXPLANATION:"):
                q["explanation"] = line.split(":", 1)[1].strip()
        if q["question"] and q["options"]:
            questions.append(q)
    return questions


def parse_coding(content: str) -> dict:
    """Returns {'PROBLEM':..., 'INPUT':..., 'EXPECTED OUTPUT':..., 'HINT':...}."""
    sections, current, lines = {}, None, []
    for line in content.split("\n"):
        s = line.strip()
        up = s.upper()
        if up.startswith("CODING TASK:"):
            continue
        matched = None
        for kw in ["PROBLEM:", "INPUT:", "EXPECTED OUTPUT:", "HINT:"]:
            if up.startswith(kw):
                if current:
                    sections[current] = "\n".join(lines).strip()
                current = kw.rstrip(":")
                lines = [s[len(kw):].strip()]
                matched = kw
                break
        if matched is None and current is not None:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    return sections
