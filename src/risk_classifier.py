from typing import Dict
from transformers import pipeline

# Loading a zero-shot classification model once at import time
# (facebook/bart-large-mnli is widely used for zero-shot)
_risk_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

RISK_LABELS = ["Low risk", "Medium risk", "High risk"]


def classify_legal_risk(text: str) -> Dict[str, float]:
    """
    Classify a piece of legal text into risk levels:
    Low / Medium / High using zero-shot classification.

    Returns a dict mapping label -> score, normalized to sum ~1.
    """
    if not text or not text.strip():
        return {label: 0.0 for label in RISK_LABELS}

    result = _risk_classifier(
        text,
        candidate_labels=RISK_LABELS,
        multi_label=False,  # picks a single best label
    )

    # result["labels"] is ordered by descending score
    labels = result["labels"]
    scores = result["scores"]

    # Normalizing to a dict with consistent label ordering
    scores_dict = {label: 0.0 for label in RISK_LABELS}
    for label, score in zip(labels, scores):
        if label in scores_dict:
            scores_dict[label] = float(score)

    # Optional: re-normalize to sum to 1.0
    total = sum(scores_dict.values()) or 1.0
    scores_dict = {k: v / total for k, v in scores_dict.items()}

    return scores_dict

import re
from typing import List, Dict

# ... existing RISK_LABELS and classify_legal_risk above ...


SECTION_PATTERN = re.compile(
    r"(SECTION\s+\d+[A-Za-z0-9.\-]*.*?)(?=SECTION\s+\d+[A-Za-z0-9.\-]*|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Splitting a legal document into sections based on 'SECTION <number>' patterns.
    If no sections are detected, fall back to a single pseudo-section.

    Returns a list of dicts: [{"title": "...", "text": "..."}, ...]
    """
    if not text or not text.strip():
        return []

    sections = []
    matches = list(SECTION_PATTERN.finditer(text))

    if matches:
        for match in matches:
            block = match.group(1).strip()

            # First line is the title, rest is body
            lines = block.splitlines()
            if not lines:
                continue

            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                # If no body, still keep entire block as text
                body = block

            sections.append(
                {
                    "title": title,
                    "text": body,
                }
            )
    else:
        # Fallback: treat the entire text as one section
        sections.append(
            {
                "title": "Full Document",
                "text": text.strip(),
            }
        )

    return sections


def classify_legal_risk_sections(text: str) -> List[Dict[str, object]]:
    """
    Runs legal risk classification per detected section.
    Returns a list of:
      {
        "title": str,
        "text": str,
        "top_label": str,
        "scores": {label: float}
      }
    """
    raw_sections = split_into_sections(text)
    results = []

    for section in raw_sections:
        sec_text = section["text"]
        scores = classify_legal_risk(sec_text)
        top_label = max(scores.items(), key=lambda x: x[1])[0]

        results.append(
            {
                "title": section["title"],
                "text": sec_text,
                "top_label": top_label,
                "scores": scores,
            }
        )

    return results
