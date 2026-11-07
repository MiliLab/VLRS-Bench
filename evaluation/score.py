#!/usr/bin/env python3
"""Score VLRS-Bench predictions using the paper's evaluation protocol."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TASKS = (
    "CR", "CFR", "SIR", "MIR", "ST-CFR", "ST-CCR", "ST-ER", "ST-CR",
    "PR", "ER", "ST-CS-PR", "ST-M-PR", "ST-SU-PR", "ST-SQ-PR",
)
DIMENSIONS = {
    "SC": TASKS[:4], "ST-C": TASKS[4:8], "Cognition": TASKS[:8],
    "Pre-D": ("PR",), "Post-D": ("ER",), "Decision": TASKS[8:10],
    "OP": TASKS[10:12], "SP": TASKS[12:], "Prediction": TASKS[10:],
}
OPTION_SET = re.compile(r"[A-H](?:\s*[,;，、；\s]\s*[A-H])*", re.I)
BLANK = re.compile(r"\bBLANK\s*(\d+)\s*:\s*(.*?)(?=\bBLANK\s*\d+\s*:|$)", re.I | re.S)
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def choices(text):
    text = str(text).strip().rstrip(".").strip()
    return set(re.findall(r"[A-H]", text.upper())) if OPTION_SET.fullmatch(text) else set()


def blanks(text):
    text = str(text).strip()
    labeled = BLANK.findall(text)
    if labeled:
        return {int(i): value.strip(" ;；,，\n") for i, value in labeled}
    return {
        i: value.strip()
        for i, value in enumerate(re.split(r"[;；]", text), 1)
        if value.strip()
    }


def question_type(item, answer):
    tag = str(item.get("qa_type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single_choice": "single_choice", "single_answer": "single_choice", "sc": "single_choice",
        "multi_choice": "multi_choice", "multiple_choice": "multi_choice", "mc": "multi_choice",
        "multi_answer": "multi_choice", "multi_select": "multi_choice",
        "true_false": "true_false", "true/false": "true_false", "yes_or_no": "true_false", "tf": "true_false",
        "fill_blank": "fill_blank", "fill_in_the_blank": "fill_blank", "fb": "fill_blank",
    }
    if tag in aliases:
        return aliases[tag]
    if answer.casefold().strip().rstrip(".") in {"true", "false"}:
        return "true_false"
    options = choices(answer)
    return "multi_choice" if len(options) > 1 else "single_choice" if options else "fill_blank"


def item_score(kind, reference, prediction, similarity):
    if kind in {"single_choice", "multi_choice"}:
        correct, selected = choices(reference), choices(prediction)
        if correct and selected == correct:
            return 1.0
        return 0.5 if kind == "multi_choice" and selected and selected < correct else 0.0
    if kind == "true_false":
        normalize = lambda text: str(text).strip().rstrip(".").casefold()
        return float(normalize(reference) == normalize(prediction))
    expected, supplied = blanks(reference), blanks(prediction)
    correct = sum(
        bool(supplied.get(i)) and similarity(phrase, supplied[i]) > 0.8
        for i, phrase in expected.items()
    )
    if correct:
        return 1.0 if correct == len(expected) else 0.5
    return 0.0


def evaluate(items, predictions, supported, similarity):
    task_scores, qa_scores = defaultdict(list), defaultdict(list)
    seen = set()
    for item in items:
        key = str(item["id"])
        if key in seen:
            raise ValueError(f"Duplicate benchmark ID: {key}")
        seen.add(key)
        task = str(item["job"]).upper().replace("_", "-")
        if task not in TASKS:
            raise ValueError(f"Unknown benchmark task: {task}")
        if task not in supported:
            continue
        reference = next(m["value"] for m in item["conversations"] if m["from"] == "gpt")
        kind = question_type(item, reference)
        score = item_score(kind, reference, predictions[key], similarity) if key in predictions else 0.0
        task_scores[task].append(score)
        qa_scores[kind].append(score)
    if unknown := predictions.keys() - seen:
        raise ValueError(f"Unknown prediction ID: {min(unknown)}")
    if missing_tasks := supported - task_scores.keys():
        raise ValueError(f"Supported tasks absent from benchmark: {sorted(missing_tasks)}")

    def aggregate(scores):
        return {"n": len(scores), "points": sum(scores), "mean": sum(scores) / len(scores) if scores else None}

    def collect(tasks):
        return [s for task in tasks for s in task_scores.get(task, ())]

    return {
        "overall": aggregate(collect(TASKS)),
        "by_task": {task: aggregate(collect([task])) for task in TASKS},
        "by_ability_or_dimension": {dim: aggregate(collect(tasks)) for dim, tasks in DIMENSIONS.items()},
        "by_question_type": {kind: aggregate(scores) for kind, scores in sorted(qa_scores.items())},
    }


class MiniLM:
    def __init__(self):
        self.model = None
        self.embeddings = {}

    def __call__(self, left, right):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(ENCODER)
        missing = [s for s in dict.fromkeys([left, right]) if s not in self.embeddings]
        if missing:
            embeddings = self.model.encode(missing, normalize_embeddings=True)
            self.embeddings.update(zip(missing, embeddings))
        return float(self.embeddings[left] @ self.embeddings[right])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Released VLRS-Bench.json")
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with id and answer")
    parser.add_argument("--supported-tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    predictions = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["id"])
        if key in predictions:
            parser.error(f"Duplicate prediction ID: {key}")
        predictions[key] = row["answer"]
    items = json.loads(args.benchmark.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(items, predictions, set(args.supported_tasks), MiniLM()), indent=2))


if __name__ == "__main__":
    main()
