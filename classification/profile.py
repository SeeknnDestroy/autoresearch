"""Prompt profile loading for the classification lane."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .banking77 import canonicalize_label_name, humanize_label


def resolve_default_profile_path() -> Path:
    env_path = os.environ.get("AUTORESEARCH_CLASSIFICATION_PROFILE")
    if env_path:
        return Path(env_path).expanduser()
    candidates = [
        Path.cwd() / "classification_profile.json",
        Path(__file__).resolve().parent.parent / "classification_profile.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_PROFILE_PATH = resolve_default_profile_path()


@dataclass(frozen=True)
class FewShotExample:
    text: str
    label: str


@dataclass(frozen=True)
class ClassificationProfile:
    model: str
    system_prompt: str
    user_prompt_template: str
    options: dict[str, Any]
    few_shot_examples: tuple[FewShotExample, ...]
    label_descriptions: dict[str, str]
    label_order: str | list[str]

    def ordered_labels(self, labels: list[str]) -> list[str]:
        if self.label_order == "alphabetical":
            return sorted(labels)
        if isinstance(self.label_order, list):
            canonical_order = [canonicalize_label_name(label) for label in self.label_order]
            ordered = [label for label in canonical_order if label in labels]
            remainder = [label for label in sorted(labels) if label not in ordered]
            return ordered + remainder
        raise ValueError(f"Unsupported label_order value: {self.label_order!r}")

    def labels_block(self, labels: list[str]) -> str:
        lines = []
        for label in self.ordered_labels(labels):
            description = self.label_descriptions.get(label, humanize_label(label))
            lines.append(f"- {label}: {description}")
        return "\n".join(lines)

    def few_shot_block(self) -> str:
        if not self.few_shot_examples:
            return "No few-shot examples."
        blocks = []
        for example in self.few_shot_examples:
            blocks.append(f'Text: "{example.text}"\nLabel: {example.label}')
        return "\n\n".join(blocks)

    def render_prompt(self, text: str, labels: list[str]) -> str:
        return self.user_prompt_template.format(
            labels_block=self.labels_block(labels),
            few_shot_block=self.few_shot_block(),
            text=text,
        )


def load_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> ClassificationProfile:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    few_shot_examples = tuple(
        FewShotExample(
            text=(item["text"]).strip(),
            label=canonicalize_label_name(item["label"]),
        )
        for item in data.get("few_shot_examples", [])
    )
    label_descriptions = {
        canonicalize_label_name(key): value.strip()
        for key, value in data.get("label_descriptions", {}).items()
    }
    return ClassificationProfile(
        model=data["model"],
        system_prompt=data["system_prompt"].strip(),
        user_prompt_template=data["user_prompt_template"],
        options=dict(data.get("options", {})),
        few_shot_examples=few_shot_examples,
        label_descriptions=label_descriptions,
        label_order=data.get("label_order", "alphabetical"),
    )
