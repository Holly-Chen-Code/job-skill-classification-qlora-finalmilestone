"""Shared helper functions for model inference."""

import re


def clean_prediction(text: str) -> str:
    """Return one clean skill-category label from generated text."""
    text = str(text).strip()
    text = re.sub(
        r"^(skill category|skill_name|skill name|answer|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "[empty]"

    result = lines[0].strip(" \t\n\r\"'`.,;:")
    return result or "[empty]"
