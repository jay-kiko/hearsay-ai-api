"""Small pieces shared by the generation modules, kept in one place so they
can't drift apart."""
from __future__ import annotations

from app.models import PersonaIn

# Used when detect gave no buyer context: deliberately generic, so it never
# assumes the choice is B2B software.
DEFAULT_BUYER_CONTEXT = "People and organizations choosing what to use, buy, or work with in this industry."


def clean_str_list(raw: object) -> list[str]:
    """A tool-call field that should be a list of strings. Forced tool calls
    have returned these as one comma-joined string, and iterating a bare
    string would shred it per character, so anything but a list is empty."""
    if not isinstance(raw, list):
        return []
    return [v for v in raw if isinstance(v, str)]


def persona_payload(personas: list[PersonaIn]) -> list[dict[str, str]]:
    """How a persona is shown to the prompt writer."""
    return [
        {
            "personaId": p.id,
            "title": p.title,
            "description": p.desc,
            "role": p.role,
            "pains": p.pains,
            "criteria": p.criteria,
        }
        for p in personas
    ]
