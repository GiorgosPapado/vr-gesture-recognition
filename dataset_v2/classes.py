"""Canonical class table for VRGestures v2 (24 classes) and name normalisation."""
import re

GESTURES_PER_HAND = ["Click", "Close", "Grab", "Home", "Infinity", "OpenPalm", "Push", "Rotate", "Point", "Scroll", "Back"]
CLASSES = (
    [f"Left{g}" for g in GESTURES_PER_HAND]
    + [f"Right{g}" for g in GESTURES_PER_HAND]
    + ["Teleportation", "Zoom"]
)
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def hand_of(canonical):
    if canonical.startswith("Left"):
        return "left"
    if canonical.startswith("Right"):
        return "right"
    return "both"


def _variants(c):
    # explicit spellings accepted for each canonical name: exact, lower/upper case, and with a space,
    # underscore or hyphen between the hand and the gesture
    out = {c, c.lower(), c.upper()}
    m = re.match(r"^(Left|Right)(.+)$", c)
    if m:
        for sep in (" ", "_", "-"):
            out.add(f"{m[1]}{sep}{m[2]}")
            out.add(f"{m[1]}{sep}{m[2]}".lower())
    return out


NAME_TABLE = {v: c for c in CLASSES for v in _variants(c)}
NAME_TABLE.update({v + ".json": c for v, c in list(NAME_TABLE.items())})  # loose sample files: '<Class>.json'


def normalise(token):
    """Return (canonical_name or None, is_exact). ``is_exact`` False means a variant spelling was mapped."""
    c = NAME_TABLE.get(token)
    if c is None:
        c = NAME_TABLE.get(token.lower())
    return c, c is not None and token == c
