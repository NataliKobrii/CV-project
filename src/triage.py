"""Triage output: the mapping from state to priority and color, and the
operator summary line."""
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import STATE_COLORS, STATE_PRIORITY  # noqa: E402


def color_for(state):
    """Return the display color of a state."""
    return STATE_COLORS.get(state, STATE_COLORS["unknown"])


def priority_of(state):
    """Return the rescue priority of a state; a lower value is more urgent."""
    return STATE_PRIORITY.get(state, 99)


def summarize(states):
    """Turn a list of states into the operator summary line, ordered by
    priority, for example '1 motionless · 3 mobile'."""
    # Unknown states are omitted so the operator summary lists only
    # actionable people.
    counts = Counter(s for s in states if s and s != "unknown")
    parts = [f"{counts[s]} {s}" for s in sorted(counts, key=priority_of)]
    return " · ".join(parts) if parts else "no people"


def priority_list(track_states):
    """Sort a {track_id: state} mapping into a list with the most urgent
    tracks first."""
    return sorted(track_states.items(), key=lambda kv: (priority_of(kv[1]), kv[0]))
