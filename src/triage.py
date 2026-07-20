"""Triage output: state -> priority, colors, and the operator summary line."""
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import STATE_COLORS, STATE_PRIORITY  # noqa: E402


def color_for(state):
    return STATE_COLORS.get(state, STATE_COLORS["unknown"])


def priority_of(state):
    return STATE_PRIORITY.get(state, 99)


def summarize(states):
    """['mobile','motionless',...] -> '1 motionless · 3 mobile' (priority order)."""
    counts = Counter(s for s in states if s and s != "unknown")
    parts = [f"{counts[s]} {s}" for s in sorted(counts, key=priority_of)]
    return " · ".join(parts) if parts else "no people"


def priority_list(track_states):
    """{track_id: state} -> [(track_id, state)] sorted most-urgent-first."""
    return sorted(track_states.items(), key=lambda kv: (priority_of(kv[1]), kv[0]))
