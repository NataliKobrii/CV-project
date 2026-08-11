"""A parser for the VATIC-style annotation files of Okutama-Action.

Each line is whitespace-separated, and the quoted attributes may contain
spaces:
    track_id xmin ymin xmax ymax frame lost occluded generated "Person" "Action1" ["Action2" ...]
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import resolve_state  # noqa: E402

_QUOTED = re.compile(r'"([^"]*)"')


@dataclass
class OkutamaBox:
    """One annotated person box in one frame of an Okutama video."""
    track_id: int
    frame: int
    box: tuple  # (x1, y1, x2, y2)
    lost: bool
    occluded: bool
    actions: tuple
    state: str  # The triage state resolved from the actions.


def parse_annotations(txt_path, drop_lost=True):
    """Parse the file and return a dictionary from frame number to boxes,
    sorted by frame."""
    frames = defaultdict(list)
    for line in Path(txt_path).read_text().splitlines():
        if not line.strip():
            continue
        # The line begins with numeric fields and ends with quoted strings,
        # so we split on the first quote to separate the two parts.
        head = line.split('"')[0].split()
        if len(head) < 9:
            continue
        tid, x1, y1, x2, y2, frame, lost, occluded, _generated = map(int, head[:9])
        if drop_lost and lost:
            continue
        # The quoted attributes at the end of the line hold the action labels.
        attrs = _QUOTED.findall(line)
        actions = tuple(a for a in attrs if a != "Person")
        frames[frame].append(
            OkutamaBox(
                track_id=tid,
                frame=frame,
                box=(x1, y1, x2, y2),
                lost=bool(lost),
                occluded=bool(occluded),
                actions=actions,
                state=resolve_state(actions),
            )
        )
    return dict(sorted(frames.items()))


def tracks_from_frames(frames):
    """Reindex the frame dictionary into one list of boxes per track, sorted
    by frame."""
    tracks = defaultdict(list)
    for boxes in frames.values():
        for b in boxes:
            tracks[b.track_id].append(b)
    for tid in tracks:
        tracks[tid].sort(key=lambda b: b.frame)
    return dict(tracks)


def state_distribution(frames):
    """Count the boxes per triage state, for the dataset statistics
    tables."""
    counts = defaultdict(int)
    for boxes in frames.values():
        for b in boxes:
            counts[b.state] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    import json
    from config import OKUTAMA_DIR

    for txt in sorted(OKUTAMA_DIR.glob("*.txt")):
        frames = parse_annotations(txt)
        n_boxes = sum(len(v) for v in frames.values())
        print(f"{txt.name}: {len(frames)} frames, {n_boxes} boxes, "
              f"{len(tracks_from_frames(frames))} tracks")
        print("  states:", json.dumps(state_distribution(frames)))
