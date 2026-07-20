"""Convert VisDrone-DET annotations to a YOLO-format person-only dataset.

VisDrone annotation line: x,y,w,h,score,category,truncation,occlusion
Person categories: 1 = pedestrian, 2 = people. Category 0 is an "ignored
region" and score==0 in *train/val ground truth* means the box should be
ignored, so both are skipped.

Images are symlinked (not copied) to save disk.
"""
from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, VISDRONE_PERSON, VISDRONE_TRAIN, VISDRONE_VAL  # noqa: E402

PERSON_CATS = {1, 2}


def convert_split(src_dir: Path, split: str, out_root: Path = VISDRONE_PERSON):
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img, n_box = 0, 0
    for ann in sorted((src_dir / "annotations").glob("*.txt")):
        img_path = src_dir / "images" / (ann.stem + ".jpg")
        if not img_path.exists():
            continue
        w, h = Image.open(img_path).size
        lines = []
        for row in ann.read_text().splitlines():
            parts = row.strip().strip(",").split(",")
            if len(parts) < 6:
                continue
            x, y, bw, bh, score, cat = map(int, parts[:6])
            if cat not in PERSON_CATS or score == 0 or bw <= 0 or bh <= 0:
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            if not (0 < cx < 1 and 0 < cy < 1):
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {min(nw,1):.6f} {min(nh,1):.6f}")
        (lbl_out / (ann.stem + ".txt")).write_text("\n".join(lines))
        link = img_out / img_path.name
        if not link.exists():
            link.symlink_to(img_path.resolve())
        n_img += 1
        n_box += len(lines)
    print(f"{split}: {n_img} images, {n_box} person boxes -> {out_root}")
    return n_img, n_box


def write_dataset_yaml(out_root: Path = VISDRONE_PERSON):
    yaml_path = DATA_DIR / "visdrone_person.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: person\n"
    )
    print("wrote", yaml_path)
    return yaml_path


if __name__ == "__main__":
    convert_split(VISDRONE_TRAIN, "train")
    convert_split(VISDRONE_VAL, "val")
    write_dataset_yaml()
