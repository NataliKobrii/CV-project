"""Local detection experiments (smoke-scale versions of notebook 01).

1. Zero-shot COCO yolo11n person AP50 on VisDrone val  (the domain-gap baseline)
2. Short fine-tune on a fraction of VisDrone-person train (proves training path)
3. Re-evaluate fine-tuned weights with the SAME custom evaluator

Full-scale fine-tuning lives in notebooks/01_detection_finetune.ipynb.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import DATA_DIR, MODELS_DIR  # noqa: E402
from eval_detect import append_csv, evaluate_on_visdrone  # noqa: E402


def main(epochs=3, fraction=0.25, imgsz=960, batch=8, device="mps"):
    # 1 — zero-shot baseline
    row = evaluate_on_visdrone("yolo11n.pt", tag="yolo11n zero-shot (COCO)",
                               imgsz=1280, device=device)
    append_csv(row)

    # 2 — mini fine-tune
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(data=str(DATA_DIR / "visdrone_person.yaml"),
                epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction,
                device=device, project=str(MODELS_DIR),
                name="yolo11n_visdrone_ft", exist_ok=True,
                workers=4, plots=False, verbose=False)
    best = MODELS_DIR / "yolo11n_visdrone_ft" / "weights" / "best.pt"

    # 3 — fine-tuned eval, same evaluator
    row = evaluate_on_visdrone(str(best),
                               tag=f"yolo11n fine-tuned ({epochs}ep, {fraction:.0%} train)",
                               imgsz=1280, device=device)
    append_csv(row)


if __name__ == "__main__":
    main()
