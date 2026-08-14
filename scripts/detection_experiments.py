"""Local detection experiments, the small-scale version of notebook 01.

First we measure the zero-shot person AP50 of the COCO-pretrained yolo11n
on the VisDrone validation set, which is the domain-gap baseline. Then we
run a short fine-tune on a fraction of the training split to prove the
training path. Finally, we re-evaluate the fine-tuned weights with the same
custom evaluator.

The full-scale fine-tuning is saved in notebooks/01_detection_finetune.ipynb.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import DATA_DIR, MODELS_DIR  # noqa: E402
from eval_detect import append_csv, evaluate_on_visdrone  # noqa: E402


def main(epochs=3, fraction=0.25, imgsz=960, batch=8, device="mps"):
    """Run the baseline evaluation, the short fine-tune and the re-evaluation."""
    # The zero-shot baseline.
    row = evaluate_on_visdrone("yolo11n.pt", tag="yolo11n zero-shot (COCO)",
                               imgsz=1280, device=device)
    append_csv(row)

    # A short fine-tune.
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(data=str(DATA_DIR / "visdrone_person.yaml"),
                epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction,
                device=device, project=str(MODELS_DIR),
                name="yolo11n_visdrone_ft", exist_ok=True,
                workers=4, plots=False, verbose=False)
    best = MODELS_DIR / "yolo11n_visdrone_ft" / "weights" / "best.pt"

    # Evaluate the fine-tuned weights with the same evaluator.
    row = evaluate_on_visdrone(str(best),
                               tag=f"yolo11n fine-tuned ({epochs} epochs, {fraction:.0%} of train)",
                               imgsz=1280, device=device)
    append_csv(row)


if __name__ == "__main__":
    main()
