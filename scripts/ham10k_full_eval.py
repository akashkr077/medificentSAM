import argparse
import csv
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def extract_masks(mask_zip: Path, work_dir: Path) -> Path:
    mask_root = work_dir / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(mask_zip) as zf:
        zf.extractall(mask_root)

    candidates = list(mask_root.glob("**/*_segmentation.png"))
    if not candidates:
        raise FileNotFoundError(f"No *_segmentation.png files found in {mask_zip}")
    return candidates[0].parent


def find_image_zip(ham10k_zip: Path, work_dir: Path) -> Path:
    with zipfile.ZipFile(ham10k_zip) as zf:
        nested = [name for name in zf.namelist() if name.endswith(".zip")]
        if nested:
            out_dir = work_dir / "raw"
            out_dir.mkdir(parents=True, exist_ok=True)
            zf.extract(nested[0], out_dir)
            return out_dir / nested[0]
    return ham10k_zip


def make_inputs(image_zip: Path, mask_dir: Path, input_dir: Path, limit: int | None) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with zipfile.ZipFile(image_zip) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if limit is not None:
            names = names[:limit]

        for name in names:
            image_id = Path(name).stem
            gt_path = mask_dir / f"{image_id}_segmentation.png"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_id}: {gt_path}")

            out_path = input_dir / f"2DBox_HAM10K_{image_id}.npz"
            if out_path.exists():
                count += 1
                continue

            with zf.open(name) as fp:
                img = Image.open(io.BytesIO(fp.read())).convert("RGB")
            arr = np.asarray(img)
            h, w = arr.shape[:2]

            mask = Image.open(gt_path).convert("L")
            if mask.size != (w, h):
                mask = mask.resize((w, h), Image.Resampling.NEAREST)
            mask_np = np.asarray(mask) > 0
            ys, xs = np.where(mask_np)
            if len(xs) == 0:
                box = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
            else:
                pad = 5
                box = np.array(
                    [
                        [
                            max(0, int(xs.min()) - pad),
                            max(0, int(ys.min()) - pad),
                            min(w - 1, int(xs.max()) + pad),
                            min(h - 1, int(ys.max()) + pad),
                        ]
                    ],
                    dtype=np.float32,
                )

            np.savez_compressed(out_path, imgs=arr, boxes=box)
            count += 1

    return count


def run_inference(repo_dir: Path, input_root: Path, output_dir: Path, model: Path, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_dir / "infer_scripts" / "infer_torch.py"),
        "-i",
        str(input_root),
        "-o",
        str(output_dir),
        "--model",
        str(model),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=repo_dir, check=True)


def compute_metrics(pred_dir: Path, mask_dir: Path, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for pred_path in sorted(pred_dir.glob("2DBox_HAM10K_*.npz")):
        match = re.search(r"(ISIC_\d+)", pred_path.stem)
        if match is None:
            continue
        image_id = match.group(1)
        gt_path = mask_dir / f"{image_id}_segmentation.png"
        if not gt_path.exists():
            continue

        pred = np.load(pred_path)["segs"] > 0
        gt_img = Image.open(gt_path).convert("L")
        if gt_img.size != (pred.shape[1], pred.shape[0]):
            gt_img = gt_img.resize((pred.shape[1], pred.shape[0]), Image.Resampling.NEAREST)
        gt = np.asarray(gt_img) > 0

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        rows.append(
            {
                "case": pred_path.name,
                "image_id": image_id,
                "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
                "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "pixel_accuracy": (tp + tn) / pred.size,
                "pred_area": int(pred.sum()),
                "gt_area": int(gt.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    fields = [
        "case",
        "image_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "pixel_accuracy",
        "pred_area",
        "gt_area",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with (metrics_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (metrics_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "min", "max"])
        for metric in ["dice", "iou", "precision", "recall", "pixel_accuracy"]:
            vals = np.array([row[metric] for row in rows], dtype=float)
            writer.writerow([metric, vals.mean(), vals.min(), vals.max()])

    print(f"Computed metrics for {len(rows)} predictions")
    print((metrics_dir / "summary.csv").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ham10k-zip", type=Path, required=True)
    parser.add_argument("--mask-zip", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/ham10k_eval_work"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/ham10k_eval_outputs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = extract_masks(args.mask_zip, args.work_dir)
    image_zip = find_image_zip(args.ham10k_zip, args.work_dir)
    input_root = args.work_dir / "inputs"
    input_img_dir = input_root / "imgs"

    count = make_inputs(image_zip, mask_dir, input_img_dir, args.limit)
    print(f"Prepared {count} model inputs in {input_img_dir}")

    pred_dir = args.output_dir / "predictions"
    run_inference(repo_dir, input_root, pred_dir, args.model, args.device)

    metrics_dir = args.output_dir / "metrics"
    compute_metrics(pred_dir, mask_dir, metrics_dir)


if __name__ == "__main__":
    main()
