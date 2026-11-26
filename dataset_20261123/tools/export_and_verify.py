import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from build_labels import MAX_ENCODER_LENGTH, encoder_flags, json_write, normalize, read_jsonl
from prepare import IMAGE_TYPES, ROOT, pixel_sha, sha_file

SPLITS = ("train", "val", "test", "holdout", "quarantine")
MANIFEST_NAMES = {"train": "train.txt", "val": "validation.txt", "test": "test.txt",
                  "holdout": "holdout.txt"}
PRIORITY = {"train": 0, "val": 1, "holdout": 2, "test": 3}


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def validate_receipts(source=None):
    metadata = read_jsonl(ROOT / "audit/preprocessing.jsonl")
    character_set = set((ROOT / "characters.txt").read_text(encoding="utf-8").splitlines()) | {" "}
    group_splits = {}
    source_pixels = {}
    receipts = []
    lines_by_split_status = Counter()
    reason_counts = Counter()
    crop_paths = set()
    nonempty_receipts = Counter()
    known_count = Counter()
    by_receipt = {}
    for item in metadata:
        rid, group, split = (item[key] for key in ("receipt_id", "group_id", "split"))
        verify(rid not in by_receipt, f"Repeated receipt: {rid}")
        verify(split in SPLITS, f"Unknown split: {split}")
        verify(group_splits.setdefault(group, split) == split, f"Cross-split group: {group}")
        frame = ROOT / f"images/{rid}.png"
        label = ROOT / f"labels/{rid}.json"
        verify(sha_file(frame) == item["image_sha256"], f"Receipt frame checksum mismatch: {rid}")
        with Image.open(frame) as image:
            image.load()
            verify(image.mode == "RGB" and list(image.size) == item["output_size"],
                   f"Receipt frame mode/size mismatch: {rid}")
        if source is not None:
            verify(sha_file(source / item["original_name"]) == item["source_sha256"],
                   f"Original file checksum mismatch: {rid}")

        verify(source_pixels.setdefault(item["pixel_sha256"], split) == split,
               f"Cross-split original pixels: {rid}")
        record = json.loads(label.read_text(encoding="utf-8"))
        width, height = item["output_size"]
        verify(record["receipt_id"] == rid and record["group_id"] == group and
               record["split"] == split and record["output_size"] == [width, height] and
               record["image"] == f"images/{rid}.png", f"Label/frame metadata mismatch: {rid}")
        verify(record["lines"], f"Receipt has zero label records: {rid}")
        known_count.update(["prior_visual_review" if item["previous_receipt_id"] else "new_ocr"])
        for index, line in enumerate(record["lines"], 1):
            line_id = f"{rid}_line_{index:03d}"
            status = line["status"]
            text = line["text"]
            verify(line["line_id"] == line_id, f"Line identifier mismatch: {line_id}")
            verify(status in {"verified", "pseudo", "needs_review", "excluded"}, f"Status: {line_id}")
            verify(normalize(text) == text, f"Label text is not normalized: {line_id}")
            verify(status != "pseudo" or split == "train", f"Pseudo-label in evaluation: {line_id}")
            verify(line["source"] == "prior_visual_review" or split == "train",
                   f"New OCR in evaluation: {line_id}")
            verify(status not in {"verified", "pseudo"} or bool(text), f"Empty label: {line_id}")
            verify(status != "excluded" or line.get("exclude_code"), f"Missing exclude code: {line_id}")
            if status != "excluded":
                verify(all(code in line["hold_reasons"] for code in encoder_flags(text, character_set)),
                       f"Missing encoder flag: {line_id}")

            points = line["box"]
            in_bounds = len(points) == 4 and all(0 <= x <= width and 0 <= y <= height for x, y in points)
            verify(in_bounds, f"Line box outside frame: {line_id}")
            lines_by_split_status[(split, status)] += 1
            reason_counts.update(line["hold_reasons"])
            crop_path = line["crop"]
            if crop_path:
                verify(crop_path == f"crops/{split}/{line_id}.png" and crop_path not in crop_paths,
                       f"Unexpected or reused crop path: {line_id}")
                with Image.open(ROOT / crop_path) as crop:
                    crop.load()
                    verify(crop.mode == "RGB" and list(crop.size) == line["crop_size"],
                           f"Crop size/mode mismatch: {line_id}")
                    verify(pixel_sha(crop) == line["crop_pixel_sha256"], f"Crop pixels changed: {line_id}")
                crop_paths.add(crop_path)
            if status in {"verified", "pseudo"}:
                verify(bool(crop_path), f"Missing crop: {line_id}")
                doubled_area = abs(sum(a[0] * b[1] - b[0] * a[1]
                                       for a, b in zip(points, points[1:] + points[:1])))
                verify(doubled_area > 1, f"Degenerate four-point box: {line_id}")
        by_receipt[rid] = record
        nonempty_receipts.update([split])
        receipts.append(record)

    expected_frames = {ROOT / r["image"] for r in receipts}
    expected_labels = {ROOT / f"labels/{rid}.json" for rid in by_receipt}
    verify(set((ROOT / "images").glob("*.png")) == expected_frames, "Untracked receipt PNGs")
    verify(set((ROOT / "labels").glob("*.json")) == expected_labels, "Untracked receipt JSON")
    verify({p.relative_to(ROOT).as_posix() for p in (ROOT / "crops").rglob("*.png")} == crop_paths,
           "Untracked line PNGs")
    if source is not None:
        actual = {p.name for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES}
        verify(actual == {r["original_name"] for r in metadata}, "Source image coverage mismatch")

    summary = dict(images=len(receipts), labels=len(expected_labels), crops=len(crop_paths),
                   groups=len(group_splits), receipt_provenance=dict(known_count),
                   receipts_by_split=dict(nonempty_receipts),
                   labels_by_split_status={f"{split}/{status}": n
                                           for (split, status), n in sorted(lines_by_split_status.items())},
                   hold_reason_counts=dict(reason_counts), encoder_max_label_length=MAX_ENCODER_LENGTH,
                   source_hashes_verified=source is not None)
    return receipts, summary


def build_manifests(receipts):
    candidates = defaultdict(list)
    issues = []
    for receipt in receipts:
        split = receipt["split"]
        if split == "quarantine":
            continue
        for line in receipt["lines"]:
            if line["status"] in {"verified", "pseudo"} and not line["hold_reasons"]:
                candidates[line["crop_pixel_sha256"]].append((receipt, line))

    chosen = defaultdict(list)
    for matches in candidates.values():
        no_conflict = len({r["text"] for _, r in matches}) == 1
        keep = max((r["split"] for r, _ in matches), key=PRIORITY.__getitem__)
        for receipt, line in matches:
            split = receipt["split"]
            if no_conflict and split == keep:
                chosen[split].append((receipt["receipt_id"], line))
            else:
                reason = "same_crop_conflicting_transcriptions" if not no_conflict else "cross_split_crop_pixels"
                issues.append((line["line_id"], split, reason))
    for split in chosen:
        chosen[split].sort(key=lambda row: row[1]["line_id"])

    exports = {MANIFEST_NAMES[split]: chosen[split] for split in MANIFEST_NAMES if chosen[split]}
    exports["train_verified.txt"] = [(r, v) for r, v in chosen["train"] if v["status"] == "verified"]
    exports["train_pseudo.txt"] = [(r, v) for r, v in chosen["train"] if v["status"] == "pseudo"]

    hashes = {}
    groups = {}
    by_id = {r["receipt_id"]: r for r in receipts}
    for split in MANIFEST_NAMES:
        for rid, line in chosen[split]:
            digest = line["crop_pixel_sha256"]
            group = by_id[rid]["group_id"]
            verify(hashes.setdefault(digest, split) == split, "Cross-split crop pixels after export")
            verify(groups.setdefault(group, split) == split, "Cross-split receipt at export")

    active = {r for split in MANIFEST_NAMES for r, _ in chosen[split]}
    report = {
        "manifest_lines": {k: len(v) for k, v in exports.items()},
        "manifest_receipts": {split: len({rid for rid, _ in chosen[split]}) for split in MANIFEST_NAMES},
        "unexported_receipts": [
            {"receipt_id": r["receipt_id"], "split": r["split"],
             "reason": "split_conflict" if r["split"] == "quarantine" else "no_eligible_lines"}
            for r in receipts if r["receipt_id"] not in active],
        "removed_manifest_lines": [{"line_id": i, "split": s, "reason": c}
                                   for i, s, c in sorted(issues)],
        "evaluation_contains_pseudo_labels": False,
        "cross_split_identical_crop_pixels": 0,
        "cross_split_receipt_groups": 0,
    }
    return exports, report


def manifest_text(rows):
    return "".join(f"{v['crop']}\t{v['text']}\n" for _, v in rows)


def main():
    parser = argparse.ArgumentParser(description="Build and validate PaddleOCR tab-separated manifests.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    receipts, summary = validate_receipts(args.source)
    exports, report = build_manifests(receipts)
    for name, rows in exports.items():
        content = manifest_text(rows)
        path = ROOT / name
        if args.check_only:
            verify(path.read_text(encoding="utf-8") == content, f"Manifest content mismatch: {name}")
        else:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
    if not args.check_only:
        json_write(ROOT / "audit/validation.json", summary | report)
    print(json.dumps(summary | {k: report[k] for k in ("manifest_lines", "manifest_receipts")}, indent=2))
    print("Unexported receipts:", len(report["unexported_receipts"]),
          "cross-split or ambiguous crops:", len(report["removed_manifest_lines"]))


if __name__ == "__main__":
    main()
