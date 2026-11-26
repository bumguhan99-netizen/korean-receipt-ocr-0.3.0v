import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MAX_SIDE = 2400
NEAR_HASH_DISTANCE = 10
NEAR_THUMB_ERROR = 4.0


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha_file(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load_rgb(path):
    with Image.open(path) as raw:
        orientation = raw.getexif().get(274, 1)
        size = raw.size
        oriented = ImageOps.exif_transpose(raw)
        if "A" in oriented.getbands() or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = oriented.convert("RGB")
        return rgb, orientation, size


def signature(image):
    small = np.asarray(image.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
    bits = np.packbits((small[:, 1:] > small[:, :-1]).reshape(-1))
    return int.from_bytes(bits, "big"), small.astype(np.int16)


def pixel_sha(image):
    digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def groups_for(rows, signatures):
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    links = []
    for right, row in enumerate(rows):
        right_hash, right_thumb = signatures[right]
        for left in range(right):
            other = rows[left]
            left_hash, left_thumb = signatures[left]
            if row["pixel_sha256"] == other["pixel_sha256"]:
                reason = "identical_pixels"
                distance, error = 0, 0.0
            elif row["source_sequence"] and row["source_sequence"] == other["source_sequence"]:
                reason = "filename_sequence_pair"
                distance, error = None, None
            else:
                distance = (right_hash ^ left_hash).bit_count()
                if distance > NEAR_HASH_DISTANCE:
                    continue
                error = float(np.abs(right_thumb - left_thumb).mean())
                if error > NEAR_THUMB_ERROR:
                    continue
                reason = "near_identical_frame"
            parents[find(right)] = find(left)
            links.append(dict(left=other["receipt_id"], right=row["receipt_id"],
                              reason=reason, distance=distance, thumbnail_error=error))

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    for group in groups.values():
        group_id = group[0]["receipt_id"]
        known = {r["prior_split"] for r in group if r["prior_split"]}
        split = next(iter(known)) if len(known) == 1 else ("quarantine" if known else "train")
        for row in group:
            row["group_id"] = group_id
            row["split"] = split
    return links


def main():
    parser = argparse.ArgumentParser(description="Normalize receipt images and freeze receipt-level splits.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    project = args.project.resolve()
    prior_root = project / "ocr_v2_retrain/datasets/raw/receipt_labeled"
    v3_root = project / "ocr_v3/data/labels_v3"
    images = ROOT / "images"
    images.mkdir(exist_ok=False)
    output = ROOT / "audit"
    prior = {}
    for row in read_csv(v3_root / "00_splits/phone_map.csv"):
        name = row["receipt_id"]
        prior[name] = {"kind": "phone", "split": row["split"],
                       "raw": prior_root / "images" / f"{name}.png",
                       "frame": prior_root / "aligned_images" / f"{name}.png"}
    for row in read_csv(v3_root / "00_splits/holdout52_map.csv"):
        if row["split"] == "holdout":
            frame = project / "ocr_v3/data" / row["original"]
            prior[row["receipt_id"]] = {"kind": "holdout52", "split": "holdout",
                                        "raw": frame, "frame": frame}

    prior_by_hash = defaultdict(list)
    for receipt_id, item in prior.items():
        im, _, _ = load_rgb(item["raw"])
        prior_by_hash[pixel_sha(im)].append(receipt_id)

    rows = []
    signatures = []
    for index, path in enumerate(sorted(p for p in source.iterdir()
                                        if p.is_file() and p.suffix.lower() in IMAGE_TYPES), 1):
        receipt_id = f"receipt_{index:04d}"
        raw, orientation, source_size = load_rgb(path)
        raw_hash = pixel_sha(raw)
        signatures.append(signature(raw))
        matches = prior_by_hash[raw_hash]
        preferred = next((key for key in matches if prior[key]["kind"] == "phone"), None)
        chosen = preferred or (matches[0] if matches else None)
        splits = {prior[key]["split"] for key in matches}
        if len(splits) > 1:
            prior_split = "quarantine"
            chosen = None
        else:
            prior_split = prior[chosen]["split"] if chosen else None
        if chosen:
            frame, _, _ = load_rgb(prior[chosen]["frame"])
            transform = "verified_reference_aligned_frame" if preferred else "verified_reference_frame"
        else:
            frame = raw.copy()
            frame.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            transform = "exif_rgb_lanczos" if frame.size != raw.size else "exif_rgb"
        rel = f"images/{receipt_id}.png"
        dest = ROOT / rel
        frame.save(dest, "PNG", compress_level=3)
        sequence = re.fullmatch(r"(\d{6}[-_]\d{4})-\d+", path.stem)
        rows.append(dict(receipt_id=receipt_id, original_name=path.name, image=rel,
                         source_sha256=sha_file(path), pixel_sha256=raw_hash, image_sha256=sha_file(dest),
                         source_size=list(source_size), exif_orientation=orientation,
                         oriented_size=list(raw.size), output_size=list(frame.size), transform=transform,
                         previous_receipt_id=chosen, prior_split=prior_split,
                         pixel_matched_ids=matches, source_sequence=sequence[1] if sequence else None))
        if index % 100 == 0:
            print(f"Prepared {index}", flush=True)

    links = groups_for(rows, signatures)
    (output / "preprocessing.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "receipt_links.json").write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")
    with (output / "ocr_queue.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["receipt_id", "image", "width", "height"])
        for row in rows:
            if row["split"] == "train" and not row["previous_receipt_id"]:
                writer.writerow([row["receipt_id"], row["image"], *row["output_size"]])
    print("Prior:", Counter((r["prior_split"] or "new") for r in rows))
    print("Splits:", Counter(r["split"] for r in rows), "links:", len(links))


if __name__ == "__main__":
    main()
