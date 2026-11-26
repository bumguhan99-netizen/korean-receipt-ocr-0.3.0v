import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare import ROOT, pixel_sha, sha_file

MAX_ENCODER_LENGTH = 23
EXPECTED_MODEL_SHA = "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b"
ALT_WIDTHS = (48, 80, 128, 192, 256, 320, 480, 640)
ROTATE = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
          270: Image.Transpose.ROTATE_90}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text):
    return unicodedata.normalize("NFC", " ".join(text.split()))


def json_write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class Recognizer:
    def __init__(self, path, dictionary, threads):
        import onnxruntime as ort

        if sha_file(path) != EXPECTED_MODEL_SHA:
            raise ValueError("Unexpected recognition model SHA-256")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.characters = [""] + dictionary + [" "]
        if self.session.get_outputs()[0].shape[-1] != len(self.characters):
            raise ValueError("Model output does not match character dictionary")

    def recognize(self, entries):
        buckets = defaultdict(list)
        result = {}
        for index, image in entries:
            width = min(640, max(1, math.ceil(48 * image.width / image.height)))
            bucket = next(size for size in ALT_WIDTHS if size >= width)
            resized = image.resize((width, 48), Image.Resampling.BILINEAR)
            # PP-OCR recognition consumes normalized BGR, with zero padding on the right.
            pixels = np.asarray(resized, dtype=np.float32)[:, :, ::-1]
            buckets[bucket].append((index, (pixels / 127.5 - 1).transpose(2, 0, 1)))

        for width, items in buckets.items():
            for start in range(0, len(items), 16):
                batch = items[start:start + 16]
                tensor = np.zeros((len(batch), 3, 48, width), dtype=np.float32)
                for offset, (_, pixels) in enumerate(batch):
                    tensor[offset, :, :, :pixels.shape[2]] = pixels
                predictions, = self.session.run(None, {self.input_name: tensor})
                indices = predictions.argmax(axis=2)
                maxima = predictions.max(axis=2)
                for (index, _), tokens, probs in zip(batch, indices, maxima):
                    selected = tokens != 0
                    selected[1:] &= tokens[1:] != tokens[:-1]
                    chars = [self.characters[i] for i in tokens[selected]]
                    confidence = probs[selected]
                    text = normalize("".join(chars))
                    mean = float(confidence.mean()) if chars else 0.0
                    minimum = float(confidence.min()) if chars else 0.0
                    result[index] = (text, mean, minimum)
        return result


def inverse_point(x, y, width, height, rotation):
    if rotation == 90:
        return y, height - x
    if rotation == 180:
        return width - x, height - y
    if rotation == 270:
        return width - y, x
    return x, y


def bbox_corners(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def read_groups(line, digit_ratio, global_slope):
    words = [w for w in line.get("words") or [] if w["text"].strip()]
    words.sort(key=lambda w: w["box"][0])
    if not words:
        return []
    sizes = [w["box"][3] - w["box"][1] for w in words]
    char_width = max(3, statistics.median(sizes) * digit_ratio)
    groups = [[]]
    for word in words:
        if groups[-1] and word["box"][0] - groups[-1][-1]["box"][2] >= 2 * char_width:
            groups.append([])
        groups[-1].append(word)
    return [(g, char_width, global_slope) for g in groups]


def crop_windows_group(frame, group, char_width, slope):
    boxes = [w["box"] for w in group]
    heights = [y2 - y1 for _, y1, _, y2 in boxes]
    median_height = statistics.median(heights)
    centers = np.array([((a + c) / 2, (b + d) / 2) for a, b, c, d in boxes])
    if len(boxes) > 1:
        center_span = centers[-1, 0] - centers[0, 0]
        if center_span >= 2.5 * median_height:
            fitted = float(np.polyfit(centers[:, 0], centers[:, 1], deg=1)[0])
            if abs(fitted - slope) <= 0.15:
                slope = fitted
    angle = math.atan(slope)
    c, s = math.cos(angle), math.sin(angle)
    xs, ys = zip(*[(x * c + y * s, -x * s + y * c)
                   for box in boxes for x, y in bbox_corners(box)])
    pad = max(1.0, median_height * 0.12)
    vbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    points = [(max(0.0, min(frame.width, x * c - y * s)),
               max(0.0, min(frame.height, x * s + y * c))) for x, y in bbox_corners(vbox)]
    width = int(round(vbox[2] - vbox[0]))
    height = int(round(vbox[3] - vbox[1]))
    if min(width, height) < 3:
        return None, points, "", None
    pil_quad = [part for i in (0, 3, 2, 1) for part in points[i]]
    crop = frame.transform((width, height), Image.Transform.QUAD, pil_quad,
                           resample=Image.Resampling.BICUBIC)
    text = group[0]["text"]
    for previous, word in zip(group, group[1:]):
        if word["box"][0] - previous["box"][2] >= char_width * 0.5:
            text += " "
        text += word["text"]
    deviations = -centers[:, 0] * s + centers[:, 1] * c
    return crop, points, normalize(text), float(np.ptp(deviations)) / max(1, median_height)


def start_auto(frame, draft, allowed):
    rotated = int(draft["rotation"])
    rotation = frame.transpose(ROTATE[rotated]) if rotated else frame
    scaled_w, scaled_h = draft["width"], draft["height"]
    scale_x = scaled_w / frame.width
    scale_y = scaled_h / frame.height
    scale_in_x, scale_in_y = (scale_y, scale_x) if rotated in (90, 270) else (scale_x, scale_y)
    lines = draft["lines"]
    for line in lines:
        for word in line.get("words") or []:
            x1, y1, x2, y2 = word["box"]
            word["box"] = [x1 / scale_in_x, y1 / scale_in_y, x2 / scale_in_x, y2 / scale_in_y]
    digit_samples = [(x2 - x1) / ((y2 - y1) * len(w["text"]))
                     for line in lines for w in (line.get("words") or [])
                     if re.fullmatch(r"[0-9]{2,}", w["text"])
                     for x1, y1, x2, y2 in [w["box"]] if y2 > y1 and x2 > x1]
    digit_ratio = min(0.82, max(0.4, statistics.median(digit_samples))) if digit_samples else 0.6
    slope = math.tan(math.radians(-max(-18, min(18, draft["angle"]))))
    records = []
    for line in lines:
        for group, char_width, row_slope in read_groups(line, digit_ratio, slope):
            crop, box, win_text, spread = crop_windows_group(rotation, group, char_width, row_slope)
            mapped = [inverse_point(x, y, frame.width, frame.height, rotated) for x, y in box]
            first_corner = min(range(4), key=lambda i: mapped[i][0] + mapped[i][1])
            mapped = mapped[first_corner:] + mapped[:first_corner]
            points = [[round(x, 2), round(y, 2)] for x, y in mapped]
            record = dict(source="windows_ocr+ppocrv5", status="needs_review", box=points,
                          windows_text=win_text, text=win_text,
                          image_rotation_cw=rotated, flags=[], hold_reasons=[])
            if crop is None or not win_text or not any(ch.isalnum() for ch in win_text):
                record.update(status="excluded", exclude_code="E07")
            else:
                if crop.width < 8 or crop.height < 9 or (len(win_text) >= 3 and crop.width < 1.15 * crop.height):
                    record["hold_reasons"].append("small_or_vertical_crop")
                if spread > 0.60:
                    record["hold_reasons"].append("irregular_baseline")
                record["hold_reasons"].extend(encoder_flags(win_text, allowed))
            records.append((record, crop))
    return records


def finish_auto(items, recognizer):
    queued = [(idx, crop) for idx, (row, crop) in enumerate(items)
              if crop is not None and row["status"] != "excluded"]
    results = recognizer.recognize(queued)
    for idx, (row, _) in enumerate(items):
        if idx not in results:
            continue
        text, score, minimum = results[idx]
        row.update(paddle_text=text, paddle_confidence=round(score, 5),
                   paddle_min_character_confidence=round(minimum, 5))
        if text != row["windows_text"]:
            row["hold_reasons"].append("ocr_disagreement")
        if score < 0.93 or minimum < 0.75:
            row["hold_reasons"].append("low_recognition_confidence")
        if not row["hold_reasons"]:
            row["status"] = "pseudo"


def encoder_flags(text, allowed):
    reasons = []
    if len(text) > MAX_ENCODER_LENGTH:
        reasons.append("encoder_length")
    if set(text) - allowed:
        reasons.append("out_of_vocabulary")
    return reasons


def prior_records(project):
    output = defaultdict(list)
    labels = project / "ocr_v3/data/labels_v3"
    sources = (labels / "10_lines/phone_humanbox.jsonl", labels / "10_lines/holdout52_L1.jsonl")
    for path in sources:
        for record in read_jsonl(path):
            if record["status"] in {"verified", "excluded"}:
                output[record["receipt_id"]].append(record)
    return output


def label_receipt(receipt, old, project, allowed, recognizer):
    with Image.open(ROOT / receipt["image"]) as frame:
        frame.load()
    width, height = frame.size
    previous = receipt["previous_receipt_id"]
    receipt_id = receipt["receipt_id"]
    entries = []
    if previous:
        for item in old[previous]:
            raw_text = item.get("text") or ""
            text = normalize(raw_text)
            flags = list(item.get("flags") or [])
            points = item["box"]["points"]
            clipped = [[round(max(0, min(width, x)), 2), round(max(0, min(height, y)), 2)]
                       for x, y in points]
            record = dict(source="prior_visual_review", previous_line_id=item["line_id"],
                          previous_review_date=item.get("reviewed_at"), box=clipped,
                          status=item["status"], text=text, flags=flags,
                          exclude_code=item.get("exclude_code"),
                          crop_rotation_degrees=item.get("crop_rotation_degrees") or 0,
                          hold_reasons=encoder_flags(text, allowed) if text else [])
            if text != raw_text:
                record["previous_text"] = raw_text
            if clipped != points:
                record["flags"].append("legacy_box_clipped_to_frame")
            crop = None
            if item["status"] == "verified":
                source_crop = project / "ocr_v3" / item["crop"]
                if item.get("crop_sha256") and sha_file(source_crop) != item["crop_sha256"]:
                    raise ValueError(f"Reference crop hash mismatch: {item['line_id']}")
                with Image.open(source_crop) as image:
                    crop = image.convert("RGB")
            entries.append((record, crop))
    elif receipt["split"] == "train":
        path = ROOT / "audit/windows_ocr" / f"{receipt_id}.json"
        draft = json.loads(path.read_text(encoding="utf-8"))
        entries = start_auto(frame, draft, allowed)
        finish_auto(entries, recognizer)

    for index, (line, crop) in enumerate(entries, 1):
        line_id = f"{receipt_id}_line_{index:03d}"
        line["line_id"] = line_id
        if len(line["text"]) >= 24 and "long" not in line["flags"]:
            line["flags"].append("long")
        if "out_of_vocabulary" in line["hold_reasons"] and "oov" not in line["flags"]:
            line["flags"].append("oov")
        line["crop"] = None
        if crop is not None and line["status"] != "excluded":
            # Image coordinates are local to the receipt; file names carry no original IDs.
            path = f"crops/{receipt['split']}/{line_id}.png"
            line["crop"] = path
            line["crop_size"] = list(crop.size)
            line["crop_pixel_sha256"] = pixel_sha(crop)
            crop.save(ROOT / path, "PNG", compress_level=3)
    return {key: receipt[key] for key in
            ("receipt_id", "group_id", "split", "image", "output_size", "previous_receipt_id")} | {
                "lines": [line for line, _ in entries]}


def main():
    parser = argparse.ArgumentParser(description="Prepare line crops and local two-engine OCR labels.")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot", type=int, default=0)
    args = parser.parse_args()
    project = args.project.resolve()
    source_dictionary = project / "ocr_v3/models/korean_characters.txt"
    dictionary = source_dictionary.read_text(encoding="utf-8").splitlines()
    allowed = set(dictionary) | {" "}
    recognizer = Recognizer(args.recognizer, dictionary, args.threads)
    receipts = read_jsonl(ROOT / "audit/preprocessing.jsonl")
    if args.pilot:
        raws = [r for r in receipts if not r["previous_receipt_id"] and r["split"] == "train"]
        for receipt in raws[:args.pilot]:
            with Image.open(ROOT / receipt["image"]) as image:
                draft = ROOT / "audit/windows_ocr" / f"{receipt['receipt_id']}.json"
                items = start_auto(image, json.loads(draft.read_text(encoding="utf-8")), allowed)
            finish_auto(items, recognizer)
            print(receipt["receipt_id"], Counter(v["status"] for v, _ in items),
                  Counter(code for v, _ in items for code in v["hold_reasons"]), flush=True)
        return

    prior = prior_records(project)
    for split in sorted({r["split"] for r in receipts}):
        (ROOT / "crops" / split).mkdir(parents=True, exist_ok=True)
    (ROOT / "labels").mkdir(exist_ok=True)
    model = ROOT / "characters.txt"
    if not model.exists():
        with model.open("xb") as handle:
            handle.write(source_dictionary.read_bytes())
    for index, receipt in enumerate(receipts, 1):
        out = ROOT / "labels" / f"{receipt['receipt_id']}.json"
        if out.exists():
            continue
        json_write(out, label_receipt(receipt, prior, project, allowed, recognizer))
        if index % 25 == 0 or index == len(receipts):
            print(f"Labeled {index}/{len(receipts)}", flush=True)


if __name__ == "__main__":
    main()
