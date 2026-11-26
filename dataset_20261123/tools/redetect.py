import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from build_labels import ROOT, ROTATE, Recognizer, inverse_point, json_write
from prepare import sha_file

DETECTOR_SHA = "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"


def detector(path, threads):
    if sha_file(path) != DETECTOR_SHA:
        raise ValueError("Unexpected detection model SHA-256")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def detect(session, image):
    factor = min(1.0, 1536 / max(image.size))
    width, height = [max(32, int(round(v * factor / 32) * 32)) for v in image.size]
    rgb = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    normalized = (rgb[:, :, ::-1] / 255 - np.array([0.485, 0.456, 0.406], np.float32))
    normalized /= np.array([0.229, 0.224, 0.225], np.float32)
    tensor = normalized.transpose(2, 0, 1)[None].copy()
    pred = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0]
    mask = (pred > 0.3).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y), (w, h), theta = cv2.minAreaRect(contour)
        if min(w, h) < 4:
            continue
        if h > w:
            w, h = h, w
            theta += 90
        if theta > 90:
            theta -= 180
        if w < 1.12 * h or abs(theta) > 35:
            continue
        center_x, center_y = int(round(x)), int(round(y))
        if pred[min(pred.shape[0] - 1, center_y), min(pred.shape[1] - 1, center_x)] < 0.35:
            continue
        box = cv2.boxPoints(((x, y), (w + h * 0.28, h * 1.34), theta))
        xs = sorted(box, key=lambda point: point[0])
        left = sorted(xs[:2], key=lambda point: point[1])
        right = sorted(xs[2:], key=lambda point: point[1])
        quad = [left[0], right[0], right[1], left[1]]
        points = [[float(np.clip(px * image.width / pred.shape[1], 0, image.width)),
                   float(np.clip(py * image.height / pred.shape[0], 0, image.height))]
                  for px, py in quad]
        out_w, out_h = (round(math.dist(points[0], points[i])) for i in (1, 3))
        if out_w < 8 or out_h < 9:
            continue
        crop = image.transform((out_w, out_h), Image.Transform.QUAD,
                               [p for i in (0, 3, 2, 1) for p in points[i]],
                               resample=Image.Resampling.BICUBIC)
        yield points, crop


def orientation_score(predictions):
    values = sorted((mean for text, mean, _ in predictions.values()
                     if len(text) > 1 and any(ch.isalnum() for ch in text)), reverse=True)[:14]
    return sum(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    args = parser.parse_args()
    vocabulary = (ROOT / "characters.txt").read_text(encoding="utf-8").splitlines()
    rec = Recognizer(args.recognizer, vocabulary, 4)
    det = detector(args.detector, 4)
    labels = sorted((ROOT / "labels").glob("*.json"))
    out = ROOT / "audit/visual"
    output_crops = out / "det_crops"
    output_crops.mkdir(exist_ok=False)
    pending = []
    for path in labels:
        label = json.loads(path.read_text(encoding="utf-8"))
        if not label["previous_receipt_id"] and not any(x["status"] == "pseudo" for x in label["lines"]):
            pending.append(label)
    font = ImageFont.truetype("arial.ttf", 16)
    records = []
    columns, batch, width, cell_height = 2, 8, 640, 312
    for offset in range(0, len(pending), batch):
        frame = Image.new("RGB", (columns * width, (batch // columns) * cell_height), "white")
        draw = ImageDraw.Draw(frame)
        for i, label in enumerate(pending[offset:offset + batch]):
            rid = label["receipt_id"]
            raw = Image.open(ROOT / label["image"])
            turns = (label["lines"][0].get("image_rotation_cw", 0) % 180)
            img = raw.transpose(ROTATE[turns]) if turns else raw
            proposals = list(detect(det, img))
            forward = rec.recognize((j, crop) for j, (_, crop) in enumerate(proposals))
            reverse = rec.recognize((j, crop.transpose(ROTATE[180])) for j, (_, crop) in enumerate(proposals))
            flip = orientation_score(reverse) > orientation_score(forward)
            predictions = reverse if flip else forward
            preferred = []
            for j, (pts, crop) in enumerate(proposals):
                text, confidence, _ = predictions[j]
                if not text or not any(ch.isalnum() for ch in text):
                    continue
                rank = confidence + (0.15 if 2 <= len(text) <= 19 else 0) \
                       + (0.10 if any("가" <= c <= "힣" for c in text) else 0)
                preferred.append((rank, j, pts, crop, text))
            preferred.sort(reverse=True)
            x, y = i % columns * width, i // columns * cell_height
            draw.text((x + 6, y), rid, fill="black", font=font)
            seen = set()
            for index, (_, _, pts, crop, guess) in enumerate(preferred):
                if len(seen) >= 4:
                    break
                if guess in seen:
                    continue
                seen.add(guess)
                key = f"{rid}_det_{len(seen):02d}"
                if flip:
                    crop = crop.transpose(ROTATE[180])
                rel = output_crops.relative_to(ROOT).as_posix() + f"/{key}.png"
                crop.save(ROOT / rel, "PNG")
                pts = [inverse_point(px, py, raw.width, raw.height, turns) for px, py in pts]
                corner = min(range(4), key=lambda k: pts[k][0] + pts[k][1])
                pts = pts[corner:] + pts[:corner]
                coords = [[round(px, 2), round(py, 2)] for px, py in pts]
                records.append(dict(receipt_id=rid, key=key, box=coords, audit_crop=rel,
                                    rotation_cw=turns + 180 * int(flip)))
                row = y + 17 + (len(seen) - 1) * 71
                draw.text((x + 6, row), key, fill="black", font=font)
                scale = min((width - 12) / crop.width, 48 / crop.height, 4)
                zoom = crop.resize((max(1, round(crop.width * scale)),
                                    max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
                frame.paste(zoom, (x + 6, row + 18))
            raw.close()
            draw.line((x, y + 307, x + width - 5, y + 307), fill="#111111", width=2)
        frame.save(out / f"redetected_{offset // batch + 1:02d}.png")
        print(f"Re-detected {min(offset + batch, len(pending))}/{len(pending)}", flush=True)
    json_write(out / "redetected.json", records)


if __name__ == "__main__":
    main()
