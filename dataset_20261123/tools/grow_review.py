import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from build_labels import ROOT, ROTATE, inverse_point, json_write


def forward_point(x, y, width, height, rotation):
    if rotation == 90:
        return height - y, x
    if rotation == 180:
        return width - x, height - y
    if rotation == 270:
        return y, width - x
    return x, y


def main():
    audit = ROOT / "audit/visual"
    candidates = json.loads((audit / "redetected.json").read_text(encoding="utf-8"))
    dest = audit / "expanded_crops"
    dest.mkdir(exist_ok=False)
    font = ImageFont.truetype("arial.ttf", 16)
    by_receipt = {}
    for candidate in candidates:
        by_receipt.setdefault(candidate["receipt_id"], []).append(candidate)
    ids = list(by_receipt)
    output = []
    for offset in range(0, len(ids), 8):
        frame = Image.new("RGB", (1280, 1248), "white")
        draw = ImageDraw.Draw(frame)
        for i, rid in enumerate(ids[offset:offset + 8]):
            with Image.open(ROOT / "images" / f"{rid}.png") as raw:
                for j, item in enumerate(by_receipt[rid]):
                    degrees = item["rotation_cw"]
                    image = raw.transpose(ROTATE[degrees]) if degrees else raw
                    pts = [forward_point(x, y, *raw.size, degrees) for x, y in item["box"]]
                    first = min(range(4), key=lambda k: pts[k][0] + pts[k][1])
                    a, b, c, d = pts[first:] + pts[:first]
                    left = tuple((x + y) / 2 for x, y in zip(a, d))
                    right = tuple((x + y) / 2 for x, y in zip(b, c))
                    width = max(1, math.dist(left, right))
                    height = max(1, (math.dist(a, d) + math.dist(b, c)) / 2)
                    u = tuple((q - p) / width for p, q in zip(left, right))
                    v = (-u[1], u[0])
                    center = tuple((x + y) / 2 for x, y in zip(left, right))
                    grown_w, grown_h = width + 0.28 * height, height * 1.48
                    corners = [(min(image.width, max(0, center[0] + x * u[0] + y * v[0])),
                                min(image.height, max(0, center[1] + x * u[1] + y * v[1])))
                               for x, y in [(-grown_w / 2, -grown_h / 2), (grown_w / 2, -grown_h / 2),
                                            (grown_w / 2, grown_h / 2), (-grown_w / 2, grown_h / 2)]]
                    crop = image.transform((round(grown_w), round(grown_h)), Image.Transform.QUAD,
                                           [p for k in (0, 3, 2, 1) for p in corners[k]],
                                           resample=Image.Resampling.BICUBIC)
                    key = item["key"]
                    rel = f"audit/visual/expanded_crops/{key}.png"
                    crop.save(ROOT / rel)
                    mapped = [inverse_point(x, y, *raw.size, degrees) for x, y in corners]
                    first = min(range(4), key=lambda k: mapped[k][0] + mapped[k][1])
                    mapped = mapped[first:] + mapped[:first]
                    output.append(item | {"audit_crop": rel, "box": [[round(x, 2), round(y, 2)] for x, y in mapped]})
                    x, y = i % 2 * 640 + 6, i // 2 * 312 + j * 72 + 6
                    draw.text((x, y), key, fill="black", font=font)
                    scale = min(622 / crop.width, 51 / crop.height, 4)
                    displayed = crop.resize((max(1, round(crop.width * scale)),
                                             max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
                    frame.paste(displayed, (x, y + 17))
            y = i // 2 * 312 + 299
            draw.line((i % 2 * 640, y, i % 2 * 640 + 629, y), fill="black", width=2)
        frame.save(audit / f"expanded_{offset // 8 + 1:02d}.png")
    json_write(audit / "expanded.json", output)


if __name__ == "__main__":
    main()
