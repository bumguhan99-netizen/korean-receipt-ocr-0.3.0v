import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit/visual"
OUT.mkdir(exist_ok=False)
font = ImageFont.truetype("arial.ttf", 16)


def draw_crop(canvas, draw, line, x, y, width):
    draw.text((x + 6, y + 3), line["line_id"], fill="black", font=font)
    with Image.open(ROOT / line["crop"]) as crop:
        scale = min((width - 14) / crop.width, 50 / crop.height, 4)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        zoom = crop.resize(size, Image.Resampling.LANCZOS)
        canvas.paste(zoom, (x + 6, y + 24))
    draw.line((x + 6, y + 76, x + width - 8, y + 76), fill="#cccccc")


def score(line):
    text = line.get("paddle_text") or ""
    checks = (
        0.1 if line["paddle_text"] == line["windows_text"] else 0,
        0.15 if any("가" <= char <= "힣" for char in text) else 0,
        0.10 if 2 <= len(text) <= 14 else 0,
        0.10 if not any(c in text for c in "?*|") else 0,
    )
    return line.get("paddle_confidence", 0) + sum(checks)


files = sorted((ROOT / "labels").glob("*.json"))
receipts = [json.loads(p.read_text(encoding="utf-8")) for p in files]
missing = [r for r in receipts if not r["previous_receipt_id"] and
           not any(x["status"] == "pseudo" for x in r["lines"])]
for offset in range(0, len(missing), 8):
    batch = missing[offset:offset + 8]
    canvas = Image.new("RGB", (1250, 4 * 255), "white")
    draw = ImageDraw.Draw(canvas)
    for i, receipt in enumerate(batch):
        x, y = (i % 2) * 625, (i // 2) * 255
        options = [v for v in receipt["lines"] if v["crop"] and v.get("paddle_text") and
                   "small_or_vertical_crop" not in v["hold_reasons"]]
        selected = sorted(options, key=score, reverse=True)[:3]
        for j, line in enumerate(selected):
            draw_crop(canvas, draw, line, x, y + j * 80, 610)
        draw.line((x, y + 242, x + 600, y + 242), fill="#111111", width=2)
    canvas.save(OUT / f"uncovered_{offset // 8 + 1:02d}.png")

rng = random.Random(20261123)
pseudo = [v for r in receipts for v in r["lines"] if v["status"] == "pseudo"]
verified = [v for r in receipts for v in r["lines"] if v["status"] == "verified" and r["split"] in {"val", "test"}]
for name, items, n in (("pseudo", pseudo, 36), ("evaluation", verified, 18)):
    selected = rng.sample(items, n)
    for index in range(0, n, 18):
        canvas = Image.new("RGB", (1350, 6 * 82), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(selected[index:index + 18]):
            draw_crop(canvas, draw, row, i % 3 * 450, i // 3 * 82, 445)
        canvas.save(OUT / f"{name}_{index // 18 + 1:02d}.png")
