import json
import math
from collections import Counter, defaultdict

from PIL import Image

from build_labels import ROOT, encoder_flags, json_write, normalize
from prepare import pixel_sha


def audit_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    audit = ROOT / "audit/visual"
    pseudo_review = audit_read(audit / "pseudo_review_v2.json")
    visual_choices = audit_read(audit / "direct_readings.json")
    candidate_by_id = {r["key"]: r for r in audit_read(audit / "expanded.json")}
    additions = defaultdict(list)
    for key, text in visual_choices.items():
        receipt, suffix = key.split("_")
        candidate = candidate_by_id[f"receipt_{receipt}_det_{suffix}"]
        additions[candidate["receipt_id"]].append(candidate | {"text": text})
    for item in audit_read(audit / "manual_boxes.json"):
        additions[item["receipt_id"]].append(item | {"audit_crop": None, "rotation_cw": 0})
    char_set = set((ROOT / "characters.txt").read_text(encoding="utf-8").splitlines()) | {" "}
    original_dir = ROOT / "labels"
    staging = ROOT / "labels_final"
    archive = ROOT / "audit/labels_before_visual_review"
    if (original_dir.resolve().parent != ROOT or staging.resolve().parent != ROOT
            or archive.resolve().parent != ROOT / "audit" or archive.exists()):
        raise ValueError("Unexpected label directory or prior audit version exists")
    staging.mkdir(exist_ok=False)
    changes = []
    seen = set()
    for source in sorted(original_dir.glob("*.json")):
        receipt = audit_read(source)
        rid = receipt["receipt_id"]
        lines = receipt["lines"]
        for line in lines:
            key = line["line_id"].removeprefix("receipt_").replace("_line_", "_")
            if key not in pseudo_review:
                continue
            if line["status"] != "pseudo" or receipt["split"] != "train":
                raise ValueError(f"Unexpected sampled candidate: {line['line_id']}")
            seen.add(key)
            transcription = pseudo_review[key]
            old_value = line["text"]
            if transcription is None:
                line["status"] = "needs_review"
                line["hold_reasons"].append("visual_cropped_or_mixed")
                action = "held_by_visual_review"
            else:
                line.update(source="direct_visual_review", status="verified",
                            text=normalize(transcription),
                            hold_reasons=encoder_flags(transcription, char_set))
                action = "visual_match" if transcription == old_value else "visual_transcription_corrected"
            changes.append(dict(line_id=line["line_id"], action=action,
                                before=old_value, after=transcription))
        for item in additions.get(rid, []):
            raw_text = item["text"]
            text = normalize(raw_text)
            line_id = f"{rid}_line_{len(lines) + 1:03d}"
            rel = f"crops/train/{line_id}.png"
            box = item["box"]
            if item["audit_crop"]:
                with Image.open(ROOT / item["audit_crop"]) as source_crop:
                    crop = source_crop.convert("RGB")
                method = "expanded_detected_box_visual_transcription"
            else:
                crop_width = round(math.dist(box[0], box[1]))
                crop_height = round(math.dist(box[0], box[3]))
                with Image.open(ROOT / receipt["image"]) as frame:
                    crop = frame.transform((crop_width, crop_height), Image.Transform.QUAD,
                                           [p for i in (0, 3, 2, 1) for p in box[i]],
                                           resample=Image.Resampling.BICUBIC)
                method = "manual_box_and_visual_transcription"
            target = ROOT / rel
            if target.exists():
                with Image.open(target) as existing:
                    if pixel_sha(existing) != pixel_sha(crop):
                        raise ValueError(f"A different crop already exists: {rel}")
            else:
                with target.open("xb") as handle:
                    crop.save(handle, "PNG", compress_level=3)
            holds = encoder_flags(text, char_set)
            flags = (["long"] if len(text) >= 24 else []) + (["oov"] if "out_of_vocabulary" in holds else [])
            lines.append(dict(source="direct_visual_review", status="verified", text=text, box=box,
                              line_id=line_id, flags=flags, hold_reasons=holds, crop=rel,
                              crop_size=list(crop.size), crop_pixel_sha256=pixel_sha(crop),
                              image_rotation_cw=item["rotation_cw"], visual_method=method))
            changes.append(dict(line_id=line_id, image=receipt["image"], action=method, after=text))
        json_write(staging / source.name, receipt)
    if seen != set(pseudo_review):
        raise ValueError("Some visual sample decisions were not applied")
    original_dir.rename(archive)
    staging.rename(original_dir)
    json_write(audit / "changes.json", changes)
    print(dict(Counter(r["action"] for r in changes)))
    print("Receipts read from re-cropped images:", len(additions))


if __name__ == "__main__":
    main()
