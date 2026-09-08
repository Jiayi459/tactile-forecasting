#!/usr/bin/env python3
"""Plot the ActionSense corpus per-action point estimates used in manuscript Sec. 6.

The plot reads the shared-scorer CSVs directly.  It intentionally uses only the
3 s physical-state runs, because no full-corpus map run or per-action confidence
interval has been produced.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "actionsense" / "results" / "corpus-aggregate"


def read_rows(path: Path) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
    model: dict[str, dict[str, float | int]] = {}
    persistence: dict[str, dict[str, float | int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            action = row["action"]
            if action == "(all actions)":
                continue
            values: dict[str, float | int] = {
                "n": int(row["n_clips"]),
                "r2": float(row["r2"]),
                "skill": float(row["skill"]),
                "hd_ratio": float(row["hausdorff_ratio"]),
            }
            (persistence if row["model"] == "persistence" else model)[action] = values
    return model, persistence


def read_traits(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {row["verb"]: row["trait_class"][0].upper() for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "action_predictability")
    args = parser.parse_args()

    seq2seq, persistence = read_rows(RESULTS / "as_preds_seq2seq_corpus.csv")
    probgru, persistence_pg = read_rows(RESULTS / "as_preds_probgru_corpus.csv")
    if seq2seq.keys() != probgru.keys() or persistence.keys() != persistence_pg.keys():
        raise SystemExit("the two 3 s runs do not cover the same actions")
    for action in seq2seq:
        if seq2seq[action]["n"] != probgru[action]["n"]:
            raise SystemExit(f"recording count differs for {action}")

    actions = sorted(seq2seq, key=lambda a: float(seq2seq[a]["r2"]), reverse=True)
    traits = read_traits(ROOT / "docs" / "actionsense" / "trait_partition.csv")
    if not set(actions) <= traits.keys():
        raise SystemExit("an action has no prespecified smooth/abrupt label")
    labels = [f"{action} [{traits[action]}] ({int(seq2seq[action]['n'])})" for action in actions]

    blue, orange, gray = "#0072B2", "#D55E00", "#666666"
    pale, black = "#DDDDDD", "#222222"
    width, height = 2115, 1035  # 7.05 x 3.45 in at 300 dpi
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font = ImageFont.truetype(regular, 22)
    small = ImageFont.truetype(regular, 20)
    title_font = ImageFont.truetype(bold, 25)

    left, top, bottom, gap = 375, 82, 900, 42
    panel_width = (width - left - 25 - 2 * gap) // 3
    row_step = (bottom - top) / len(actions)
    panels = (
        ("R²", "r2", 0.10, 0.78, None, (0.2, 0.4, 0.6)),
        ("Skill vs. persistence", "skill", -1.05, 0.32, 0.0, (-1.0, -0.5, 0.0)),
        ("Hausdorff ratio", "hd_ratio", 0.77, 1.05, 1.0, (0.8, 0.9, 1.0)),
    )

    def xcoord(value: float, x0: int, lo: float, hi: float) -> float:
        return x0 + (value - lo) / (hi - lo) * panel_width

    def dashed_vertical(x: float) -> None:
        yy = top
        while yy < bottom:
            draw.line((x, yy, x, min(yy + 12, bottom)), fill=gray, width=2)
            yy += 22

    for panel_idx, (title, key, lo, hi, reference, ticks) in enumerate(panels):
        x0 = left + panel_idx * (panel_width + gap)
        draw.text((x0 + panel_width / 2, 28), title, fill=black, font=title_font, anchor="ma")
        draw.line((x0, top, x0, bottom), fill=black, width=2)
        draw.line((x0, bottom, x0 + panel_width, bottom), fill=black, width=2)
        for tick in ticks:
            xx = xcoord(tick, x0, lo, hi)
            draw.line((xx, top, xx, bottom), fill=pale, width=2)
            draw.line((xx, bottom, xx, bottom + 8), fill=black, width=2)
            draw.text((xx, bottom + 13), f"{tick:.1f}", fill=black, font=small, anchor="ma")
        if reference is not None:
            dashed_vertical(xcoord(reference, x0, lo, hi))

        for row_idx, action in enumerate(actions):
            yy = top + (row_idx + 0.5) * row_step
            if key == "r2":
                xx = xcoord(float(persistence[action][key]), x0, lo, hi)
                draw.line((xx - 5, yy - 5, xx + 5, yy + 5), fill=gray, width=3)
                draw.line((xx - 5, yy + 5, xx + 5, yy - 5), fill=gray, width=3)
            xx = xcoord(float(seq2seq[action][key]), x0, lo, hi)
            draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill="white", outline=blue, width=3)
            xx = xcoord(float(probgru[action][key]), x0, lo, hi)
            draw.rectangle((xx - 6, yy - 6, xx + 6, yy + 6), fill="white", outline=orange, width=3)

    for row_idx, label in enumerate(labels):
        yy = top + (row_idx + 0.5) * row_step
        draw.text((left - 16, yy), label, fill=black, font=font, anchor="rm")

    legend_y = 990
    legend = (("Persistence", gray, "x"), ("Seq2Seq", blue, "o"), ("ProbGRU", orange, "s"))
    starts = (705, 1015, 1300)
    for (name, color, marker), xx in zip(legend, starts):
        if marker == "x":
            draw.line((xx - 6, legend_y - 6, xx + 6, legend_y + 6), fill=color, width=3)
            draw.line((xx - 6, legend_y + 6, xx + 6, legend_y - 6), fill=color, width=3)
        elif marker == "o":
            draw.ellipse((xx - 7, legend_y - 7, xx + 7, legend_y + 7), fill="white", outline=color, width=3)
        else:
            draw.rectangle((xx - 7, legend_y - 7, xx + 7, legend_y + 7), fill="white", outline=color, width=3)
        draw.text((xx + 16, legend_y), name, fill=black, font=font, anchor="lm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out.with_suffix(".png"), dpi=(300, 300), optimize=True)


if __name__ == "__main__":
    main()
