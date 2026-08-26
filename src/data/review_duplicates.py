"""Visual contact sheets for cross-class duplicate review groups.

Reads ONLY from raw_dir (source images are opened read-only and never
modified) and from the review manifest produced by
src/data/audit_dataset.py (`data/audit/cross_class_duplicate_review.csv`).
Generated contact sheets are written under data/audit/duplicate_review/ —
never under data/raw/.

Usage:
    python -m src.data.review_duplicates --group-id DUPGROUP_0001
    python -m src.data.review_duplicates --all --unresolved-only
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFont

from src.data.duplicate_review import RESOLUTION_UNRESOLVED, load_review_csv, parse_members_field
from src.utils.config import load_config

THUMB_SIZE = (260, 260)
PADDING = 14
LABEL_LINE_HEIGHT = 14
LABEL_HEIGHT = LABEL_LINE_HEIGHT * 3 + 8
HEADER_HEIGHT = 28


def _load_font():
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover - Pillow always ships a default font
        return None


def render_contact_sheet_for_group(
    row: Dict[str, Any], raw_dir: Path, output_dir: Path
) -> Path:
    """Render one duplicate group's members side by side with labels.

    Never writes into raw_dir; only reads source images from it. Returns
    the path to the generated PNG under output_dir.
    """
    group_id = row["duplicate_group_id"]
    sha256 = row.get("sha256", "")
    resolution = row.get("resolution", RESOLUTION_UNRESOLVED)
    members = parse_members_field(row.get("paths", ""))
    if not members:
        raise ValueError(f"review row for group '{group_id}' has no members in its 'paths' field")

    font = _load_font()
    cell_w = THUMB_SIZE[0] + PADDING
    cell_h = THUMB_SIZE[1] + LABEL_HEIGHT
    canvas_w = cell_w * len(members) + PADDING
    canvas_h = HEADER_HEIGHT + cell_h + PADDING

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header = f"{group_id}  sha256={sha256[:20]}...  resolution={resolution}"
    draw.text((PADDING, 6), header, fill=(0, 0, 0), font=font)

    x = PADDING
    y = HEADER_HEIGHT
    for canonical_class, rel_path in members:
        image_path = raw_dir / rel_path
        with Image.open(image_path) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail(THUMB_SIZE)
            thumb = thumb.copy()

        paste_x = x + (THUMB_SIZE[0] - thumb.width) // 2
        paste_y = y + (THUMB_SIZE[1] - thumb.height) // 2
        canvas.paste(thumb, (paste_x, paste_y))

        filename = Path(rel_path).name
        label_y = y + THUMB_SIZE[1] + 4
        for line in (canonical_class, filename, rel_path):
            draw.text((x, label_y), line, fill=(0, 0, 0), font=font)
            label_y += LABEL_LINE_HEIGHT

        x += cell_w

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{group_id}.png"
    canvas.save(output_path, format="PNG")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.review_duplicates",
        description="Generate visual contact sheets for cross-class duplicate review groups.",
    )
    parser.add_argument(
        "--group-id", type=str, default=None, help="Render a single group by its duplicate_group_id."
    )
    parser.add_argument(
        "--all", action="store_true", help="Render every cross-class duplicate group in the review manifest."
    )
    parser.add_argument(
        "--unresolved-only",
        action="store_true",
        help="With --all, only render groups whose resolution is still UNRESOLVED.",
    )
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    parser.add_argument(
        "--review-csv", type=str, default=None, help="Override path to cross_class_duplicate_review.csv."
    )
    parser.add_argument("--raw-dir", type=str, default=None, help="Override the raw dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the contact-sheet output directory.")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.group_id and not args.all:
        parser.error("either --group-id or --all is required")

    config = load_config(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(config["paths"]["raw_dir"])
    audit_dir = Path(config["paths"]["audit_dir"])
    review_csv = Path(args.review_csv) if args.review_csv else audit_dir / "cross_class_duplicate_review.csv"
    output_dir = Path(args.output_dir) if args.output_dir else audit_dir / "duplicate_review"

    rows = load_review_csv(review_csv)
    if not rows:
        print(
            f"No review rows found in {review_csv}. Run `python run_pipeline.py audit` first.",
            file=sys.stderr,
        )
        return 1

    if args.group_id:
        rows = [r for r in rows if r["duplicate_group_id"] == args.group_id]
        if not rows:
            print(f"group_id '{args.group_id}' not found in {review_csv}", file=sys.stderr)
            return 1

    if args.unresolved_only:
        rows = [r for r in rows if r.get("resolution") == RESOLUTION_UNRESOLVED]

    for row in rows:
        output_path = render_contact_sheet_for_group(row, raw_dir, output_dir)
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
