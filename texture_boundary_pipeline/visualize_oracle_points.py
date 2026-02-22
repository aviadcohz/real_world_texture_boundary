"""
Visualize oracle point prompts for a pipeline run.

For each crop in processed_bboxes.json produces a figure with 5 columns:
  col 0 - Original source image with bbox drawn + oracle points mapped back
           to original image coordinates (so you see exactly what goes to SAM)
  col 1 - Crop (1024×1024) with oracle points in crop space
  col 2 - Mask A  (binary) with mask_a oracle points
  col 3 - Mask B  (binary) with mask_b oracle points
  col 4 - Boundary mask

Oracle points are in [x, y] format in the 1024×1024 crop space.
Reverse-mapping to original:
    orig_x = bbox_x1 + point_x * (bbox_x2 - bbox_x1) / 1024
    orig_y = bbox_y1 + point_y * (bbox_y2 - bbox_y1) / 1024

Rows are grouped into pages of PAGE_SIZE (default 16).
Output: <run_dir>/oracle_points_viz_page_<N>.png

Usage:
    python visualize_oracle_points.py <run_dir>
    python visualize_oracle_points.py  # uses default path below
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── colours ───────────────────────────────────────────────────────────────────
COLOUR_A    = "#00cfff"   # cyan  – mask A oracle points
COLOUR_B    = "#ff4040"   # red   – mask B oracle points
EDGE_A      = "#0055aa"
EDGE_B      = "#880000"
BBOX_COLOUR = "#ffdd00"   # yellow – bounding box on original image
MARKER_SIZE = 12
EDGE_WIDTH  = 2

PAGE_SIZE = 16   # rows per output figure


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.array(Image.open(path).convert("RGB"))
    return None


def _load_gray(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.array(Image.open(path).convert("L"))
    return None


def _plot_points(ax, points, colour, edge_colour, label_prefix, marker_size=None):
    """Plot oracle points with outlined labels."""
    ms = marker_size or MARKER_SIZE
    outline = [pe.withStroke(linewidth=2, foreground="black")]
    for i, (x, y) in enumerate(points):
        ax.plot(x, y,
                marker="o", color=colour,
                markersize=ms,
                markeredgecolor=edge_colour,
                markeredgewidth=EDGE_WIDTH,
                zorder=5)
        ax.text(x + 14, y, f"{label_prefix}{i + 1}",
                color=colour, fontsize=8, fontweight="bold",
                zorder=6, path_effects=outline)


def _map_points_to_orig(points, coords):
    """
    Map oracle points from 1024×1024 crop space to original image coordinates.

    Args:
        points: list of [x, y] in crop space (0..1023)
        coords: [x1, y1, x2, y2] bbox in original image

    Returns:
        list of [orig_x, orig_y]
    """
    x1, y1, x2, y2 = coords
    crop_w = x2 - x1
    crop_h = y2 - y1
    return [
        [x1 + px * crop_w / 1024, y1 + py * crop_h / 1024]
        for px, py in points
    ]


# ── page renderer ─────────────────────────────────────────────────────────────

def _render_page(rows, page_num, output_dir, col_titles):
    n_rows = len(rows)
    fig_h  = max(4 * n_rows, 4)

    fig, axes = plt.subplots(
        n_rows, 5,
        figsize=(28, fig_h),
        gridspec_kw={"wspace": 0.03, "hspace": 0.40},
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=4)

    for row_idx, entry in enumerate(rows):
        box        = entry["box"]
        orig_img   = entry["orig_img"]
        crop_img   = entry["crop_img"]
        mask_a_img = entry["mask_a_img"]
        mask_b_img = entry["mask_b_img"]
        bnd_img    = entry["bnd_img"]

        oracle_pts  = box.get("oracle_points")
        coords      = box.get("coords")          # [x1, y1, x2, y2]
        description = box.get("description", "—")
        category    = box.get("crop_category", "")

        axs = axes[row_idx]

        # ── col 0 : original image + bbox + reverse-mapped oracle points ──────
        ax = axs[0]
        if orig_img is not None:
            ax.imshow(orig_img)
            if coords:
                x1, y1, x2, y2 = coords
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=BBOX_COLOUR,
                    facecolor="none", zorder=4,
                )
                ax.add_patch(rect)
                # small label inside bbox
                ax.text(x1 + 4, y1 + 14, category,
                        color=BBOX_COLOUR, fontsize=7, fontweight="bold",
                        zorder=5,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="black")])
            if oracle_pts and coords:
                mapped_a = _map_points_to_orig(
                    oracle_pts["point_prompt_mask_a"], coords)
                mapped_b = _map_points_to_orig(
                    oracle_pts["point_prompt_mask_b"], coords)
                _plot_points(ax, mapped_a, COLOUR_A, EDGE_A, "A",
                             marker_size=10)
                _plot_points(ax, mapped_b, COLOUR_B, EDGE_B, "B",
                             marker_size=10)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "original\nmissing",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

        ax.set_ylabel(f"[{category}]\n{description}",
                      fontsize=6.5, rotation=0, ha="right", va="center",
                      labelpad=85)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── col 1 : crop (1024 px space) + oracle points ──────────────────────
        ax = axs[1]
        if crop_img is not None:
            ax.imshow(crop_img)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "crop missing",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        if oracle_pts:
            _plot_points(ax, oracle_pts["point_prompt_mask_a"],
                         COLOUR_A, EDGE_A, "A")
            _plot_points(ax, oracle_pts["point_prompt_mask_b"],
                         COLOUR_B, EDGE_B, "B")
        ax.axis("off")

        # ── col 2 : mask A + A points ─────────────────────────────────────────
        ax = axs[2]
        if mask_a_img is not None:
            ax.imshow(mask_a_img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "no mask",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        if oracle_pts:
            _plot_points(ax, oracle_pts["point_prompt_mask_a"],
                         COLOUR_A, EDGE_A, "A")
        ax.axis("off")

        # ── col 3 : mask B + B points ─────────────────────────────────────────
        ax = axs[3]
        if mask_b_img is not None:
            ax.imshow(mask_b_img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "no mask",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        if oracle_pts:
            _plot_points(ax, oracle_pts["point_prompt_mask_b"],
                         COLOUR_B, EDGE_B, "B")
        ax.axis("off")

        # ── col 4 : boundary mask ─────────────────────────────────────────────
        ax = axs[4]
        if bnd_img is not None:
            ax.imshow(bnd_img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "no mask",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        ax.axis("off")

    # legend + bbox indicator
    from matplotlib.lines import Line2D
    legend_elems = [
        mpatches.Patch(edgecolor=BBOX_COLOUR, facecolor="none",
                       linewidth=2, label="Bounding box (original coords)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOUR_A,
               markeredgecolor=EDGE_A, markersize=MARKER_SIZE,
               label="Mask A oracle pts"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOUR_B,
               markeredgecolor=EDGE_B, markersize=MARKER_SIZE,
               label="Mask B oracle pts"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

    out_path = output_dir / f"oracle_points_viz_page_{page_num:02d}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved page {page_num}: {out_path}")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def visualize_oracle_points(run_dir: str | Path, page_size: int = PAGE_SIZE):
    run_dir = Path(run_dir)
    json_path = run_dir / "processed_bboxes.json"

    with open(json_path) as f:
        data = json.load(f)

    col_titles = [
        "Original + bbox + mapped pts",
        "Crop 1024px (cyan=A | red=B)",
        "Mask A",
        "Mask B",
        "Boundary mask",
    ]

    # Cache original images so each source image is loaded only once
    orig_cache: dict[str, np.ndarray | None] = {}

    rows = []
    for img_entry in data:
        img_path = img_entry.get("image_path", "")
        if img_path not in orig_cache:
            orig_cache[img_path] = _load(Path(img_path))

        for box in img_entry["boxes"]:
            crop_name = box["crop_name"]
            category  = box["crop_category"]
            stem      = Path(crop_name).stem

            crop_img   = _load     (run_dir / "crops"          / category / crop_name)
            mask_a_img = _load_gray(run_dir / "masks_textures" / category / f"{stem}_mask_a.png")
            mask_b_img = _load_gray(run_dir / "masks_textures" / category / f"{stem}_mask_b.png")
            bnd_img    = _load_gray(run_dir / "masks"          / category / f"{stem}.png")

            rows.append({
                "box":        box,
                "orig_img":   orig_cache[img_path],
                "crop_img":   crop_img,
                "mask_a_img": mask_a_img,
                "mask_b_img": mask_b_img,
                "bnd_img":    bnd_img,
            })

    print(f"Total crops: {len(rows)}")
    with_pts = sum(1 for r in rows if r["box"].get("oracle_points"))
    print(f"  With oracle points   : {with_pts}")
    print(f"  Without oracle points: {len(rows) - with_pts}")

    saved = []
    for page_num, start in enumerate(range(0, len(rows), page_size), start=1):
        page_rows = rows[start: start + page_size]
        out = _render_page(page_rows, page_num, run_dir, col_titles)
        saved.append(out)

    print(f"\nDone — {len(saved)} page(s) saved to {run_dir}")
    return saved


if __name__ == "__main__":
    default_run = "/home/aviad/results/debug_for_training/run_20260222_140936"
    run_dir = sys.argv[1] if len(sys.argv) > 1 else default_run
    visualize_oracle_points(run_dir)
