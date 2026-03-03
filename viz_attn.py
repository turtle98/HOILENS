"""
viz_attn.py — visualize attention maps saved by ours_qwen_hoi_sft.py

Usage:
    python viz_attn.py attn_map_b0.npz
    python viz_attn.py attn_map_b0.npz --out my_fig.png

    # Overlay HO pair #2 attention on the image:
    python viz_attn.py attn_map_b0.npz --pair 2 --image path/to/image.jpg
    python viz_attn.py attn_map_b0.npz --pair 2 --image path/to/image.jpg --out pair2_attn.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def add_region_box(ax, row_start, row_end, col_start, col_end, label, color):
    rect = patches.Rectangle(
        (col_start - 0.5, row_start - 0.5),
        col_end - col_start, row_end - row_start,
        linewidth=1.5, edgecolor=color, facecolor='none', label=label
    )
    ax.add_patch(rect)


def viz_pair_on_image(ax, attn_row, img, title):
    """
    attn_row : 1-D array of length n_img_tokens (e.g. 256 = 16×16)
    img      : PIL.Image  (will be resized to display size)
    """
    n = len(attn_row)
    # infer grid size (assume square)
    grid = int(round(n ** 0.5))
    assert grid * grid == n, f"Image token count {n} is not a perfect square"

    heat = attn_row.reshape(grid, grid)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)   # normalize 0→1

    # upsample heatmap to image size with PIL
    h, w = img.size[1], img.size[0]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    heat_arr = np.asarray(heat_img) / 255.0   # [H, W] in [0,1]

    ax.imshow(img)
    ax.imshow(heat_arr, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03)
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="path to attn_map_b*.npz")
    parser.add_argument("--out",   default=None, help="output png path (default: <npz>.png)")
    parser.add_argument("--pair",  type=int, default=None,
                        help="HO pair index to visualize as heatmap on image")
    parser.add_argument("--image", default=None,
                        help="path to input image (required with --pair)")
    args = parser.parse_args()

    data      = np.load(args.npz)
    attn      = data["attn_map"]          # [total_len, total_len]
    q_len     = int(data["q_len"])
    hoi_start = int(data["hoi_start"])
    hoi_end   = int(data["hoi_end"])
    img_start = int(data["img_start"])
    img_end   = int(data["img_end"])
    total_len = attn.shape[0]
    a_len     = total_len - q_len

    print(f"total_len={total_len}  q_len={q_len}  a_len={a_len}")
    print(f"img_start={img_start}  img_end={img_end}  ({img_end-img_start} tokens)")
    print(f"hoi_start={hoi_start}  hoi_end={hoi_end}  ({hoi_end-hoi_start} pairs)")

    # ── Pair heatmap mode ─────────────────────────────────────────────────────
    if args.pair is not None:
        if args.image is None:
            parser.error("--image is required when --pair is specified")

        pair_idx = args.pair
        row = hoi_start + pair_idx
        assert 0 <= pair_idx < (hoi_end - hoi_start), \
            f"pair index {pair_idx} out of range [0, {hoi_end-hoi_start})"

        attn_row = attn[row, img_start:img_end]   # [n_img_tokens]
        img = Image.open(args.image).convert("RGB")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # left: heatmap overlay on image
        viz_pair_on_image(
            axes[0], attn_row, img,
            f"HO pair #{pair_idx} attention on image\n"
            f"(row {row}, {img_end-img_start} img tokens → "
            f"{int(round((img_end-img_start)**0.5))}×{int(round((img_end-img_start)**0.5))} grid)"
        )

        # right: 1-D attention profile across image tokens
        axes[1].bar(range(len(attn_row)), attn_row, width=1.0)
        axes[1].set_title(f"HO pair #{pair_idx} → image token attention weights")
        axes[1].set_xlabel("Image token index (raster order)")
        axes[1].set_ylabel("Attention weight")

        plt.tight_layout()
        out = args.out or args.npz.replace(".npz", f"_pair{pair_idx}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()
        return

    # ── Default: full 3-panel overview ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Full map with region annotations
    ax = axes[0]
    im = ax.imshow(attn, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03)
    add_region_box(ax, img_start, img_end, img_start, img_end,   "image×image", "cyan")
    add_region_box(ax, hoi_start, hoi_end, img_start, img_end,   "HOI→image",   "red")
    add_region_box(ax, q_len, total_len,   hoi_start, hoi_end,   "ans→HOI",     "lime")
    ax.axhline(q_len - 0.5, color="white", lw=0.8, linestyle="--")
    ax.axvline(q_len - 0.5, color="white", lw=0.8, linestyle="--")
    ax.set_title("Full attention map (last layer, avg heads)")
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    ax.legend(loc="upper left", fontsize=7)

    # 2. HOI tokens → image tokens
    ax = axes[1]
    sub = attn[hoi_start:hoi_end, img_start:img_end]
    im2 = ax.imshow(sub, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(im2, ax=ax, fraction=0.03)
    ax.set_title(f"HOI feat tokens → image tokens\n({hoi_end-hoi_start} pairs × {img_end-img_start} img toks)")
    ax.set_xlabel("Image token index"); ax.set_ylabel("HOI pair index")

    # 3. Answer tokens → HOI tokens
    ax = axes[2]
    if a_len > 0:
        sub2 = attn[q_len:, hoi_start:hoi_end]
        im3  = ax.imshow(sub2, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(im3, ax=ax, fraction=0.03)
        ax.set_title(f"Answer tokens → HOI feat tokens\n({a_len} ans toks × {hoi_end-hoi_start} pairs)")
        ax.set_xlabel("HOI pair index"); ax.set_ylabel("Answer token index")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    out = args.out or args.npz.replace(".npz", ".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
