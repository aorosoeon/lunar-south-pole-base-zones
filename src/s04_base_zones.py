"""Sprint 4: candidate base zones from temperatures + terrain, Diviner reference vs CNN prediction.

Rule (applied per grid cell, ~0.5 km):
  gentle   : steeper of the 2 triangles <= 8 deg (Artemis III landing-slope limit)
  sunlit   : colder of the 2 triangles has temp_max >= 200 K
  cold trap: any triangle with temp_max < 110 K (classic limit for surface water ice to stay stable)
  base zone: gentle AND sunlit AND within 2 km of a cold trap, in patches of at least 1 km^2
The same rule runs on the Diviner temperatures (reference) and on the CNN's predicted temperatures.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, ListedColormap
from scipy import ndimage

from common import N, ROOT, load_feat

CELL_KM, SLOPE_MAX, SUN_MIN, COLD_MAX, NEAR_KM, MIN_KM2 = 0.5, 8.0, 200.0, 110.0, 2.0, 1.0
f = load_feat()
ice = pd.read_parquet(ROOT / "data" / "south.parquet", columns=["ice_depth"]).ice_depth.to_numpy()
y = f.temp_max.to_numpy()
p = np.load(ROOT / "data" / "s03b_preds_cnn_aug.npz")["cnn_aug_tta"]


def cells(v, how):
    a = v.reshape(N, N, 2)
    return getattr(a, how)(2)


slope_c = cells(f.slope.to_numpy(), "max")
split_c = f.split.to_numpy().reshape(N, N, 2)[:, :, 0]
elev_c = cells(f.elev.to_numpy(), "mean")


def zones(temp, cold_max=COLD_MAX, sun_min=SUN_MIN):
    cold = cells(temp, "min") < cold_max
    near = ndimage.distance_transform_edt(~cold) * CELL_KM <= NEAR_KM
    z = (slope_c <= SLOPE_MAX) & (cells(temp, "min") >= sun_min) & near
    lab, n = ndimage.label(z)
    size = ndimage.sum(z, lab, range(1, n + 1)) * CELL_KM ** 2
    keep = np.isin(lab, 1 + np.where(size >= MIN_KM2)[0])
    return keep, cold


z_ref, cold_ref = zones(y)

# Sanity check: does "temp_max < 110 K" match the dataset's own ice model?
# ice_depth uses an undocumented code -1 in the coldest areas (all triangles below 100 K); the label only
# documents 0 (stable at the surface) and -999 (deeper than 2.87 m). We read -1 as "stable at the surface".
print("ice_depth codes:", {v: int((ice == v).sum()) for v in (-999, -1, 0)})
ice0 = cells((ice <= 0) & (ice > -999), "max")
print(f"cold-trap rule (< 110 K) vs surface-stable ice (ice_depth -1 or 0): IoU {(cold_ref & ice0).sum() / (cold_ref | ice0).sum():.2f}")

# The CNN smooths shadow edges, so its temperatures in cold traps run warm. Choose the model's two
# thresholds on the VALIDATION blocks only (best base-zone IoU vs the reference), then score on TEST.
val_c = split_c == 1
best = max(((tc, ts) for tc in range(110, 171, 10) for ts in range(160, 231, 10)),
           key=lambda t: ((zones(p, *t)[0] & z_ref)[val_c].sum() / (zones(p, *t)[0] | z_ref)[val_c].sum()))
print(f"model thresholds chosen on validation: cold trap < {best[0]} K, sunlit >= {best[1]} K")
z_cnn, cold_cnn = zones(p, *best)
z_raw, cold_raw = zones(p)

lab, n_ref = ndimage.label(z_ref)
print(f"\nreference base zones (whole map): {z_ref.sum() * CELL_KM**2:,.0f} km2 in {n_ref} patches "
      f"({z_ref.mean():.1%} of the map); cold traps {cold_ref.sum() * CELL_KM**2:,.0f} km2")


def agree(a, b, m):
    a, b = a[m], b[m]
    tp = (a & b).sum()
    return {"reference km2": b.sum() * CELL_KM ** 2, "model km2": a.sum() * CELL_KM ** 2,
            "recall": tp / b.sum(), "precision": tp / a.sum(), "IoU": tp / (a | b).sum()}


rows = []
for name, m in [("test", split_c == 2), ("val", split_c == 1), ("train (in-sample)", split_c == 0)]:
    rows.append({"area": name, "what": "base zones, same thresholds", **agree(z_raw, z_ref, m)})
    rows.append({"area": name, "what": "base zones, val-tuned", **agree(z_cnn, z_ref, m)})
    rows.append({"area": name, "what": "cold traps, val-tuned", **agree(cold_cnn, cold_ref, m)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "s04_zone_agreement.csv", index=False)
print("\nCNN zones vs Diviner zones (recall = share of true area found, precision = share of model area that is true):")
print(out.round(3).to_string(index=False))


# ---- Figures (standard south-pole view: img = a[::-1, :]) -------------------------
def img(a):
    return a[::-1, :]


EXT = [-300, 300, -300, 300]
shade = LightSource(azdeg=315, altdeg=35).hillshade(img(elev_c), vert_exag=40, dx=CELL_KM, dy=CELL_KM)

fig, ax = plt.subplots(figsize=(9, 8.2), constrained_layout=True)
ax.imshow(shade, origin="lower", extent=EXT, cmap="gray", vmin=0, vmax=1.3)
ax.imshow(np.ma.masked_where(~img(cold_ref), np.ones_like(shade)), origin="lower", extent=EXT,
          cmap=ListedColormap(["#2b6cb0"]), alpha=0.85)
ax.imshow(np.ma.masked_where(~img(z_ref), np.ones_like(shade)), origin="lower", extent=EXT,
          cmap=ListedColormap(["#38d46a"]), alpha=0.95)
for lon, lab_ in [(0, "0°"), (90, "90°E"), (180, "180°"), (270, "90°W")]:
    r = 1737.4 * np.cos(np.radians(80.2))
    ax.text(r * np.sin(np.radians(lon)), r * np.cos(np.radians(lon)), lab_, fontsize=8, ha="center", va="center",
            bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
ax.legend(handles=[plt.matplotlib.patches.Patch(color="#38d46a", label="candidate base zone: slope ≤ 8°, sunlit (≥ 200 K), ≤ 2 km from a cold trap"),
                   plt.matplotlib.patches.Patch(color="#2b6cb0", label="cold trap: annual max < 110 K (surface ice can stay stable)")],
          loc="lower left", fontsize=8, framealpha=0.9)
ax.set_xlabel("km from pole"); ax.set_ylabel("km from pole")
ax.set_title("Lunar south pole: candidate base zones from Diviner temperatures and terrain")
fig.savefig(ROOT / "figures" / "s04_base_zones_reference.png", dpi=130)

# Model vs reference on held-out blocks near the pole
both, only_m, only_r = img(z_cnn & z_ref), img(z_cnn & ~z_ref), img(~z_cnn & z_ref)
held = img(split_c > 0)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), constrained_layout=True)
for ax, title, layers in [
        (axes[0], "Reference: Diviner temperatures", [(img(cold_ref), "#2b6cb0"), (img(z_ref), "#38d46a")]),
        (axes[1], f"Model: CNN-predicted temperatures, terrain only\n(thresholds tuned on validation: cold < {best[0]} K, sunlit ≥ {best[1]} K)", [(img(cold_cnn), "#2b6cb0"), (both, "#38d46a"),
                                                                      (only_m, "#f59e0b"), (only_r, "#e11d48")])]:
    ax.imshow(shade, origin="lower", extent=EXT, cmap="gray", vmin=0, vmax=1.3)
    for m, col in layers:
        ax.imshow(np.ma.masked_where(~m, np.ones_like(shade)), origin="lower", extent=EXT, cmap=ListedColormap([col]), alpha=0.9)
    ax.contour(held, levels=[0.5], origin="lower", extent=EXT, colors="c", linewidths=1.3)
    ax.set_xlim(-100, 100); ax.set_ylim(-60, 130); ax.set_title(title); ax.set_xlabel("km from pole")
axes[0].set_ylabel("km from pole")
axes[1].legend(handles=[plt.matplotlib.patches.Patch(color=c, label=l) for c, l in [
    ("#38d46a", "zone found by model and reference"), ("#f59e0b", "model only (false alarm)"),
    ("#e11d48", "reference only (missed)"), ("#2b6cb0", "cold trap")]] +
    [plt.matplotlib.lines.Line2D([], [], color="c", label="held-out blocks (no labels used)")], loc="lower left", fontsize=8, framealpha=0.9)
fig.savefig(ROOT / "figures" / "s04_base_zones_model_vs_reference.png", dpi=130)
print("\nfigures written")
