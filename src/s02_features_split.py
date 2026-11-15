"""Sprint 2: per-triangle features, CNN grid arrays, map orientation check, geographic split.

Input:  data/south.parquet (from s01)
Output: data/features.parquet  (one row per triangle: features, target, split)
        data/grid.npz          (CNN inputs/targets on the 1200 x 1200 cell grid, 2 triangles per cell)
        figures/s02_*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
N = 1200
df = pd.read_parquet(ROOT / "data" / "south.parquet")

# ---- 1. Features from triangle geometry -------------------------------------
V = df[[f"tri{v}_{a}" for v in (1, 2, 3) for a in "xyz"]].to_numpy().reshape(-1, 3, 3)
c = V.mean(1)
n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
n *= np.sign((n * c).sum(1, keepdims=True))           # point the normal away from the Moon centre
n /= np.linalg.norm(n, axis=1, keepdims=True)
up = c / np.linalg.norm(c, axis=1, keepdims=True)
lat, lon = np.radians(df.tri_clat.to_numpy()), np.radians(df.tri_clon.to_numpy())
east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], 1)

feat = pd.DataFrame({
    "elev": df.tri_calt,                                         # km vs 1737.4 km sphere
    "slope": np.degrees(np.arccos(np.clip((n * up).sum(1), -1, 1))),
    "n_north": (n * north).sum(1),   # > 0: faces north = towards the equator = towards the sun
    "n_east": (n * east).sum(1),
    "lat": df.tri_clat,
    "sin_lon": np.sin(lon), "cos_lon": np.cos(lon),
    "temp_max": df.temp_max,
    "gi": df.gi, "gj": df.gj, "tri_k": df.tri_k,
})
print(feat[["elev", "slope", "n_north", "n_east"]].describe(percentiles=[0.5, 0.99]).T[["min", "50%", "99%", "max"]].round(3).to_string())
print("\nSpearman correlation with temp_max (one feature at a time):")
print(feat[["elev", "slope", "n_north", "n_east", "lat"]].corrwith(feat.temp_max, method="spearman").round(3).to_string())


# ---- 2. Map orientation -----------------------------------------------------
# Standard south-pole view (seen from outside the Moon): 0 deg longitude at the top, 90E on the right.
# Image column = Moon-centred y, image row (upwards) = Moon-centred x.
# gj counts along +y, gi counts along -x, so: img[row, col] = A[N-1-gi, gj].
def grid(col):
    return feat[col].to_numpy().reshape(N, N, 2).mean(2)          # [gi, gj], mean of the 2 triangles


def img(a):
    return a[::-1, :]


EXT = [-300, 300, -300, 300]


def polar_xy(lat_deg, lon_deg, r=1737.4):
    rho = r * np.cos(np.radians(lat_deg))                         # distance from the polar axis (km)
    return rho * np.sin(np.radians(lon_deg)), rho * np.cos(np.radians(lon_deg))


def decorate(ax):
    for lon_deg, lab in [(0, "0°"), (90, "90°E"), (180, "180°"), (270, "90°W")]:
        x, y = polar_xy(-80.6, lon_deg)
        ax.plot([0, x], [0, y], color="w", lw=0.4, ls=":")
        ax.text(*polar_xy(-80.2, lon_deg), lab, color="w", fontsize=8, ha="center", va="center",
                bbox=dict(fc="k", ec="none", alpha=0.4, pad=1))
    ax.set_xlabel("km from pole"); ax.set_ylabel("km from pole")


# Approximate centres of well-known permanently shadowed craters (IAU gazetteer, rounded).
CRATERS = {"Shackleton": (-89.67, 129.8), "Haworth": (-87.45, -5.0), "Shoemaker": (-88.1, 44.9),
           "Faustini": (-87.2, 84.3), "de Gerlache": (-88.5, -87.1), "Sverdrup": (-88.3, -152.0),
           "Cabeus": (-85.3, -42.1)}
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
im = ax.imshow(img(grid("temp_max")), origin="lower", extent=EXT, cmap="inferno")
decorate(ax)
cx, cy = polar_xy(df.tri_clat.to_numpy(), df.tri_clon.to_numpy())
print("\nOrientation check: coldest temp_max within 5 km of each crater centre")
for name, (la, lo) in CRATERS.items():
    x, y = polar_xy(la, lo)
    near = (cx - x) ** 2 + (cy - y) ** 2 < 25
    print(f"  {name:12s} at ({x:6.1f}, {y:6.1f}) km: min {df.temp_max[near].min():5.1f} K, median {df.temp_max[near].median():5.1f} K")
    ax.plot(x, y, "c+", ms=8); ax.text(x + 3, y + 3, name, color="c", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8, label="annual max surface temperature (K)")
ax.set_title("Annual max temperature, standard south-pole view (0° up)")
fig.savefig(FIG / "s02_orientation_check.png", dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
for ax, col, cmap, lab, lim in [(axes[0], "slope", "magma", "slope (deg)", (0, 30)),
                                (axes[1], "n_north", "RdBu_r", "normal, north component (> 0 faces the equator)", (-0.3, 0.3))]:
    im = ax.imshow(img(grid(col)), origin="lower", extent=EXT, cmap=cmap, vmin=lim[0], vmax=lim[1])
    decorate(ax); fig.colorbar(im, ax=ax, shrink=0.8, label=lab)
axes[0].set_title("Slope"); axes[1].set_title("Which way the surface faces")
fig.savefig(FIG / "s02_features.png", dpi=130)

# ---- 3. Geographic split ------------------------------------------------------
# 12 x 12 blocks of 100 x 100 cells (~50 km). Blocks are assigned train / val / test,
# stratified by their cold-triangle share so all 3 splits contain shadowed craters.
# Training cells within BUF cells (~5 km) of a val or test block become buffer (no labels used).
B, BUF = 100, 10
nb = N // B
cold = (feat.temp_max < 110).to_numpy().reshape(N, N, 2).mean(2)
block_cold = cold.reshape(nb, B, nb, B).mean((1, 3)).ravel()
rng = np.random.default_rng(42)
block_split = np.zeros(nb * nb, int)
order = np.argsort(block_cold)
for chunk in np.array_split(order, len(order) // 7):        # in each group of ~7 similar blocks: 1 val, 1 test
    pick = rng.permutation(chunk)
    block_split[pick[0]], block_split[pick[1]] = 1, 2
cell_split = np.kron(block_split.reshape(nb, nb), np.ones((B, B), int))
held = cell_split > 0
dist_ok = np.ones_like(held)
for di in range(-BUF, BUF + 1):                               # square buffer: any held cell within BUF cells
    for dj in range(-BUF, BUF + 1):
        sh = np.roll(np.roll(held, di, 0), dj, 1)
        if di > 0: sh[:di] = False
        if di < 0: sh[di:] = False
        if dj > 0: sh[:, :dj] = False
        if dj < 0: sh[:, dj:] = False
        dist_ok &= ~sh | held
cell_split[(cell_split == 0) & ~dist_ok] = -1
feat["split"] = np.repeat(cell_split.ravel(), 2)            # triangle order = (gi, gj, k)

print("\nSplit summary (triangles):")
for s, name in [(0, "train"), (1, "val"), (2, "test"), (-1, "buffer")]:
    m = feat.split == s
    print(f"  {name:6s} {m.mean():6.1%}  n={m.sum():>9,}  mean temp_max {feat.temp_max[m].mean():6.1f} K  "
          f"cold (<110 K) {(feat.temp_max[m] < 110).mean():5.1%}")

fig, ax = plt.subplots(figsize=(7, 6.3), constrained_layout=True)
cmap = plt.matplotlib.colors.ListedColormap(["0.85", "#4C78A8", "#F58518", "#54A24B"])
ax.imshow(img(cell_split), origin="lower", extent=EXT, cmap=cmap, vmin=-1.5, vmax=2.5)
ax.contour(img(grid("temp_max")) < 110, levels=[0.5], origin="lower", extent=EXT, colors="k", linewidths=0.4)
decorate(ax)
ax.legend(handles=[plt.matplotlib.patches.Patch(color=cmap(i), label=l) for i, l in
                   enumerate(["buffer (~5 km, unused)", "train", "validation", "test"])],
          loc="lower left", fontsize=8, framealpha=0.9)
ax.set_title("Geographic split: 50 km blocks, black outlines = cold traps (<110 K)")
fig.savefig(FIG / "s02_split.png", dpi=130)

# ---- 4. Save ----------------------------------------------------------------
feat.to_parquet(ROOT / "data" / "features.parquet", index=False)
CH = ["elev", "slope", "n_north", "n_east"]                   # per-triangle channels (x2 triangles)
X = np.concatenate([feat[c].to_numpy().reshape(N, N, 2).transpose(2, 0, 1) for c in CH], 0)
X = np.concatenate([X] + [grid(c)[None] for c in ["lat", "sin_lon", "cos_lon"]], 0).astype(np.float32)
Y = feat.temp_max.to_numpy().reshape(N, N, 2).transpose(2, 0, 1).astype(np.float32)
np.savez_compressed(ROOT / "data" / "grid.npz", X=X, Y=Y, split=cell_split.astype(np.int8),
                    channels=np.array([f"{c}_k{k}" for c in CH for k in (0, 1)] + ["lat", "sin_lon", "cos_lon"]))
print(f"\nsaved features.parquet and grid.npz: X {X.shape}, Y {Y.shape}")
