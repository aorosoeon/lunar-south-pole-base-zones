"""Sprint 1: parse the Diviner PRP south-pole table, check the columns, check the mesh layout, draw first maps.

Input:  data/dlre_prp_south.tab (fixed-width ASCII, 210-byte rows, 1 header row + 2,880,000 rows)
Output: data/south.parquet, figures/s01_*.png, printed checks
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "data" / "dlre_prp_south.tab"
OUT = ROOT / "data" / "south.parquet"
FIG = ROOT / "figures"
R_MOON = 1737.4  # km, reference sphere from the label
N_GRID = 1200    # cells per side (2 triangles per cell)

t0 = time.time()
df = pd.read_csv(TAB, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
print(f"parsed {len(df):,} rows x {df.shape[1]} cols in {time.time() - t0:.1f}s")
assert len(df) == 2 * N_GRID * N_GRID, "row count does not match a 1200 x 1200 x 2 mesh"

# ---- 1. Column checks -------------------------------------------------------
print("\nNaN per column:", int(df.isna().sum().sum()), "in total")
print(df.describe(percentiles=[0.01, 0.5, 0.99]).T[["min", "1%", "50%", "99%", "max"]].round(3).to_string())
n_ice_invalid = int((df.ice_depth == -999).sum())
print(f"\nice_depth = -999 (deeper than 2.87 m): {n_ice_invalid:,} rows ({n_ice_invalid / len(df):.1%})")

# ---- 2. Mesh layout ---------------------------------------------------------
# Row order seen in the first rows: 2 triangles per cell, cells step along y first, then along x.
row = np.arange(len(df))
df["gi"] = row // (2 * N_GRID)          # x-direction index (x falls as gi rises)
df["gj"] = (row // 2) % N_GRID          # y-direction index (y rises as gj rises)
df["tri_k"] = row % 2                   # which of the 2 triangles in the cell

V = df[[f"tri{v}_{a}" for v in (1, 2, 3) for a in "xyz"]].to_numpy().reshape(-1, 3, 3)  # (tri, vertex, xyz)
cx, cy = V[:, :, 0].mean(1), V[:, :, 1].mean(1)
cell_x = cx.reshape(N_GRID, N_GRID, 2).mean(2)
cell_y = cy.reshape(N_GRID, N_GRID, 2).mean(2)
print("\nlayout check:")
print("  x falls along gi in every column:", bool((np.diff(cell_x, axis=0) < 0).all()))
print("  y rises along gj in every row:   ", bool((np.diff(cell_y, axis=1) > 0).all()))

# Shared vertices: the 2 triangles in a cell share 2 vertices; neighbour cells share 2 corners.
def n_shared(a, b):
    return np.array([len(set(map(tuple, x)) & set(map(tuple, y))) for x, y in zip(a, b)])


rng = np.random.default_rng(0)
cells = rng.integers(0, N_GRID - 1, size=(2000, 2))
t_a = 2 * (cells[:, 0] * N_GRID + cells[:, 1])
print("  same cell, triangle 0 vs 1 share:", np.bincount(n_shared(V[t_a], V[t_a + 1])).nonzero()[0], "vertices")
corners = lambda t: np.concatenate([V[t], V[t + 1]], axis=1)  # noqa: E731
print("  cell vs next cell along y share:", np.unique(n_shared(corners(t_a), corners(t_a + 2))), "unique vertices (expect 2)")
print("  cell vs next cell along x share:", np.unique(n_shared(corners(t_a), corners(t_a + 2 * N_GRID))), "unique vertices (expect 2)")

r_centroid = np.linalg.norm(V.mean(1), axis=1)
print(f"  |centroid| - {R_MOON} vs tri_calt: max abs diff {np.abs(r_centroid - R_MOON - df.tri_calt).max():.4f} km")
edge = np.linalg.norm(V[:, 1] - V[:, 0], axis=1)
print(f"  edge length v1-v2: median {np.median(edge):.3f} km, range {edge.min():.3f}-{edge.max():.3f} km")

df.to_parquet(OUT, index=False)
print(f"\nwrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.0f} MB)")

# ---- 3. First maps ----------------------------------------------------------
def cell_mean(col):
    return df[col].to_numpy().reshape(N_GRID, N_GRID, 2).mean(2)


def to_img(a):
    # rows = gj (y up with origin='lower'), columns = x rising left to right
    return a.T[:, ::-1]


ext = [-300, 300, -300, 300]
lat = to_img(cell_mean("tri_clat"))
fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
for ax, col, cmap, lab in [(axes[0], "temp_max", "inferno", "annual max surface temperature (K)"),
                           (axes[1], "tri_calt", "terrain", "elevation vs 1737.4 km sphere (km)")]:
    im = ax.imshow(to_img(cell_mean(col)), origin="lower", extent=ext, cmap=cmap)
    cs = ax.contour(lat, levels=[-85, -80], origin="lower", extent=ext, colors="w", linewidths=0.6)
    ax.clabel(cs, fmt=lambda v: f"{abs(v):.0f}°S", fontsize=8)
    ax.set_xlabel("x from pole (km)"); ax.set_ylabel("y from pole (km)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=lab)
axes[0].set_title("Diviner PRP: annual max temperature, south pole")
axes[1].set_title("Terrain elevation (same mesh)")
fig.savefig(FIG / "s01_maps.png", dpi=130)

fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.hist(df.temp_max, bins=200, color="0.3")
ax.set_xlabel("temp_max (K)"); ax.set_ylabel("triangles"); ax.set_title("Target distribution (2.88M triangles)")
fig.savefig(FIG / "s01_temp_max_hist.png", dpi=130)
print(f"done in {time.time() - t0:.0f}s")
