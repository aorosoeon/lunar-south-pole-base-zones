"""Sprint 5: slide-sized figures (bigger fonts, fewer panels)."""
import matplotlib.pyplot as plt
import numpy as np

from common import N, ROOT, load_feat

plt.rcParams.update({"font.size": 13})
f = load_feat()
y = f.temp_max.to_numpy()
p = np.load(ROOT / "data" / "s03b_preds_cnn_aug.npz")["cnn_aug_tta"]
held = (f.split.to_numpy().reshape(N, N, 2)[:, :, 0] > 0)[::-1, :]
img = lambda v: v.reshape(N, N, 2).mean(2)[::-1, :]  # noqa: E731
EXT = [-300, 300, -300, 300]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
for ax, a, t in [(axes[0], img(y), "Diviner reference"), (axes[1], img(p), "CNN prediction from terrain only")]:
    im = ax.imshow(a, origin="lower", extent=EXT, cmap="inferno", vmin=30, vmax=330)
    ax.contour(held, levels=[0.5], origin="lower", extent=EXT, colors="c", linewidths=1.6)
    ax.set_xlim(-100, 100); ax.set_ylim(-60, 130); ax.set_title(t, fontsize=15, pad=8); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01); cb.set_label("annual max temperature (K)")
axes[0].plot([-90, -70], [-50, -50], "w", lw=3); axes[0].text(-80, -46, "20 km", color="w", ha="center", fontsize=11)
fig.savefig(ROOT / "figures" / "s05_slide_results_map.png", dpi=150)
print("ok")

# Title-slide image: the full temperature map, no axes
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img(y), origin="lower", extent=EXT, cmap="inferno", vmin=30, vmax=330)
ax.set_axis_off()
fig.savefig(ROOT / "figures" / "s05_hero_map.png", dpi=200)
print("hero ok")
