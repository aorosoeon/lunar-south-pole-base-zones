"""Sprint 3b figure: truth vs MLP vs augmented CNN on held-out blocks near the pole."""
import matplotlib.pyplot as plt
import numpy as np

from common import N, ROOT, load_feat

f = load_feat()
y = f.temp_max.to_numpy()
p_mlp = np.load(ROOT / "data" / "s03_preds_mlp.npz")["mlp"]
p_cnn = np.load(ROOT / "data" / "s03b_preds_cnn_aug.npz")["cnn_aug_tta"]
split = f.split.to_numpy()


def img(v):  # triangle vector -> standard south-pole view image (0 deg up)
    return v.reshape(N, N, 2).mean(2)[::-1, :]


EXT = [-300, 300, -300, 300]
held = img((split > 0).astype(float)) > 0.5
panels = [("Diviner reference (target)", img(y), "inferno", (30, 330)),
          ("MLP, one triangle at a time", img(p_mlp), "inferno", (30, 330)),
          ("CNN + augmentation (test-time avg.)", img(p_cnn), "inferno", (30, 330)),
          ("CNN absolute error", img(np.abs(p_cnn - y)), "viridis", (0, 60))]
fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
for ax, (title, a, cmap, lim) in zip(axes, panels):
    im = ax.imshow(a, origin="lower", extent=EXT, cmap=cmap, vmin=lim[0], vmax=lim[1])
    ax.contour(held, levels=[0.5], origin="lower", extent=EXT, colors="c", linewidths=1.2)
    ax.set_xlim(-100, 100); ax.set_ylim(-60, 130); ax.set_title(title, fontsize=11)
    ax.set_xlabel("km from pole")
    fig.colorbar(im, ax=ax, shrink=0.75, label="K")
axes[0].set_ylabel("km from pole")
for name, (x, yy) in {"Shoemaker (test)": (40.7, 40.8), "Haworth (val)": (-6.7, 77.0), "Shackleton": (7.7, -6.4)}.items():
    axes[0].annotate(name, (x, yy), (x + 12, yy + 14), color="c", fontsize=8, arrowprops=dict(arrowstyle="-", color="c"))
fig.suptitle("Held-out blocks (cyan outline) near the south pole: the model never saw these temperatures", fontsize=12)
fig.savefig(ROOT / "figures" / "s03b_compare.png", dpi=120)
