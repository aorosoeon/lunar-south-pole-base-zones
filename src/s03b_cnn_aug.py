"""Sprint 3b: U-Net with physics-consistent augmentation and no absolute longitude.

Changes vs s03_cnn.py:
- sin/cos longitude replaced by a 'direction to the pole' vector (2 channels, grid frame).
- Each training crop gets a random 90-degree rotation and/or mirror flip (8 variants).
  The pole-direction vector is turned with the crop and n_east changes sign under a mirror flip,
  so every variant is a valid piece of terrain at another longitude. Annual max temperature
  does not depend on longitude (the sun goes all the way round), so the label stays valid.
- GroupNorm instead of BatchNorm, dropout in the deep layers, validation every 100 steps.
"""
import time

import numpy as np
import torch
from torch import nn

from common import ROOT, grid_to_tri, load_feat, report, score

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
STEPS, CROP, BATCH = 3000, 256, 8

g = np.load(ROOT / "data" / "grid.npz")
X, Y, S = g["X"].copy(), g["Y"], g["split"]
# channels: elev_k0/1, slope_k0/1, n_north_k0/1, n_east_k0/1, lat, sin_lon, cos_lon
cos_l, sin_l = X[10], X[9]
norm = np.hypot(cos_l, sin_l)
X[9], X[10] = cos_l / norm, -sin_l / norm      # unit vector to the pole: (along gi, along gj)
PI, PJ, NE = 9, 10, [6, 7]
trm = S == 0
mu = X[:, trm].mean(1)
sd = X[:, trm].std(1)
mu[[PI, PJ] + NE] = 0                          # vector-like channels: scale only, never shift
sd[[PI, PJ]] = 1
ym, ys = Y[:, trm].mean(), Y[:, trm].std()
Xn = torch.from_numpy(((X - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
Yn = torch.from_numpy(((Y - ym) / ys).astype(np.float32))
M = torch.from_numpy(trm)


def spatial(t, k, flip):
    """Mirror (optional), then rotate k x 90 degrees, over the last 2 dims."""
    if flip:
        t = torch.flip(t, dims=(-1,))
    return torch.rot90(t, k, dims=(-2, -1))


def unspatial(t, k, flip):
    t = torch.rot90(t, -k, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if flip else t


def augment_x(x, k, flip):
    """Input tensor (C, H, W): move pixels, then turn the vector channels the same way."""
    x = spatial(x, k, flip).clone()
    vi, vj = x[PI].clone(), x[PJ].clone()
    if flip:
        vj = -vj
        x[NE] = -x[NE]
    for _ in range(k % 4):                     # rot90 maps components (vi, vj) -> (-vj, vi)
        vi, vj = -vj, vi
    x[PI], x[PJ] = vi, vj
    return x


# Self-test: a field pointing to a pole must still point to the (moved) pole after every transform.
H = 9
ii, jj = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing="ij")
pole = torch.tensor([2.3, 6.1])
for k in range(4):
    for flip in (False, True):
        x = torch.zeros(11, H, H)
        d = torch.stack([pole[0] - ii, pole[1] - jj]); d /= d.norm(dim=0)
        x[PI], x[PJ] = d
        marker = torch.zeros(1, H, H); marker[0, 2, 6] = 1          # cell nearest the pole
        xa, ma = augment_x(x, k, flip), spatial(marker, k, flip)
        pi_new, pj_new = [int(v) for v in torch.nonzero(ma[0])[0]]
        ok = ((xa[PI] * (pi_new - ii) + xa[PJ] * (pj_new - jj)) > 0).float().mean().item()
        assert ok > 0.95, (k, flip, ok)
print("augmentation self-test passed: pole direction stays consistent under all 8 transforms", flush=True)


def block(cin, cout, drop=0.0):
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.Dropout2d(drop)) if drop else nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, cin, cout, ch=(32, 64, 128, 128, 256), drop=(0, 0, 0.1, 0.1, 0.2)):
        super().__init__()
        self.down = nn.ModuleList([block(a, b, p) for a, b, p in zip((cin,) + ch[:-1], ch, drop)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i], drop[i]) for i in range(len(ch) - 1)])
        self.head = nn.Conv2d(ch[0], cout, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else self.pool(x))
            skips.append(x)
        for i in reversed(range(len(self.up))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


model = UNet(Xn.shape[0], 2).to(DEV)
print(f"device {DEV}, parameters {sum(p.numel() for p in model.parameters()):,}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
f = load_feat()
y, split = f.temp_max.to_numpy(np.float32), f.split.to_numpy()
va_t = split == 1


def predict_full(tta=False):
    model.eval()
    outs = []
    with torch.no_grad():
        for k, flip in ([(k, fl) for k in range(4) for fl in (False, True)] if tta else [(0, False)]):
            o = model(augment_x(Xn, k, flip)[None].to(DEV))[0].cpu()
            outs.append(unspatial(o, k, flip))
    out = torch.stack(outs).mean(0).numpy()
    return grid_to_tri(out * ys + ym).astype(np.float32)


rng = np.random.default_rng(0)
best, best_state, best_step, t0, run = np.inf, None, 0, time.time(), 0.0
for step in range(1, STEPS + 1):
    model.train()
    xs, ts, ms = [], [], []
    while len(xs) < BATCH:
        i, j = rng.integers(0, 1200 - CROP + 1, 2)
        m = M[i:i + CROP, j:j + CROP]
        if m.float().mean() < 0.3:
            continue
        k, flip = int(rng.integers(4)), bool(rng.integers(2))
        xs.append(augment_x(Xn[:, i:i + CROP, j:j + CROP], k, flip))
        ts.append(spatial(Yn[:, i:i + CROP, j:j + CROP], k, flip))
        ms.append(spatial(m, k, flip))
    xb, tb = torch.stack(xs).to(DEV), torch.stack(ts).to(DEV)
    mb = torch.stack(ms)[:, None].expand(-1, 2, -1, -1).to(DEV)
    opt.zero_grad()
    loss = ((model(xb) - tb) ** 2)[mb].mean()
    loss.backward()
    opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 100 == 0:
        p = predict_full()
        r = score(y[va_t], p[va_t])
        print(f"step {step:5d}  train mse(std) {run:.4f}  val MAE {r['MAE']:6.2f} K  val cold MAE {r['MAE_cold']:6.1f} K  ({time.time() - t0:.0f}s)", flush=True)
        if r["MAE"] < best:
            best, best_step, best_state = r["MAE"], step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print(f"best val MAE {best:.2f} K at step {best_step}", flush=True)
model.load_state_dict(best_state)
torch.save(best_state, ROOT / "data" / "s03b_cnn_aug.pt")
p1, p8 = predict_full(), predict_full(tta=True)
np.savez_compressed(ROOT / "data" / "s03b_preds_cnn_aug.npz", cnn_aug=p1, cnn_aug_tta=p8)
import pandas as pd  # noqa: E402
out = pd.concat([report("CNN + augmentation", y, p1, split), report("CNN + augmentation + 8-way TTA", y, p8, split)])
out.to_csv(ROOT / "data" / "s03b_scores.csv", index=False)
print("\n" + out.round(2).to_string(index=False))
trc = (split == 0) & (y < 110)
print(f"\ncold MAE on the TRAIN area: {np.abs(p1[trc] - y[trc]).mean():.1f} K (was 31.6 K for the first CNN)")
