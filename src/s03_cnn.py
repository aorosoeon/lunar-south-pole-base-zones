"""Sprint 3b: U-Net on the 1200 x 1200 cell grid (2 output channels = the 2 triangles in each cell).

Inputs are read everywhere (terrain is known everywhere); the loss uses train-cell labels only.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import ROOT, grid_to_tri, load_feat, report

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() and "--cpu" not in sys.argv else "cpu"
STEPS, CROP, BATCH = 2000, 256, 8

g = np.load(ROOT / "data" / "grid.npz")
X, Y, S = g["X"], g["Y"], g["split"]
trm = S == 0
mu = X[:, trm].mean(1)[:, None, None]
sd = X[:, trm].std(1)[:, None, None]
ym, ys = Y[:, trm].mean(), Y[:, trm].std()
Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
Yn = torch.from_numpy(((Y - ym) / ys).astype(np.float32))
M = torch.from_numpy(trm)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, cout, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.down = nn.ModuleList([block(a, b) for a, b in zip((cin,) + ch[:-1], ch)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in range(len(ch) - 1)])
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


model = UNet(Xn.shape[0], 2)
print(f"device {DEV}, parameters {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ---- GPU vs CPU check on identical weights (see tensorflow-metal ReLU bug) ----
if DEV == "mps":
    xb = Xn[None, :, :CROP, :CROP]
    for mode in ("eval", "train"):
        torch.manual_seed(1)
        m_cpu = UNet(Xn.shape[0], 2); m_gpu = UNet(Xn.shape[0], 2); m_gpu.load_state_dict(m_cpu.state_dict()); m_gpu.to(DEV)
        getattr(m_cpu, mode)(); getattr(m_gpu, mode)()
        o_cpu, o_gpu = m_cpu(xb), m_gpu(xb.to(DEV))
        o_cpu.square().mean().backward(); o_gpu.square().mean().backward()
        g_cpu, g_gpu = m_cpu.down[0][0].weight.grad, m_gpu.down[0][0].weight.grad.cpu()
        print(f"  {mode}: max |out diff| {(o_cpu - o_gpu.cpu()).abs().max().item():.2e} "
              f"(out scale {o_cpu.abs().max().item():.2f}), grad rel diff "
              f"{((g_cpu - g_gpu).norm() / g_cpu.norm()).item():.2e}", flush=True)

model.to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
f = load_feat()
y, split = f.temp_max.to_numpy(np.float32), f.split.to_numpy()
va_t = split == 1


def predict_full():
    model.eval()
    with torch.no_grad():
        out = model(Xn[None].to(DEV))[0].cpu().numpy()
    return grid_to_tri(out * ys + ym).astype(np.float32)


rng = np.random.default_rng(0)
best, best_state, t0, run = np.inf, None, time.time(), 0.0
for step in range(1, STEPS + 1):
    model.train()
    xs, ts, ms = [], [], []
    while len(xs) < BATCH:
        i, j = rng.integers(0, 1200 - CROP + 1, 2)
        m = M[i:i + CROP, j:j + CROP]
        if m.float().mean() < 0.3:          # skip crops that are mostly held-out area
            continue
        xs.append(Xn[:, i:i + CROP, j:j + CROP]); ts.append(Yn[:, i:i + CROP, j:j + CROP]); ms.append(m)
    xb, tb, mb = torch.stack(xs).to(DEV), torch.stack(ts).to(DEV), torch.stack(ms)[:, None].expand(-1, 2, -1, -1).to(DEV)
    opt.zero_grad()
    loss = ((model(xb) - tb) ** 2)[mb].mean()
    loss.backward()
    opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 200 == 0:
        p = predict_full()
        val_mae = np.abs(p[va_t] - y[va_t]).mean()
        print(f"step {step:5d}  train mse(std) {run:.4f}  val MAE {val_mae:6.2f} K  ({time.time() - t0:.0f}s)", flush=True)
        if val_mae < best:
            best, best_state = val_mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
p_cnn = predict_full()
torch.save(best_state, ROOT / "data" / "s03_cnn.pt")
np.savez_compressed(ROOT / "data" / "s03_preds_cnn.npz", cnn=p_cnn)
out = report("CNN (U-Net, grid)", y, p_cnn, split)
out.to_csv(ROOT / "data" / "s03_scores_cnn.csv", index=False)
print("\n" + out.round(2).to_string(index=False))
