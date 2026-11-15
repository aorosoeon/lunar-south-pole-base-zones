"""Sprint 3a: latitude-only regression and an MLP on per-triangle features (no neighbours)."""
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from common import FEATS, ROOT, load_feat, report

torch.manual_seed(0)
f = load_feat()
y, split = f.temp_max.to_numpy(np.float32), f.split.to_numpy()
tr, va = split == 0, split == 1

# ---- Baseline 0: latitude only ----------------------------------------------
lat = f[["lat"]].to_numpy()
lin = LinearRegression().fit(lat[tr], y[tr])
p_lat = lin.predict(lat).astype(np.float32)
res = [report("latitude-only linear", y, p_lat, split)]

# ---- Baseline 1: MLP, one triangle at a time ------------------------------------
X = f[FEATS].to_numpy(np.float32)
mu, sd = X[tr].mean(0), X[tr].std(0)              # train-only standardisation
ym, ys = y[tr].mean(), y[tr].std()
Xt = torch.from_numpy((X - mu) / sd)
yt = torch.from_numpy((y - ym) / ys)

model = nn.Sequential(nn.Linear(len(FEATS), 128), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
tr_idx, va_idx = np.where(tr)[0], np.where(va)[0]


def predict(idx=None):
    model.eval()
    with torch.no_grad():
        xs = Xt if idx is None else Xt[idx]
        return torch.cat([model(b) for b in xs.split(65536)]).squeeze(1).numpy() * ys + ym


best, best_state, bad, t0 = np.inf, None, 0, time.time()
for epoch in range(30):
    model.train()
    perm = torch.from_numpy(np.random.default_rng(epoch).permutation(tr_idx))
    for b in perm.split(8192):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xt[b]).squeeze(1), yt[b])
        loss.backward()
        opt.step()
    val_mae = np.abs(predict(va_idx) - y[va_idx]).mean()
    print(f"epoch {epoch:2d}  train mse(std) {loss.item():.4f}  val MAE {val_mae:6.2f} K  ({time.time() - t0:.0f}s)", flush=True)
    if val_mae < best - 0.01:
        best, best_state, bad = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 4:
            break
model.load_state_dict(best_state)
p_mlp = predict().astype(np.float32)
res.append(report("MLP (per triangle)", y, p_mlp, split))
torch.save(model.state_dict(), ROOT / "data" / "s03_mlp.pt")
np.savez_compressed(ROOT / "data" / "s03_preds_mlp.npz", lat=p_lat, mlp=p_mlp)

import pandas as pd  # noqa: E402
out = pd.concat(res)
out.to_csv(ROOT / "data" / "s03_scores_mlp.csv", index=False)
print("\n" + out.round(2).to_string(index=False))
