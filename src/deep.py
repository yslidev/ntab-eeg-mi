"""EEGNet-v4 in PyTorch, plus a subject-grouped training loop.

Deliberately small: ~2k parameters. With ~4,000 training epochs of 64x481
samples, anything larger memorises the training subjects immediately -- which
is itself one of the results we report.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Clamp(nn.Module):
    def forward(self, x): return torch.clamp(x, 1e-7, 1e4)


class EEGNet(nn.Module):
    def __init__(self, n_chan=64, n_times=481, n_classes=2, F1=8, D=2, F2=16,
                 kern=64, drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),   # depthwise, spatial
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),                        # separable
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n = self.block2(self.block1(torch.zeros(1, 1, n_chan, n_times))).numel()
        self.head = nn.Linear(n, n_classes)

    def forward(self, x):
        x = self.block2(self.block1(x.unsqueeze(1)))
        return self.head(x.flatten(1))


def _standardise(X):
    """Per-trial scaling by a SINGLE scalar, not per channel.

    The first version of this z-scored every channel of every trial
    independently, which normalises away exactly the thing the task lives in:
    left-versus-right motor imagery is a *relative* power difference between
    channels over the two hemispheres. Per-channel normalisation forces every
    channel to unit variance and deletes it. The network then trained to 53.5%
    on its own training set. Scaling each trial by one number preserves the
    across-channel pattern while still removing the between-subject amplitude
    differences a CNN would otherwise have to absorb."""
    mu = X.mean(-1, keepdims=True)
    X = X - mu
    sd = X.std(axis=(1, 2), keepdims=True) + 1e-9
    return X / sd


def train_eval(Xtr, ytr, Xte, yte, Xva=None, yva=None, epochs=100, bs=64,
               lr=3e-3, wd=1e-4, device=None, seed=0, verbose=False, kern=32):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr, Xte = _standardise(Xtr), _standardise(Xte)
    tr = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    te = torch.tensor(Xte, dtype=torch.float32).to(device)
    net = EEGNet(Xtr.shape[1], Xtr.shape[2], kern=kern, drop=0.4).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, int(np.ceil(len(ytr) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps, pct_start=0.25)
    if Xva is not None:
        va = torch.tensor(_standardise(Xva), dtype=torch.float32).to(device)
        yv = torch.tensor(yva, dtype=torch.long).to(device)
    best, best_state, hist = -1.0, None, []
    n = len(yt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = tr[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        # Only consider checkpoints once the one-cycle schedule has annealed.
        # Selecting on a noisy validation signal from the very first epochs
        # restored an essentially untrained network, which is what made the
        # first cross-subject run report 50.7%.
        warm = int(0.6 * epochs)
        if Xva is not None and (ep + 1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                acc = (net(va).argmax(1) == yv).float().mean().item()
                tracc = (net(tr[:1500].to(device)).argmax(1).cpu()
                         == yt[:1500]).float().mean().item()
            hist.append(dict(epoch=ep + 1, loss=tot / n, val_acc=acc, train_acc=tracc))
            if acc > best and ep >= warm:
                best = acc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if verbose:
                print(f"  ep{ep+1:3d} loss {tot/n:.3f} train {tracc:.3f} val {acc:.3f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(te).argmax(1).cpu().numpy()
        trpred = net(tr[:2000].to(device)).argmax(1).cpu().numpy()
    return pred, dict(train_acc=float((trpred == ytr[:len(trpred)]).mean()),
                      hist=hist, net=net)
