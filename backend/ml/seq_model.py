"""
backend/ml/seq_model.py
=======================
The SEQUENCE model: a PyTorch LSTM that reads a learner's review history, in order.

  history  --LSTM-->  h  --small network-->  (strength, scale, shape)
  recall(gap) = strength * exp( -(gap / scale) ** shape )

  * strength in (0,1): how well the concept is known right after the last review
  * scale (hours):     how long the memory lasts
  * shape (>0.5):      how sharply it fades (shape 1 = plain exponential forgetting)

Because recall is a Weibull-shaped curve of the gap, it can never rise as time passes and it
keeps fading beyond any gap seen in training - the same guarantees as the survival model - while the
LSTM is free to pick up patterns in the ORDER of reviews (improving or worsening streaks, spacing).

Trained with binary cross-entropy against "did the next review go well?".
PyTorch is optional: if it is not installed, torch_available() is False and the app simply
skips this model (train.py reports it as 'not available').
"""
import numpy as np

from backend.ml.sequences import STEP_DIM, MAX_LEN

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:                      # not installed (or broken install)
    TORCH_OK = False


def torch_available() -> bool:
    return TORCH_OK


def pad_batch(seqs, idx):
    """-> (x float32 (B, T, STEP_DIM), lengths int64 (B,)) for the sequences at positions `idx`."""
    chosen = [np.asarray(seqs[i], dtype="float32")[-MAX_LEN:] for i in idx]
    lengths = np.array([len(s) for s in chosen], dtype="int64")
    x = np.zeros((len(chosen), int(lengths.max()), STEP_DIM), dtype="float32")
    for row, s in enumerate(chosen):
        x[row, : len(s)] = s
    return x, lengths


def weibull_recall(strength, scale, shape, gap):
    """NumPy version of the curve (used to document and test the maths)."""
    z = (np.maximum(np.asarray(gap, dtype=float), 1e-3) / scale) ** shape
    return strength * np.exp(-np.minimum(z, 50.0))


if TORCH_OK:
    class RecallNet(nn.Module):
        def __init__(self, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=STEP_DIM, hidden_size=hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 3))
            with torch.no_grad():                                  # start at: strength 0.88, scale ~3 days, shape ~1.5
                self.head[2].bias.copy_(torch.tensor([2.0, 4.3, 0.5]))

        def curve_params(self, x, lengths):
            out, _ = self.lstm(x)                                  # (B, T, H)
            last = out[torch.arange(out.size(0)), (lengths - 1).clamp(min=0)]   # output at each sequence's last real step
            a, b, c = self.head(last).unbind(dim=-1)
            return torch.sigmoid(a), torch.exp(b.clamp(-3.0, 9.0)), 0.5 + F.softplus(c)

        def forward(self, x, lengths, gap):
            strength, scale, shape = self.curve_params(x, lengths)
            z = (gap.clamp(min=1e-3) / scale) ** shape
            return strength * torch.exp(-z.clamp(max=50.0))


class SequenceRecallModel:
    """fit / predict / save / load around RecallNet."""

    def __init__(self, hidden: int = 32):
        if not TORCH_OK:
            raise ImportError("PyTorch is not installed (pip install torch)")
        self.hidden = hidden
        self.net = RecallNet(hidden)

    def fit(self, seqs, gaps, labels, val=None, epochs: int = 60, lr: float = 3e-3,
            batch_size: int = 128, patience: int = 8, seed: int = 0):
        """seqs: list of (n_i, 4) arrays; gaps: hours to the next review; labels: 0/1.
        val = (seqs, gaps, labels) for early stopping (recommended)."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        gaps = np.asarray(gaps, dtype="float32")
        labels = np.asarray(labels, dtype="float32")
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        best_loss, best_state, bad = float("inf"), None, 0
        self.history = []
        for epoch in range(epochs):
            self.net.train()
            order = rng.permutation(len(seqs))
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                x, ln = pad_batch(seqs, idx)
                p = self.net(torch.from_numpy(x), torch.from_numpy(ln), torch.from_numpy(gaps[idx]))
                loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), torch.from_numpy(labels[idx]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            monitor = self._loss(*val) if val is not None else self._loss(seqs, gaps, labels)
            self.history.append(monitor)
            if monitor < best_loss - 1e-4:
                best_loss, bad = monitor, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _loss(self, seqs, gaps, labels) -> float:
        p = np.clip(self.predict(seqs, gaps), 1e-6, 1 - 1e-6)
        y = np.asarray(labels, dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict(self, seqs, gaps, batch_size: int = 512) -> np.ndarray:
        """Chance that the NEXT review (after `gaps` hours) goes well, one value per sequence."""
        self.net.eval()
        gaps = np.asarray(gaps, dtype="float32")
        out = []
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                idx = list(range(start, min(start + batch_size, len(seqs))))
                x, ln = pad_batch(seqs, idx)
                p = self.net(torch.from_numpy(x), torch.from_numpy(ln), torch.from_numpy(gaps[idx]))
                out.append(p.numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def curve(self, seq):
        """(strength, scale_hours, shape) the model predicts for one history - for explanations."""
        self.net.eval()
        x, ln = pad_batch([seq], [0])
        with torch.no_grad():
            s, sc, sh = self.net.curve_params(torch.from_numpy(x), torch.from_numpy(ln))
        return float(s[0]), float(sc[0]), float(sh[0])

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        torch.save({"state": self.net.state_dict(), "hidden": self.hidden}, str(path))

    @classmethod
    def load(cls, path):
        ckpt = torch.load(str(path), map_location="cpu")
        m = cls(int(ckpt["hidden"]))
        m.net.load_state_dict(ckpt["state"])
        m.net.eval()
        return m
