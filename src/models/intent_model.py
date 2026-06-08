"""
EdgeTransAI — Intent Model (BiLSTM)
=====================================
Predicts driver routing intent (straight / left / right / U-turn) from a
short observation window of GPS trajectory points.  Deployed on-edge
(TensorRT INT8) with a forward-pass budget of < 2 ms.

Training
--------
    python -m src.models.intent_model --train --data data/trajectories.npy

Inference
---------
    model = IntentModel(config)
    model.load("outputs/checkpoints/intent_best.pt")
    probs = model.predict(trajectory_window)  # shape (4,)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Each sample: (seq, label)
      seq   : (L, 4)  — [x, y, vx, vy] over L timesteps
      label : int in {0,1,2,3}  — straight, left, right, u-turn
    """

    def __init__(self, data_path: Optional[str] = None,
                 n_synthetic: int = 20000, seq_len: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)

        if data_path and os.path.exists(data_path):
            loaded = np.load(data_path, allow_pickle=True).item()
            self.sequences = loaded["sequences"].astype(np.float32)
            self.labels    = loaded["labels"].astype(np.int64)
        else:
            print("[IntentModel] Generating synthetic trajectory dataset …")
            seqs, labels = [], []
            for cls in range(4):
                for _ in range(n_synthetic // 4):
                    s = self._synthetic_trajectory(cls, seq_len, rng)
                    seqs.append(s)
                    labels.append(cls)
            self.sequences = np.array(seqs, dtype=np.float32)
            self.labels    = np.array(labels, dtype=np.int64)

        # Normalise
        self.sequences = (self.sequences - self.sequences.mean()) / (self.sequences.std() + 1e-8)

    @staticmethod
    def _synthetic_trajectory(cls: int, seq_len: int, rng) -> np.ndarray:
        """Produce a plausible GPS sequence for a given intent class."""
        heading = {0: 0.0, 1: -np.pi/2, 2: np.pi/2, 3: np.pi}[cls]
        speed = rng.uniform(8, 14)      # m/s
        noise = 0.3
        traj = []
        x, y = 0.0, 0.0
        for t in range(seq_len):
            h = heading + rng.normal(0, 0.05)
            vx = speed * np.cos(h) + rng.normal(0, noise)
            vy = speed * np.sin(h) + rng.normal(0, noise)
            x += vx * 0.1
            y += vy * 0.1
            traj.append([x, y, vx, vy])
        return np.array(traj)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]),
                torch.tensor(self.labels[idx]))


# ---------------------------------------------------------------------------
# BiLSTM model
# ---------------------------------------------------------------------------

class IntentModel(nn.Module):
    """
    Two-layer Bidirectional LSTM encoder → 4-class softmax.

    Architecture
    ------------
    Input  : (B, L, 4)
    BiLSTM : (B, L, 2*hidden)  →  last hidden concat
    FC     : (B, 2*hidden) → (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["intent_model"]
        self.hidden_dim  = cfg["hidden_dim"]
        self.num_layers  = cfg["num_layers"]
        self.num_classes = cfg["num_classes"]
        self.seq_len     = cfg["sequence_length"]

        self.lstm = nn.LSTM(
            input_size=cfg["input_dim"],
            hidden_size=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(cfg["hidden_dim"] * 2, cfg["num_classes"])
        self._device = "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, input_dim)  →  logits : (B, num_classes)"""
        out, (hn, _) = self.lstm(x)
        # Concatenate forward and backward final hidden states
        fwd = hn[-2]       # last forward layer
        bwd = hn[-1]       # last backward layer
        h = torch.cat([fwd, bwd], dim=-1)
        return self.fc(self.dropout(h))

    @torch.no_grad()
    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        """
        trajectory : (L, 4) numpy array
        Returns probability distribution over 4 intent classes.
        """
        t = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0).to(self._device)
        logits = self(t)
        return F.softmax(logits, dim=-1).cpu().numpy().squeeze()

    def to_device(self, device: str):
        self._device = device
        return self.to(device)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[IntentModel] Saved → {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self._device))
        self.eval()
        print(f"[IntentModel] Loaded ← {path}")


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_intent_model(config: dict, data_path: Optional[str] = None,
                       epochs: int = 100, patience: int = 15,
                       output_dir: str = "outputs/checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TrajectoryDataset(data_path, seq_len=config["intent_model"]["sequence_length"])
    n_val = int(0.1 * len(dataset))
    n_test = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test])

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = IntentModel(config).to_device(device)
    opt   = torch.optim.Adam(model.parameters(), lr=config["intent_model"]["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_acc = 0.0
    wait = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seqs, labels in train_dl:
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total_samples = 0
        with torch.no_grad():
            for seqs, labels in val_dl:
                seqs, labels = seqs.to(device), labels.to(device)
                preds = model(seqs).argmax(dim=-1)
                correct       += (preds == labels).sum().item()
                total_samples += len(labels)
        val_acc = correct / total_samples
        history["train_loss"].append(total_loss / len(train_dl))
        history["val_acc"].append(val_acc)
        sched.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={total_loss/len(train_dl):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(os.path.join(output_dir, "intent_best.pt"))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[IntentModel] Early stopping at epoch {epoch}.")
                break

    print(f"[IntentModel] Training complete. Best val accuracy: {best_val_acc:.4f}")
    np.save(os.path.join(output_dir, "intent_history.npy"), history)
    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data",   default=None)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.train:
        train_intent_model(cfg, data_path=args.data, epochs=args.epochs)
