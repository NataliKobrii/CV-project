"""The RQ2 classifiers: a pose-feature MLP against an appearance-only CNN.

Both share the same training and evaluation loop, so the comparison is fair:
the same splits, the same optimizer settings, and the same metrics (macro
F1, per-class F1 and the confusion matrix).
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(str(Path(__file__).resolve().parent))
from config import FIGURES_DIR  # noqa: E402
from features import FEATURE_DIM  # noqa: E402


def default_device():
    """Return the best available torch device: MPS, then CUDA, then the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PoseMLP(nn.Module):
    """A small multilayer perceptron that classifies pose feature vectors."""
    def __init__(self, n_classes, in_dim=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # Two hidden layers are enough for a 47-dimensional input. Dropout
            # guards against overfitting on the limited number of windows.
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        """Compute the class scores for a batch of feature vectors."""
        return self.net(x)


class AppearanceCNN(nn.Module):
    """A small CNN on 64x64 RGB person crops."""

    def __init__(self, n_classes):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(), nn.MaxPool2d(2))
        # Three convolutional blocks, each halving the resolution and doubling
        # the channels, turn the 64x64 crop into a compact feature map.
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        # Global average pooling makes the classifier independent of the exact
        # spatial size before the final linear layer.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64, n_classes))

    def forward(self, x):
        """Compute the class scores for a batch of person crops."""
        return self.head(self.features(x))


def train_classifier(model, X_train, y_train, X_val, y_val, *,
                     epochs=40, batch_size=64, lr=1e-3, device=None,
                     class_weights=None, patience=8, verbose=True):
    """The shared training loop. X holds float32 arrays and y holds integer
    label indices.

    The best model by validation macro F1 is kept. Returns (model, history).
    """
    device = device or default_device()
    model = model.to(device)
    X_train = torch.as_tensor(X_train, dtype=torch.float32).contiguous()
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32).contiguous().to(device)
    weights = (torch.as_tensor(class_weights, dtype=torch.float32).to(device)
               if class_weights is not None else None)
    # The class weights compensate for the imbalanced state distribution.
    criterion = nn.CrossEntropyLoss(weight=weights)
    # AdamW with weight decay is a standard, robust choice for small models.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X_train)
    best_f1, best_state, since_best = -1.0, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        model.eval()
        with torch.no_grad():
            pred = model(X_val_t).argmax(1).cpu().numpy()
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "loss": total / n, "val_macro_f1": f1})
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} loss {total/n:.4f} validation macro-F1 {f1:.3f}")
        # Keep the weights of the best validation epoch (early stopping).
        if f1 > best_f1:
            best_f1, since_best = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
                # Stop once the validation score has not improved for several
            # epochs, which avoids wasting time and overfitting.
        if since_best >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X, device=None, batch_size=256):
    """Return the predicted class indices for X, evaluated in batches."""
    device = device or default_device()
    model = model.to(device).eval()
    X = torch.as_tensor(X, dtype=torch.float32).contiguous()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(X[i:i + batch_size].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, int)


def evaluate(y_true, y_pred, class_names, tag, save_fig=True):
    """Print the classification report, save the confusion-matrix figure and
    return a metrics dictionary."""
    rep = classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0))
    if save_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion – {tag}")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        out = FIGURES_DIR / f"confusion_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("Saved", out)
    return {"macro_f1": rep["macro avg"]["f1-score"],
            "accuracy": rep["accuracy"],
            "per_class_f1": {c: rep[c]["f1-score"] for c in class_names}}
