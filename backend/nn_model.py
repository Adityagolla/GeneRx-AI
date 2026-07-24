"""
nn_model.py — Small feed-forward neural network for drug risk classification.

Provides a sklearn-style wrapper (fit / predict / predict_proba) around a
PyTorch MLP so it drops into the same call sites as the XGBoost model in
ml_model.py without any changes to the prediction code.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class DrugRiskNet(nn.Module):
        """Feed-forward classifier over the 11 tabular risk features."""

        def __init__(self, input_dim, num_classes=4, hidden=(64, 32)):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden:
                layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.2)]
                prev_dim = h
            layers.append(nn.Linear(prev_dim, num_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


class TorchMLPClassifier:
    """
    sklearn-style wrapper around DrugRiskNet.

    Standardizes features internally (mean_/std_ computed at fit time) since
    raw tabular features here span very different scales (age ~0-100 vs.
    mean_se_frequency ~0-100 vs. is_primary_suspect 0/1), which matters for
    gradient-based training but is irrelevant for the tree-based XGBoost model.
    """

    def __init__(self, input_dim, num_classes=4, hidden=(64, 32),
                 epochs=80, lr=1e-3, weight_decay=1e-4, class_weight=None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to train/use TorchMLPClassifier")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.mean_ = None
        self.std_ = None
        self.model = DrugRiskNet(input_dim, num_classes, hidden)

    def _scale(self, X):
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        # Train on GPU if one's available — this model is tiny (a few
        # thousand params) so the win is modest, but free when present.
        # The model is always moved back to CPU before returning: the
        # pickled artifact has to load on whatever machine serves it
        # (e.g. a GPU-less Hugging Face Space), and CPU inference on a
        # model this size is sub-millisecond anyway, so there's no reason
        # to make serving depend on CUDA being available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"[nn_model] Training on GPU: {torch.cuda.get_device_name(0)}")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        X_t = torch.tensor(self._scale(X), dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        weight = None
        if self.class_weight is not None:
            weight = torch.tensor(self.class_weight, dtype=torch.float32, device=device)

        self.model.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X_t), y_t)
            loss.backward()
            optimizer.step()

        self.model.to("cpu")
        return self

    def _predict_proba_raw(self, X):
        # Inference always runs on CPU — see the note in fit().
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(self._scale(X), dtype=torch.float32)
        self.model.to("cpu")
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(X_t), dim=1).numpy()
        return probs

    def predict_proba(self, X):
        return self._predict_proba_raw(X)

    def predict(self, X):
        return self._predict_proba_raw(X).argmax(axis=1)
