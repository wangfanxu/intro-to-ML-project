"""A scikit-learn-compatible wrapper around a small PyTorch MLP classifier,
so it can be dropped into the same Pipeline / cross_validate machinery used
for the other models.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import RANDOM_STATE


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Minimal PyTorch MLP wrapped to expose fit/predict/predict_proba."""

    def __init__(self, hidden_dims=(64, 32), lr=1e-3, epochs=20,
                 batch_size=256, weight_decay=1e-5, random_state=RANDOM_STATE):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _to_dense_tensor(self, X):
        if sparse.issparse(X):
            X = X.toarray()
        return torch.tensor(np.asarray(X), dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_t = self._to_dense_tensor(X)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        self.classes_ = np.unique(y)
        self.model_ = _MLP(X_t.shape[1], self.hidden_dims)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_t = self._to_dense_tensor(X)
            logits = self.model_(X_t)
            probs_pos = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs_pos, probs_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
