"""
Random Kitchen Sinks (RKS) and 1-hidden-layer MLP models for Problem 3,
filled in from the provided skeleton.
"""
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn

"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int = 0):
        """
            D: total number of random features (= 2h; h sin features + h cos features)
            sigma: kernel scale; w_j ~ N(0, I / sigma^2)
            ridge_lambda: ridge regularization strength
        """
        assert D % 2 == 0, "D = 2h must be even (sin + cos pairs)"
        self.D = D
        self.h = D // 2
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.rng = np.random.default_rng(seed)
        self.W = None  # (d, h), drawn lazily once the input dimension is known
        self.b = self.rng.uniform(0.0, 2 * np.pi, size=self.h)
        self.coef_ = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Map X (n, d) to [sin(XW + b), cos(XW + b), 1] of shape (n, D + 1).
        The last column is the intercept.
        """
        proj = X @ self.W + self.b
        return np.hstack([np.sin(proj), np.cos(proj), np.ones((X.shape[0], 1))])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Closed-form ridge regression on the random features:
            coef = (Phi^T Phi + lambda * n * I)^{-1} Phi^T y
        The intercept column is not penalized. lambda is scaled by n so it is the
        penalty on the *mean* squared error.
        """
        if self.W is None:
            self.W = self.rng.normal(0.0, 1.0 / self.sigma, size=(X.shape[1], self.h))
        phi = self._features(X)  # shape (n, D + 1)
        n = phi.shape[0]
        penalty = self.ridge_lambda * n * np.eye(phi.shape[1])
        penalty[-1, -1] = 0.0  # don't shrink the intercept
        self.coef_ = np.linalg.solve(phi.T @ phi + penalty, phi.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        phi = self._features(X)
        return phi @ self.coef_  # shape (n,)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.ravel(y_true) - np.ravel(y_pred)) ** 2))


def sweep_rks_model(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, param: str = "sigma", **fixed) -> Tuple:
    """
    Sweep one RKS hyper-parameter (`param` in {"D", "sigma", "ridge_lambda"}) with the
    others held fixed at `fixed` (or the class defaults).
    Returns (best_model, best_val_mse, best_value, train_mses, val_mses).
    """
    best_model, best_val_mse, best_value = None, float("inf"), None
    train_mses, val_mses = [], []
    for v in sweep_values:
        model = RandomKitchenSinks(**{**fixed, param: v}).fit(X_train, y_train)
        train_mse = mse(y_train, model.predict(X_train))
        val_mse = mse(y_val, model.predict(X_val))
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_value = model, val_mse, v

    return best_model, best_val_mse, best_value, train_mses, val_mses

"""
For 1-hidden-layer MLP (MLP)
"""
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 64,
                 activation: str = "relu", num_epochs: int = 1500, weight_decay: float = 0.0,
                 device: str = "cpu", seed: int = 0):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.seed = seed
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            ACTIVATIONS[activation](),
            nn.Linear(d_hidden, d_out),
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(self.seed)
        n = X.shape[0]
        for epoch in range(self.num_epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net(X[idx]), y[idx])
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy().ravel()


def mlp_sweep(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, param: str = "d_hidden", **fixed) -> Tuple:
    """
    Sweep one MLP hyper-parameter (e.g. "d_hidden", "lr", "batch_size") with the others fixed.
    Returns (best_model, best_val_mse, best_value, train_mses, val_mses).
    """
    best_model, best_val_mse, best_value = None, float("inf"), None
    train_mses, val_mses = [], []
    for v in sweep_values:
        model = MLP(d_in=X_train.shape[1], **{**fixed, param: v}).fit(X_train, y_train)
        train_mse = mse(y_train, model.predict(X_train))
        val_mse = mse(y_val, model.predict(X_val))
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_value = model, val_mse, v

    return best_model, best_val_mse, best_value, train_mses, val_mses
