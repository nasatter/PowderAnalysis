#!/usr/bin/env python3
"""Compare powder-spreading regression models with leakage-safe nested holdout CV.

Physical inputs (DOE level numbers are intentionally NOT used):
    PS_R0.25_percent
    RS_mm_s
    RD_mm

Default outputs:
    mean_transverse_rms_roughness_mm
    transverse_rms_roughness_variance_mm2
    mean_clipped_hull_relative_density
    variance_clipped_hull_relative_density
    mean_coordination_number
    coordination_number_variance

Implemented model families:
    quadratic_ols
    quadratic_ridge
    gaussian_process
    gaussian_mixture_regression
    rbf_svr
    random_forest
    extra_trees
    fcnn
    fcnn_single_output

Default validation design (27-case DOE):
    - Outer TEST holdout:       3 samples
    - Inner VALIDATION holdout: 3 samples per inner repeat
    - Remaining samples in an outer split are development data.
    - Hyperparameters AND model family are selected only from the inner
      train/validation splits.
    - The outer 3 samples are never used for scaling, hyperparameter selection,
      FCNN early stopping, epoch selection, or model-family selection.

The same inner validation splits are used for every model family in a given
outer split so model comparisons are directly comparable.

Parallel execution:
    The expensive outer-CV work is scheduled as independent
    (outer split x model family) process jobs in one persistent pool
    (--workers 0 = all CPU cores). This lets different model families and
    different outer splits run at the same time without repeatedly creating
    process pools. Each worker uses one CPU thread internally to prevent nested
    oversubscription. When multiprocessing is enabled, FCNN CV jobs use CPU so
    many tiny networks can train concurrently; the final deployment refit can
    still use CUDA.

FCNN stopping:
    Each inner FCNN run trains up to max_epochs, saves the weights with the
    lowest inner-validation normalized MSE, and stops when that validation loss
    has not improved by min_delta for patience epochs (after min_epochs). The
    best checkpoint is restored. For the outer refit, the selected FCNN is
    retrained on all outer development samples for the median best epoch found
    across the inner validation repeats, then evaluated once on the untouched
    outer test samples.

Example:
    python train_powder_models.py powder_features.csv

Larger hyperparameter grids:
    python train_powder_models.py powder_features.csv --search-profile full
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from torch import nn


DEFAULT_INPUTS = [
    "PS_R0.25_percent",
    "RS_mm_s",
    "RD_mm",
]

DEFAULT_OUTPUTS = [
    "mean_transverse_rms_roughness_mm",
    "transverse_rms_roughness_variance_mm2",
    "mean_clipped_hull_relative_density",
    "variance_clipped_hull_relative_density",
    "mean_coordination_number",
    "coordination_number_variance",
]

FAMILIES = [
    "quadratic_ols",
    "quadratic_ridge",
    "gaussian_process",
    "gaussian_mixture_regression",
    "rbf_svr",
    "random_forest",
    "extra_trees",
    "fcnn",
    "fcnn_single_output",
]
SELECTED_PIPELINE_NAME = "validation_selected_model"
BASELINE_NAME = "training_mean_baseline"


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0, ddof=0)
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)):
            raise ValueError("Non-finite values encountered while fitting standardizer.")
        scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.scale + self.mean

    def as_dict(self, names: list[str]) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": float(mu), "std": float(sd)}
            for name, mu, sd in zip(names, self.mean, self.scale)
        }


class GaussianMixtureRegressor:
    """Gaussian-mixture regression using p([X,Y]) and E[Y|X]."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        reg_covar: float = 1e-4,
        n_init: int = 3,
        max_iter: int = 500,
        random_state: int | None = None,
    ):
        self.n_components = int(n_components)
        self.covariance_type = str(covariance_type)
        self.reg_covar = float(reg_covar)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.gmm: GaussianMixture | None = None
        self.n_inputs_: int | None = None
        self.n_outputs_: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianMixtureRegressor":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.n_inputs_ = x.shape[1]
        self.n_outputs_ = y.shape[1]
        joint = np.hstack([x, y])
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.gmm.fit(joint)
        return self

    def _component_covariance(self, k: int) -> np.ndarray:
        assert self.gmm is not None
        d = self.gmm.means_.shape[1]
        if self.gmm.covariance_type == "full":
            return np.asarray(self.gmm.covariances_[k], dtype=np.float64)
        if self.gmm.covariance_type == "tied":
            return np.asarray(self.gmm.covariances_, dtype=np.float64)
        if self.gmm.covariance_type == "diag":
            return np.diag(np.asarray(self.gmm.covariances_[k], dtype=np.float64))
        if self.gmm.covariance_type == "spherical":
            return np.eye(d, dtype=np.float64) * float(self.gmm.covariances_[k])
        raise ValueError(f"Unknown covariance type: {self.gmm.covariance_type}")

    @staticmethod
    def _log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        d = x.shape[1]
        cov = np.asarray(cov, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            cov = cov + np.eye(d) * 1e-8
            sign, logdet = np.linalg.slogdet(cov)
        delta = x - mean
        try:
            solved = np.linalg.solve(cov, delta.T).T
        except np.linalg.LinAlgError:
            solved = delta @ np.linalg.pinv(cov).T
        quad = np.sum(delta * solved, axis=1)
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.gmm is None or self.n_inputs_ is None or self.n_outputs_ is None:
            raise RuntimeError("GaussianMixtureRegressor must be fit before predict().")
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        k_count = self.gmm.n_components
        cond_means = np.empty((n, k_count, self.n_outputs_), dtype=np.float64)
        log_weights = np.empty((n, k_count), dtype=np.float64)

        for k in range(k_count):
            mu = np.asarray(self.gmm.means_[k], dtype=np.float64)
            mu_x = mu[: self.n_inputs_]
            mu_y = mu[self.n_inputs_ :]
            cov = self._component_covariance(k)
            s_xx = cov[: self.n_inputs_, : self.n_inputs_]
            s_yx = cov[self.n_inputs_ :, : self.n_inputs_]
            dx = (x - mu_x).T
            try:
                gain_dx = np.linalg.solve(s_xx, dx)
            except np.linalg.LinAlgError:
                gain_dx = np.linalg.pinv(s_xx) @ dx
            cond_means[:, k, :] = (mu_y[:, None] + s_yx @ gain_dx).T
            log_weights[:, k] = (
                np.log(max(float(self.gmm.weights_[k]), np.finfo(float).tiny))
                + self._log_gaussian(x, mu_x, s_xx)
            )

        row_max = np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights - row_max)
        weights /= np.sum(weights, axis=1, keepdims=True)
        return np.sum(cond_means * weights[:, :, None], axis=1)


class PowderFCNN(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for next_width in hidden:
            layers.append(nn.Linear(width, next_width))
            layers.append(nn.ReLU())
            width = next_width
        layers.append(nn.Linear(width, n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Each CV worker is already one process. Keep a tiny FCNN from spawning
    # additional CPU threads inside that process (nested oversubscription).
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fcnn_early_stopping(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    x_val_z: np.ndarray,
    y_val_z: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[PowderFCNN, int, float, int]:
    """Train FCNN using ONLY the supplied inner validation set for stopping.

    Returns restored best model, best epoch (1-based), best validation MSE,
    and the actual stopping epoch.
    """
    set_seed(seed)
    hidden = tuple(int(v) for v in params["hidden"])
    model = PowderFCNN(x_train_z.shape[1], y_train_z.shape[1], hidden).to(device)
    x_train_t = torch.as_tensor(x_train_z, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train_z, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val_z, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val_z, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    max_epochs = int(params["max_epochs"])
    patience = int(params["patience"])
    min_epochs = int(params["min_epochs"])
    min_delta = float(params["min_delta"])

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stop_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss_fn(model(x_train_t), y_train_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            stop_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError("FCNN early stopping failed to save a best checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, int(best_epoch), float(best_loss), int(stop_epoch)


def train_fcnn_fixed_epochs(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    params: dict[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
) -> PowderFCNN:
    """Refit selected FCNN on all available development data.

    Epoch count comes from inner-validation best checkpoints; no outer-test data
    are used to choose it.
    """
    set_seed(seed)
    hidden = tuple(int(v) for v in params["hidden"])
    model = PowderFCNN(x_train_z.shape[1], y_train_z.shape[1], hidden).to(device)
    x_train_t = torch.as_tensor(x_train_z, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train_z, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    return model


def predict_fcnn(model: PowderFCNN, x_z: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x_z, dtype=torch.float32, device=device)
        return model(x_t).cpu().numpy().astype(np.float64)


def train_fcnn_single_output_early_stopping(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    x_val_z: np.ndarray,
    y_val_z: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[list[PowderFCNN], list[int], float, list[int]]:
    """Train one independent all-input FCNN for each output column.

    Each network sees every input feature but predicts exactly one output. The
    returned validation score is the mean normalized MSE across all outputs, so
    it is directly comparable with the existing multi-output FCNN objective.
    """
    models: list[PowderFCNN] = []
    best_epochs: list[int] = []
    stop_epochs: list[int] = []
    pred_columns: list[np.ndarray] = []

    for output_index in range(y_train_z.shape[1]):
        model, best_epoch, _best_loss, stop_epoch = train_fcnn_early_stopping(
            x_train_z,
            y_train_z[:, output_index : output_index + 1],
            x_val_z,
            y_val_z[:, output_index : output_index + 1],
            params,
            seed + 10_000 * output_index,
            device,
        )
        models.append(model)
        best_epochs.append(int(best_epoch))
        stop_epochs.append(int(stop_epoch))
        pred_columns.append(predict_fcnn(model, x_val_z, device))

    pred_z = np.column_stack([col[:, 0] for col in pred_columns])
    score = normalized_mse(y_val_z, pred_z)
    return models, best_epochs, float(score), stop_epochs


def train_fcnn_single_output_fixed_epochs(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    params: dict[str, Any],
    epochs_by_output: list[int],
    seed: int,
    device: torch.device,
) -> list[PowderFCNN]:
    """Refit one all-input/single-output FCNN per output column."""
    if len(epochs_by_output) != y_train_z.shape[1]:
        raise ValueError(
            "Single-output FCNN refit requires one epoch count per output column"
        )
    models: list[PowderFCNN] = []
    for output_index, epochs in enumerate(epochs_by_output):
        models.append(
            train_fcnn_fixed_epochs(
                x_train_z,
                y_train_z[:, output_index : output_index + 1],
                params,
                int(epochs),
                seed + 10_000 * output_index,
                device,
            )
        )
    return models


def predict_fcnn_single_output(
    models: list[PowderFCNN], x_z: np.ndarray, device: torch.device
) -> np.ndarray:
    """Predict all outputs using independent single-output FCNNs."""
    return np.column_stack(
        [predict_fcnn(model, x_z, device)[:, 0] for model in models]
    ).astype(np.float64)


def _gpr_kernel(kind: str, n_features: int):
    constant = ConstantKernel(1.0, (1e-3, 1e3))
    if kind == "rbf":
        smooth = RBF(
            length_scale=np.ones(n_features, dtype=float),
            length_scale_bounds=(1e-2, 1e2),
        )
    elif kind == "matern32":
        smooth = Matern(
            length_scale=np.ones(n_features, dtype=float),
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5,
        )
    elif kind == "matern52":
        smooth = Matern(
            length_scale=np.ones(n_features, dtype=float),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
    else:
        raise ValueError(f"Unknown GPR kernel kind: {kind}")
    noise = WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1.0))
    return constant * smooth + noise


def fit_classical_model(
    family: str,
    params: dict[str, Any],
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    seed: int,
):
    if family == "quadratic_ols":
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", LinearRegression()),
        ])
        model.fit(x_train_z, y_train_z)
        return model

    if family == "quadratic_ridge":
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Ridge(alpha=float(params["alpha"]))),
        ])
        model.fit(x_train_z, y_train_z)
        return model

    if family == "gaussian_process":
        models = []
        for output_index in range(y_train_z.shape[1]):
            gpr = GaussianProcessRegressor(
                kernel=_gpr_kernel(str(params["kernel"]), x_train_z.shape[1]),
                alpha=float(params["alpha"]),
                normalize_y=False,
                n_restarts_optimizer=int(params["restarts"]),
                random_state=seed + output_index,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gpr.fit(x_train_z, y_train_z[:, output_index])
            models.append(gpr)
        return models

    if family == "gaussian_mixture_regression":
        model = GaussianMixtureRegressor(
            n_components=int(params["n_components"]),
            covariance_type=str(params["covariance_type"]),
            reg_covar=float(params["reg_covar"]),
            n_init=int(params["n_init"]),
            max_iter=int(params["max_iter"]),
            random_state=seed,
        )
        model.fit(x_train_z, y_train_z)
        return model

    if family == "rbf_svr":
        models = []
        for output_index in range(y_train_z.shape[1]):
            svr = SVR(
                kernel="rbf",
                C=float(params["C"]),
                gamma=params["gamma"],
                epsilon=float(params["epsilon"]),
            )
            svr.fit(x_train_z, y_train_z[:, output_index])
            models.append(svr)
        return models

    if family == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=1,
        )
        model.fit(x_train_z, y_train_z)
        return model

    if family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=1,
        )
        model.fit(x_train_z, y_train_z)
        return model

    raise ValueError(f"Unknown classical family: {family}")


def predict_model(family: str, model, x_z: np.ndarray, device: torch.device) -> np.ndarray:
    if family in {"gaussian_process", "rbf_svr"}:
        return np.column_stack([est.predict(x_z) for est in model]).astype(np.float64)
    if family == "gaussian_mixture_regression":
        return np.asarray(model.predict(x_z), dtype=np.float64)
    if family == "fcnn":
        return predict_fcnn(model, x_z, device)
    if family == "fcnn_single_output":
        return predict_fcnn_single_output(model, x_z, device)
    prediction = np.asarray(model.predict(x_z), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    return prediction


def parameter_grids(profile: str) -> dict[str, list[dict[str, Any]]]:
    if profile == "standard":
        return {
            "quadratic_ols": [{}],
            "quadratic_ridge": list(ParameterGrid({
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            })),
            "gaussian_process": list(ParameterGrid({
                "kernel": ["rbf", "matern32", "matern52"],
                "alpha": [1e-8],
                "restarts": [0],
            })),
            "gaussian_mixture_regression": list(ParameterGrid({
                "n_components": [1, 2, 3],
                "covariance_type": ["full", "diag"],
                "reg_covar": [1e-4, 1e-2],
                "n_init": [3],
                "max_iter": [500],
            })),
            "rbf_svr": list(ParameterGrid({
                "C": [1.0, 10.0],
                "gamma": ["scale", 0.5],
                "epsilon": [0.05, 0.15],
            })),
            "random_forest": list(ParameterGrid({
                "n_estimators": [200],
                "max_depth": [None, 4, 8],
                "min_samples_leaf": [1, 2],
                "max_features": [1.0],
            })),
            "extra_trees": list(ParameterGrid({
                "n_estimators": [200],
                "max_depth": [None, 4, 8],
                "min_samples_leaf": [1, 2],
                "max_features": [1.0],
            })),
            "fcnn": list(ParameterGrid({
                "hidden": [(8,), (16,), (16, 8)],
                "lr": [1e-3, 3e-3],
                "weight_decay": [1e-4, 1e-3],
                "max_epochs": [5000],
                "patience": [250],
                "min_epochs": [100],
                "min_delta": [1e-5],
            })),
            "fcnn_single_output": list(ParameterGrid({
                "hidden": [(8,), (16,), (16, 8)],
                "lr": [1e-3, 3e-3],
                "weight_decay": [1e-4, 1e-3],
                "max_epochs": [5000],
                "patience": [250],
                "min_epochs": [100],
                "min_delta": [1e-5],
            })),
        }

    if profile == "full":
        return {
            "quadratic_ols": [{}],
            "quadratic_ridge": list(ParameterGrid({
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            })),
            "gaussian_process": list(ParameterGrid({
                "kernel": ["rbf", "matern32", "matern52"],
                "alpha": [1e-10, 1e-8, 1e-6],
                "restarts": [0, 2],
            })),
            "gaussian_mixture_regression": list(ParameterGrid({
                "n_components": [1, 2, 3, 4],
                "covariance_type": ["full", "tied", "diag"],
                "reg_covar": [1e-6, 1e-4, 1e-2],
                "n_init": [5],
                "max_iter": [1000],
            })),
            "rbf_svr": list(ParameterGrid({
                "C": [0.3, 1.0, 3.0, 10.0, 30.0],
                "gamma": ["scale", 0.1, 0.3, 1.0],
                "epsilon": [0.02, 0.05, 0.1, 0.2],
            })),
            "random_forest": list(ParameterGrid({
                "n_estimators": [300, 700],
                "max_depth": [None, 3, 6, 10],
                "min_samples_leaf": [1, 2, 3],
                "max_features": [1.0, "sqrt"],
            })),
            "extra_trees": list(ParameterGrid({
                "n_estimators": [300, 700],
                "max_depth": [None, 3, 6, 10],
                "min_samples_leaf": [1, 2, 3],
                "max_features": [1.0, "sqrt"],
            })),
            "fcnn": list(ParameterGrid({
                "hidden": [(8,), (16,), (16, 8), (16, 16), (32, 16)],
                "lr": [3e-4, 1e-3, 3e-3],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "max_epochs": [6000],
                "patience": [300],
                "min_epochs": [100],
                "min_delta": [1e-5],
            })),
            "fcnn_single_output": list(ParameterGrid({
                "hidden": [(8,), (16,), (16, 8), (16, 16), (32, 16)],
                "lr": [3e-4, 1e-3, 3e-3],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "max_epochs": [6000],
                "patience": [300],
                "min_epochs": [100],
                "min_delta": [1e-5],
            })),
        }

    raise ValueError(f"Unknown search profile: {profile}")


def make_holdout_splits(
    n_samples: int,
    holdout: int,
    repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if holdout <= 0 or holdout >= n_samples:
        raise ValueError("holdout must be > 0 and < number of samples")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_samples, dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    max_unique = math.comb(n_samples, holdout) if n_samples < 100 else repeats
    target = min(repeats, max_unique)
    attempts = 0
    while len(splits) < target:
        attempts += 1
        val_idx = np.sort(rng.choice(all_indices, size=holdout, replace=False))
        key = tuple(int(v) for v in val_idx)
        if key in seen:
            if attempts > repeats * 100 + 1000:
                break
            continue
        seen.add(key)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        splits.append((all_indices[train_mask], val_idx))
    return splits


def normalized_mse(y_true_z: np.ndarray, y_pred_z: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred_z) - np.asarray(y_true_z)) ** 2))


def jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _evaluate_inner_job(
    family: str,
    params: dict[str, Any],
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_seed: int,
    device_string: str,
) -> dict[str, Any]:
    """Evaluate one hyperparameter candidate on one inner holdout split."""
    try:
        x_scaler = Standardizer.fit(x_dev[train_idx])
        y_scaler = Standardizer.fit(y_dev[train_idx])
        x_train_z = x_scaler.transform(x_dev[train_idx])
        y_train_z = y_scaler.transform(y_dev[train_idx])
        x_val_z = x_scaler.transform(x_dev[val_idx])
        y_val_z = y_scaler.transform(y_dev[val_idx])
        worker_device = torch.device(device_string)

        result: dict[str, Any] = {}
        if family == "fcnn":
            model, best_epoch, _best_loss, stop_epoch = train_fcnn_early_stopping(
                x_train_z,
                y_train_z,
                x_val_z,
                y_val_z,
                params,
                model_seed,
                worker_device,
            )
            result["best_epoch"] = int(best_epoch)
            result["stop_epoch"] = int(stop_epoch)
        elif family == "fcnn_single_output":
            model, best_epochs, single_score, stop_epochs = (
                train_fcnn_single_output_early_stopping(
                    x_train_z,
                    y_train_z,
                    x_val_z,
                    y_val_z,
                    params,
                    model_seed,
                    worker_device,
                )
            )
            result["best_epochs_by_output"] = [int(v) for v in best_epochs]
            result["stop_epochs_by_output"] = [int(v) for v in stop_epochs]
            result["score"] = float(single_score)
        else:
            model = fit_classical_model(
                family, params, x_train_z, y_train_z, model_seed
            )

        pred_z = predict_model(family, model, x_val_z, worker_device)
        result["score"] = normalized_mse(y_val_z, pred_z)
        result["error"] = ""
        return result
    except Exception as exc:
        return {
            "score": math.inf,
            "best_epoch": None,
            "stop_epoch": None,
            "best_epochs_by_output": None,
            "stop_epochs_by_output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def tune_family(
    family: str,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    candidates: list[dict[str, Any]],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: torch.device,
    workers: int,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], Any]:
    """Select one family's hyperparameters using inner holdout validation.

    Candidate x inner-split evaluations are independent and are distributed
    across processes when workers != 1. CUDA FCNN evaluations remain serial to
    avoid launching multiple training processes against one GPU.
    """
    jobs: list[tuple[int, int, dict[str, Any], np.ndarray, np.ndarray, int]] = []
    for candidate_index, params in enumerate(candidates):
        for inner_number, (train_idx, val_idx) in enumerate(inner_splits):
            model_seed = seed + 10_000 * candidate_index + 100 * inner_number
            jobs.append(
                (candidate_index, inner_number, params, train_idx, val_idx, model_seed)
            )

    use_workers = int(workers)
    if family in {"fcnn", "fcnn_single_output"} and device.type == "cuda":
        use_workers = 1

    def run_one(job):
        candidate_index, inner_number, params, train_idx, val_idx, model_seed = job
        result = _evaluate_inner_job(
            family,
            params,
            x_dev,
            y_dev,
            train_idx,
            val_idx,
            model_seed,
            str(device if family in {"fcnn", "fcnn_single_output"} else torch.device("cpu")),
        )
        return candidate_index, inner_number, result

    if use_workers == 1 or len(jobs) <= 1:
        evaluated = [run_one(job) for job in jobs]
    else:
        with joblib.parallel_backend("loky", inner_max_num_threads=1):
            evaluated = joblib.Parallel(n_jobs=use_workers, batch_size=1)(
                joblib.delayed(run_one)(job) for job in jobs
            )

    by_candidate: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in candidates
    ]
    for candidate_index, inner_number, result in evaluated:
        by_candidate[candidate_index].append((inner_number, result))

    results: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(candidates):
        ordered = [r for _, r in sorted(by_candidate[candidate_index], key=lambda z: z[0])]
        split_scores = [float(r["score"]) for r in ordered]
        errors = [str(r.get("error", "")) for r in ordered if r.get("error")]
        result: dict[str, Any] = {
            "params": params,
            "mean_inner_normalized_mse": float(np.mean(split_scores)),
            "std_inner_normalized_mse": float(np.std(split_scores, ddof=0)),
            "inner_scores": split_scores,
            "errors": errors,
        }
        if family == "fcnn":
            best_epochs = [int(r["best_epoch"]) for r in ordered if r.get("best_epoch") is not None]
            stop_epochs = [int(r["stop_epoch"]) for r in ordered if r.get("stop_epoch") is not None]
            if best_epochs:
                result["median_best_epoch"] = int(round(float(np.median(best_epochs))))
            else:
                result["median_best_epoch"] = None
            result["best_epochs"] = best_epochs
            result["stop_epochs"] = stop_epochs
        elif family == "fcnn_single_output":
            epoch_rows = [
                r["best_epochs_by_output"]
                for r in ordered
                if r.get("best_epochs_by_output") is not None
            ]
            stop_rows = [
                r["stop_epochs_by_output"]
                for r in ordered
                if r.get("stop_epochs_by_output") is not None
            ]
            if epoch_rows:
                epoch_array = np.asarray(epoch_rows, dtype=float)
                result["median_best_epochs_by_output"] = [
                    int(round(float(v))) for v in np.median(epoch_array, axis=0)
                ]
            else:
                result["median_best_epochs_by_output"] = None
            result["best_epochs_by_output"] = epoch_rows
            result["stop_epochs_by_output"] = stop_rows
        results.append(result)

    finite_indices = [
        i for i, result in enumerate(results)
        if np.isfinite(result["mean_inner_normalized_mse"])
    ]
    if not finite_indices:
        all_errors = [err for result in results for err in result.get("errors", [])]
        raise RuntimeError(
            f"All {family} candidates failed during inner validation. "
            + ("First error: " + all_errors[0] if all_errors else "")
        )

    best_index = min(
        finite_indices,
        key=lambda i: results[i]["mean_inner_normalized_mse"],
    )
    best_params = candidates[best_index]
    best_score = float(results[best_index]["mean_inner_normalized_mse"])
    refit_epochs: Any = None
    if family == "fcnn":
        selected_epoch = results[best_index].get("median_best_epoch")
        if selected_epoch is None:
            raise RuntimeError("Selected FCNN candidate has no valid best epoch")
        refit_epochs = int(selected_epoch)
    elif family == "fcnn_single_output":
        selected_epochs = results[best_index].get("median_best_epochs_by_output")
        if selected_epochs is None:
            raise RuntimeError(
                "Selected single-output FCNN candidate has no valid best epochs"
            )
        refit_epochs = [int(v) for v in selected_epochs]
    return best_params, best_score, results, refit_epochs


def fit_refit_model(
    family: str,
    params: dict[str, Any],
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    seed: int,
    device: torch.device,
    fcnn_epochs: Any,
):
    if family == "fcnn":
        if fcnn_epochs is None:
            raise ValueError("FCNN refit requires epoch count selected from inner validation")
        return train_fcnn_fixed_epochs(
            x_train_z, y_train_z, params, fcnn_epochs, seed, device
        )
    if family == "fcnn_single_output":
        if fcnn_epochs is None:
            raise ValueError(
                "Single-output FCNN refit requires epoch counts selected from inner validation"
            )
        return train_fcnn_single_output_fixed_epochs(
            x_train_z, y_train_z, params, list(fcnn_epochs), seed, device
        )
    return fit_classical_model(family, params, x_train_z, y_train_z, seed)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error = y_pred - y_true
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for j in range(y_true.shape[1]):
        if np.var(y_true[:, j]) > np.finfo(np.float64).eps:
            r2[j] = r2_score(y_true[:, j], y_pred[:, j])
    return rmse, mae, r2


def architecture_description(family: str) -> str:
    return {
        "quadratic_ols": (
            "Second-order response surface with linear, squared, and pairwise "
            "interaction terms. It has no tunable regularization, but it is still "
            "scored on the same inner validation samples for fair family selection."
        ),
        "quadratic_ridge": (
            "Second-order response surface with L2 regularization. Ridge alpha is "
            "selected by inner validation."
        ),
        "gaussian_process": (
            "Six independent Gaussian-process regressors. Kernel family and other "
            "grid parameters are selected by inner validation; kernel continuous "
            "parameters are optimized using training data only."
        ),
        "gaussian_mixture_regression": (
            "Gaussian mixture regression fits a joint Gaussian mixture to the three "
            "standardized inputs and six standardized outputs, then predicts the "
            "conditional mean E[Y|X]. Number of components, covariance structure, "
            "and covariance regularization are selected by inner validation."
        ),
        "rbf_svr": (
            "Six independent RBF epsilon-SVR models. C, gamma, and epsilon are "
            "selected by inner validation."
        ),
        "random_forest": (
            "Multi-output random forest. Tree depth, leaf size, and feature "
            "subsampling candidates are selected by inner validation."
        ),
        "extra_trees": (
            "Multi-output extremely randomized trees. Hyperparameters are selected "
            "by inner validation."
        ),
        "fcnn": (
            "Multi-output ReLU FCNN. Architecture, learning rate, and weight decay "
            "are selected by inner validation. Each inner run also uses the same "
            "validation split for best-checkpoint early stopping."
        ),
        "fcnn_single_output": (
            "Bank of independent single-output ReLU FCNNs. Every network receives "
            "all physical inputs but predicts exactly one output. One shared "
            "architecture/optimizer hyperparameter candidate is selected by the mean "
            "normalized validation MSE across outputs; each output network keeps its "
            "own validation-selected stopping epoch."
        ),
    }[family]


def write_method_report(
    path: Path,
    args: argparse.Namespace,
    grids: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Powder-spreading model comparison method",
        "",
        "## Inputs and outputs",
        "",
        "Physical inputs only (DOE level numbers are not used):",
        "",
    ]
    lines += [f"- `{name}`" for name in args.inputs]
    lines += ["", "Predicted outputs:", ""]
    lines += [f"- `{name}`" for name in args.outputs]
    lines += [
        "",
        "## 3 + 3 nested holdout design",
        "",
        f"Each outer split holds out **{args.holdout} samples as the final test set**. "
        f"From the remaining development samples, each inner repeat holds out "
        f"**{args.inner_holdout} samples as validation**. The inner holdout is "
        f"repeated **{args.inner_repeats} times** to make hyperparameter selection "
        "less dependent on a single three-sample validation draw.",
        "",
        "The exact same inner train/validation splits are used for every model "
        "family within an outer split. Input and output z-score transforms are fit "
        "only on each corresponding training subset and then applied to validation "
        "or test data. The outer test samples are never used to select scaling, "
        "hyperparameters, FCNN stopping epoch, or model family.",
        "",
        "For each model family, the hyperparameter candidate with the lowest mean "
        "inner-validation normalized MSE is selected. The family with the lowest "
        "inner-validation normalized MSE is then the validation-selected model for "
        "that outer split. That model is refit on all outer development samples and "
        "evaluated once on the untouched outer test samples.",
        "",
        "## FCNN stopping rule",
        "",
        "For every inner train/validation split, each FCNN saves the checkpoint with "
        "the lowest validation normalized MSE. The multi-output FCNN uses one network "
        "for all outputs; `fcnn_single_output` trains one all-input network per output. "
        "Training stops when validation loss "
        "has not improved by `min_delta` for `patience` epochs after `min_epochs`, "
        "or when `max_epochs` is reached. The best checkpoint is restored. The "
        "median best epoch across the inner repeats is then used to refit the "
        "multi-output FCNN. For `fcnn_single_output`, the median best epoch is computed "
        "separately for each output network before refitting on all outer development "
        "samples and evaluating the untouched outer test set.",
        "",
        "Classical models do not have an epoch stopping rule, but their tunable "
        "hyperparameters are selected using the same inner validation samples.",
        "",
        "## Parallel execution",
        "",
        "Outer-split x model-family evaluations are distributed through one "
        "persistent process pool. `--workers 0` uses all available CPU cores and "
        "`--workers 1` disables multiprocessing. Each worker runs its family's "
        "inner grid serially with one CPU thread, preventing nested "
        "oversubscription and repeated pool startup. With multiprocessing enabled, "
        "FCNN cross-validation runs on CPU so independent tiny networks can run "
        "concurrently; the final deployment refit may still use CUDA.",
        "",
        "## Models and grids",
        "",
    ]
    for family in FAMILIES:
        lines += [
            f"### {family}",
            "",
            architecture_description(family),
            "",
            f"Grid size ({args.search_profile} profile): **{len(grids[family])}** candidates.",
            "",
            "```json",
            json.dumps([jsonable_params(p) for p in grids[family]], indent=2),
            "```",
            "",
        ]
    lines += [
        "## Primary reported result",
        "",
        f"`{SELECTED_PIPELINE_NAME}` is the leakage-safe performance of the full "
        "model-selection procedure: validation chooses the family/configuration, "
        "and only then is performance measured on the outer test samples. Individual "
        "family outer-test scores are also reported for diagnostic comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_empirical_report(
    path: Path,
    args: argparse.Namespace,
    overall: pd.DataFrame,
    per_output: pd.DataFrame,
    best_by_output: pd.DataFrame,
    selection_frequency: pd.DataFrame,
    final_params: dict[str, dict[str, Any]],
    final_refit_epochs: dict[str, int | None],
    final_selected_family: str,
) -> None:
    lines = [
        "# Powder-spreading model comparison",
        "",
        "## Validation design",
        "",
        f"Outer test holdout: **{args.holdout} samples**. Inner validation holdout: "
        f"**{args.inner_holdout} samples**, repeated **{args.inner_repeats} times** "
        f"inside each outer split. There are **{args.cv_repeats} outer splits**.",
        "",
        "All scaling, hyperparameter selection, FCNN early stopping, and model-family "
        "selection occur without access to the outer test samples.",
        "",
        "## Overall outer-test ranking",
        "",
        overall.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Validation-selected family frequency",
        "",
        selection_frequency.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Best individual family by output",
        "",
        best_by_output.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Per-output results",
        "",
        per_output.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Final all-data selection",
        "",
        f"Using only repeated inner 3-sample validation on the full dataset, the "
        f"selected deployment family is **{final_selected_family}**.",
        "",
        "## Model notes and final selected hyperparameters",
        "",
    ]
    for family in args.models:
        lines += [
            f"### {family}",
            "",
            architecture_description(family),
            "",
            "```json",
            json.dumps(jsonable_params(final_params[family]), indent=2),
            "```",
            "",
        ]
        if family in {"fcnn", "fcnn_single_output"}:
            lines += [
                f"Final refit epoch count selected from inner-validation best "
                f"checkpoints: **{final_refit_epochs[family]}**.",
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_comparison_plot(overall: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_data = overall.loc[overall["model"] != BASELINE_NAME].copy()
    plot_data = plot_data.sort_values("normalized_mse")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(plot_data["model"], plot_data["normalized_mse"])
    baseline_row = overall.loc[overall["model"] == BASELINE_NAME]
    if not baseline_row.empty:
        ax.axhline(
            float(baseline_row.iloc[0]["normalized_mse"]),
            linestyle="--",
            label="Training-mean baseline",
        )
        ax.legend()
    ax.set_ylabel("Outer-test normalized MSE")
    ax.set_xlabel("Model")
    ax.set_title("Powder-spreading regression model comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="powder_features.csv from powder analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <CSV directory>/model_comparison_results)",
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--outputs", nargs="+", default=DEFAULT_OUTPUTS)
    parser.add_argument(
        "--holdout",
        type=int,
        default=3,
        help="outer TEST samples per split (default: 3)",
    )
    parser.add_argument(
        "--inner-holdout",
        type=int,
        default=3,
        help="inner VALIDATION samples per repeat (default: 3)",
    )
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=5,
        help="repeated inner validation draws per outer split (default: 5)",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=20,
        help="outer repeated 3-sample test splits (default: 20)",
    )
    parser.add_argument("--search-profile", choices=("standard", "full"), default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "processes for independent outer-split x model-family jobs; "
            "0 uses all CPU cores, 1 disables multiprocessing (default: 0)"
        ),
    )
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    args.workers_resolved = -1 if args.workers == 0 else args.workers
    if args.holdout <= 0:
        parser.error("--holdout must be positive")
    if args.inner_holdout <= 0:
        parser.error("--inner-holdout must be positive")
    if args.inner_repeats <= 0:
        parser.error("--inner-repeats must be positive")
    if args.cv_repeats <= 0:
        parser.error("--cv-repeats must be positive")
    return args


def add_prediction_records(
    records: list[dict[str, Any]],
    split_number: int,
    model_name: str,
    frame: pd.DataFrame,
    test_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    y_true_z: np.ndarray,
    pred: np.ndarray,
    pred_z: np.ndarray,
    args: argparse.Namespace,
    selected_family: str | None = None,
) -> None:
    for local_i, row_idx in enumerate(test_idx):
        rec: dict[str, Any] = {
            "split": split_number,
            "model": model_name,
            "row_index": int(row_idx),
        }
        if selected_family is not None:
            rec["selected_family"] = selected_family
        if "file" in frame.columns:
            rec["file"] = frame.loc[row_idx, "file"]
        for name, value in zip(args.inputs, x[row_idx]):
            rec[name] = float(value)
        for j, name in enumerate(args.outputs):
            rec[f"true__{name}"] = float(y[row_idx, j])
            rec[f"pred__{name}"] = float(pred[local_i, j])
            rec[f"true_z__{name}"] = float(y_true_z[local_i, j])
            rec[f"pred_z__{name}"] = float(pred_z[local_i, j])
        records.append(rec)



def _run_outer_family_job(
    split_number: int,
    family_index: int,
    family: str,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    device_string: str,
) -> dict[str, Any]:
    """Tune/refit one model family for one outer CV split.

    These jobs are independent across both outer split and model family, so the
    parent process can keep one persistent process pool busy instead of opening
    a new pool for every family/grid search.
    """
    start = time.time()
    device = torch.device(device_string)
    x_dev = x[dev_idx]
    y_dev = y[dev_idx]

    inner_seed = args.seed + 1_000_000 * split_number
    inner_splits = make_holdout_splits(
        len(dev_idx), args.inner_holdout, args.inner_repeats, inner_seed
    )

    # Outer transforms are fit only on development data. Inner transforms are
    # still fit independently inside _evaluate_inner_job during tuning.
    outer_x_scaler = Standardizer.fit(x_dev)
    outer_y_scaler = Standardizer.fit(y_dev)
    x_dev_z = outer_x_scaler.transform(x_dev)
    y_dev_z = outer_y_scaler.transform(y_dev)
    x_test_z = outer_x_scaler.transform(x[test_idx])
    y_test_z = outer_y_scaler.transform(y[test_idx])

    tune_seed = args.seed + 10_000_000 * split_number + 100_000 * family_index
    best_params, best_inner_score, candidate_results, refit_epochs = tune_family(
        family,
        x_dev,
        y_dev,
        candidates,
        inner_splits,
        tune_seed,
        device,
        1,  # no nested process pool; parent parallelizes split x family jobs
    )

    model = fit_refit_model(
        family,
        best_params,
        x_dev_z,
        y_dev_z,
        tune_seed + 777,
        device,
        refit_epochs,
    )
    pred_z = predict_model(family, model, x_test_z, device)
    pred = outer_y_scaler.inverse_transform(pred_z)
    outer_mse = normalized_mse(y_test_z, pred_z)

    prediction_records: list[dict[str, Any]] = []
    add_prediction_records(
        prediction_records,
        split_number,
        family,
        frame,
        test_idx,
        x,
        y,
        y_test_z,
        pred,
        pred_z,
        args,
    )

    grid_records: list[dict[str, Any]] = []
    for candidate_number, result in enumerate(candidate_results, start=1):
        grid_records.append({
            "split": split_number,
            "model": family,
            "candidate": candidate_number,
            "params_json": json.dumps(jsonable_params(result["params"]), sort_keys=True),
            "mean_inner_normalized_mse": result["mean_inner_normalized_mse"],
            "std_inner_normalized_mse": result["std_inner_normalized_mse"],
            "inner_scores_json": json.dumps(result.get("inner_scores", [])),
            "fcnn_median_best_epoch": result.get("median_best_epoch", math.nan),
            "fcnn_best_epochs_json": json.dumps(result.get("best_epochs", [])),
            "fcnn_stop_epochs_json": json.dumps(result.get("stop_epochs", [])),
            "fcnn_single_output_median_best_epochs_json": json.dumps(
                result.get("median_best_epochs_by_output", [])
            ),
            "fcnn_single_output_best_epochs_json": json.dumps(
                result.get("best_epochs_by_output", [])
            ),
            "fcnn_single_output_stop_epochs_json": json.dumps(
                result.get("stop_epochs_by_output", [])
            ),
            "errors_json": json.dumps(result.get("errors", [])),
        })

    return {
        "split": split_number,
        "family": family,
        "best_inner_score": float(best_inner_score),
        "best_params": best_params,
        "refit_epochs": refit_epochs,
        "outer_mse": float(outer_mse),
        "pred_z": pred_z,
        "pred": pred,
        "prediction_records": prediction_records,
        "grid_records": grid_records,
        "elapsed": float(time.time() - start),
    }


def main() -> int:
    args = parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV does not exist: {args.csv}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or args.csv.parent / "model_comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_original = pd.read_csv(args.csv)
    required = list(dict.fromkeys(args.inputs + args.outputs))
    missing = [name for name in required if name not in frame_original.columns]
    if missing:
        raise SystemExit("Missing required CSV column(s): " + ", ".join(missing))

    frame = frame_original.copy()
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str).str.lower().eq("ok")].copy()
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    dropped = int(np.count_nonzero(~finite_mask)) + int(len(frame_original) - len(frame))
    frame = frame.loc[finite_mask].reset_index(drop=True)
    numeric = numeric.loc[finite_mask].reset_index(drop=True)

    if len(frame) <= args.holdout + args.inner_holdout:
        raise SystemExit(
            f"Only {len(frame)} usable rows remain; need more than outer holdout "
            f"({args.holdout}) + inner holdout ({args.inner_holdout})."
        )

    x = numeric[args.inputs].to_numpy(dtype=np.float64)
    y = numeric[args.outputs].to_numpy(dtype=np.float64)
    grids_all = parameter_grids(args.search_profile)
    grids = {family: grids_all[family] for family in args.models}
    write_method_report(output_dir / "model_method_report.md", args, grids_all)

    print(f"Usable samples: {len(frame)}; dropped/error rows: {dropped}")
    print("Inputs:  " + ", ".join(args.inputs))
    print("Outputs: " + ", ".join(args.outputs))
    print(f"Device: {device}")
    print(
        "Parallel workers: "
        + ("all CPU cores" if args.workers_resolved == -1 else str(args.workers_resolved))
    )
    if (
        device.type == "cuda"
        and any(f in args.models for f in {"fcnn", "fcnn_single_output"})
        and args.workers_resolved != 1
    ):
        print("  Multiprocess CV will run FCNN workers on CPU; final FCNN refits may use CUDA.")
    print(
        f"Nested holdout: outer TEST={args.holdout}; inner VALIDATION={args.inner_holdout}; "
        f"inner repeats={args.inner_repeats}; outer repeats={args.cv_repeats}; "
        f"profile={args.search_profile}"
    )
    for family in args.models:
        print(f"  {family}: {len(grids[family])} candidate(s)")

    outer_splits = make_holdout_splits(
        len(frame), args.holdout, args.cv_repeats, args.seed
    )
    prediction_records: list[dict[str, Any]] = []
    split_metric_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    grid_score_records: list[dict[str, Any]] = []
    selected_family_records: list[dict[str, Any]] = []

    start_time = time.time()

    # Baseline is cheap and stays in the parent process. The expensive work is
    # scheduled as one persistent pool over (outer split x model family).
    split_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for split_number, (dev_idx, test_idx) in enumerate(outer_splits, start=1):
        split_lookup[split_number] = (dev_idx, test_idx)
        y_dev = y[dev_idx]
        outer_y_scaler = Standardizer.fit(y_dev)
        y_test_z = outer_y_scaler.transform(y[test_idx])
        baseline_pred_z = np.zeros_like(y_test_z)
        baseline_pred = outer_y_scaler.inverse_transform(baseline_pred_z)
        baseline_mse = normalized_mse(y_test_z, baseline_pred_z)
        split_metric_records.append({
            "split": split_number,
            "model": BASELINE_NAME,
            "normalized_mse": baseline_mse,
        })
        add_prediction_records(
            prediction_records,
            split_number,
            BASELINE_NAME,
            frame,
            test_idx,
            x,
            y,
            y_test_z,
            baseline_pred,
            baseline_pred_z,
            args,
        )

    family_jobs: list[tuple[int, int, str, np.ndarray, np.ndarray]] = []
    for split_number, (dev_idx, test_idx) in enumerate(outer_splits, start=1):
        for family_index, family in enumerate(args.models):
            family_jobs.append((split_number, family_index, family, dev_idx, test_idx))

    parallel_outer = args.workers_resolved != 1 and len(family_jobs) > 1
    # Tiny FCNNs are much faster when many independent CPU fits run in parallel
    # than when one tiny fit is launched repeatedly on a single GPU.
    cv_device = torch.device("cpu") if parallel_outer else device
    if (
        parallel_outer
        and device.type == "cuda"
        and any(f in args.models for f in {"fcnn", "fcnn_single_output"})
    ):
        print(
            "  CV FCNN fits will run on CPU across worker processes; "
            "the final deployment refit can still use CUDA."
        )

    def make_job(job):
        split_number, family_index, family, dev_idx, test_idx = job
        return _run_outer_family_job(
            split_number,
            family_index,
            family,
            dev_idx,
            test_idx,
            x,
            y,
            frame,
            args,
            grids[family],
            str(cv_device),
        )

    if not parallel_outer:
        family_results = []
        for i, job in enumerate(family_jobs, start=1):
            split_number, _fi, family, _dev_idx, _test_idx = job
            print(
                f"Running CV job {i}/{len(family_jobs)}: "
                f"outer split {split_number}, {family}"
            )
            family_results.append(make_job(job))
    else:
        print(
            f"Launching {len(family_jobs)} independent outer-split x model-family "
            "jobs in one persistent process pool..."
        )
        with joblib.parallel_backend("loky", inner_max_num_threads=1):
            family_results = joblib.Parallel(
                n_jobs=args.workers_resolved,
                batch_size=1,
                pre_dispatch="2*n_jobs",
                verbose=10,
            )(
                joblib.delayed(_run_outer_family_job)(
                    split_number,
                    family_index,
                    family,
                    dev_idx,
                    test_idx,
                    x,
                    y,
                    frame,
                    args,
                    grids[family],
                    str(cv_device),
                )
                for split_number, family_index, family, dev_idx, test_idx in family_jobs
            )

    by_split: dict[int, dict[str, dict[str, Any]]] = {
        split_number: {} for split_number in split_lookup
    }
    for result in family_results:
        split_number = int(result["split"])
        family = str(result["family"])
        by_split[split_number][family] = result
        prediction_records.extend(result["prediction_records"])
        grid_score_records.extend(result["grid_records"])
        split_metric_records.append({
            "split": split_number,
            "model": family,
            "normalized_mse": result["outer_mse"],
        })
        selection_records.append({
            "split": split_number,
            "model": family,
            "best_inner_normalized_mse": result["best_inner_score"],
            "best_params_json": json.dumps(
                jsonable_params(result["best_params"]), sort_keys=True
            ),
            "fcnn_refit_epochs": result["refit_epochs"],
        })

    # Family selection happens only after every family has completed for that
    # outer split. Selection still uses inner-validation MSE only.
    for split_number in sorted(by_split):
        dev_idx, test_idx = split_lookup[split_number]
        results = by_split[split_number]
        family_inner_scores = {
            family: float(result["best_inner_score"])
            for family, result in results.items()
        }
        selected_family = min(family_inner_scores, key=family_inner_scores.get)
        chosen = results[selected_family]
        selected_outer_mse = float(chosen["outer_mse"])

        outer_y_scaler = Standardizer.fit(y[dev_idx])
        y_test_z = outer_y_scaler.transform(y[test_idx])
        split_metric_records.append({
            "split": split_number,
            "model": SELECTED_PIPELINE_NAME,
            "normalized_mse": selected_outer_mse,
        })
        selected_family_records.append({
            "split": split_number,
            "selected_family": selected_family,
            "selected_inner_normalized_mse": family_inner_scores[selected_family],
            "outer_test_normalized_mse": selected_outer_mse,
        })
        add_prediction_records(
            prediction_records,
            split_number,
            SELECTED_PIPELINE_NAME,
            frame,
            test_idx,
            x,
            y,
            y_test_z,
            chosen["pred"],
            chosen["pred_z"],
            args,
            selected_family=selected_family,
        )

        print(
            f"\nOuter split {split_number}/{len(outer_splits)}: "
            f"development={len(dev_idx)}, test={len(test_idx)}"
        )
        for family in args.models:
            result = results[family]
            print(
                f"  {family:28s} inner val MSE={result['best_inner_score']:.5g}; "
                f"outer holdout MSE={result['outer_mse']:.5g}; "
                f"worker time={result['elapsed']:.1f}s"
            )
        print(
            f"  -> validation selected: {selected_family}; "
            f"outer holdout MSE={selected_outer_mse:.5g}"
        )

    predictions = pd.DataFrame(prediction_records)
    split_metrics = pd.DataFrame(split_metric_records)
    selections = pd.DataFrame(selection_records)
    grid_scores = pd.DataFrame(grid_score_records)
    selected_families = pd.DataFrame(selected_family_records)

    predictions.to_csv(output_dir / "nested_cv_predictions.csv", index=False)
    split_metrics.to_csv(output_dir / "outer_split_metrics.csv", index=False)
    selections.to_csv(output_dir / "selected_hyperparameters_by_split.csv", index=False)
    grid_scores.to_csv(output_dir / "inner_grid_scores.csv", index=False)
    selected_families.to_csv(output_dir / "validation_selected_family_by_split.csv", index=False)

    model_names = [BASELINE_NAME] + args.models + [SELECTED_PIPELINE_NAME]
    overall_records: list[dict[str, Any]] = []
    per_output_records: list[dict[str, Any]] = []

    for model_name in model_names:
        subset = predictions.loc[predictions["model"] == model_name]
        if subset.empty:
            continue
        y_true = np.column_stack([
            subset[f"true__{name}"].to_numpy(float) for name in args.outputs
        ])
        y_pred = np.column_stack([
            subset[f"pred__{name}"].to_numpy(float) for name in args.outputs
        ])
        y_true_z = np.column_stack([
            subset[f"true_z__{name}"].to_numpy(float) for name in args.outputs
        ])
        y_pred_z = np.column_stack([
            subset[f"pred_z__{name}"].to_numpy(float) for name in args.outputs
        ])
        sqerr_z = (y_pred_z - y_true_z) ** 2
        model_split_mse = split_metrics.loc[
            split_metrics["model"] == model_name, "normalized_mse"
        ]
        overall_records.append({
            "model": model_name,
            "normalized_mse": float(np.mean(sqerr_z)),
            "normalized_rmse": float(np.sqrt(np.mean(sqerr_z))),
            "mean_outer_split_mse": float(model_split_mse.mean()),
            "std_outer_split_mse": float(model_split_mse.std(ddof=0)),
        })
        rmse, mae, r2 = regression_metrics(y_true, y_pred)
        for j, output_name in enumerate(args.outputs):
            per_output_records.append({
                "model": model_name,
                "output": output_name,
                "normalized_mse": float(np.mean(sqerr_z[:, j])),
                "normalized_rmse": float(np.sqrt(np.mean(sqerr_z[:, j]))),
                "rmse_physical": float(rmse[j]),
                "mae_physical": float(mae[j]),
                "r2": float(r2[j]),
            })

    overall = pd.DataFrame(overall_records).sort_values("normalized_mse").reset_index(drop=True)
    per_output = pd.DataFrame(per_output_records)
    individual = per_output.loc[per_output["model"].isin(args.models)].copy()
    best_rows = individual.loc[individual.groupby("output")["normalized_mse"].idxmin()].copy()
    best_by_output = best_rows[[
        "output",
        "model",
        "normalized_mse",
        "normalized_rmse",
        "rmse_physical",
        "mae_physical",
        "r2",
    ]].sort_values("output")

    selection_frequency = (
        selected_families["selected_family"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="outer_splits_selected")
    )
    selection_frequency["selection_fraction"] = (
        selection_frequency["outer_splits_selected"] / len(selected_families)
    )

    overall.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    per_output.to_csv(output_dir / "per_output_model_metrics.csv", index=False)
    best_by_output.to_csv(output_dir / "best_model_by_output.csv", index=False)
    selection_frequency.to_csv(output_dir / "model_selection_frequency.csv", index=False)

    # Final deployment selection: repeated inner 3-sample validation on all rows.
    print("\nSelecting final deployment model using repeated inner validation on all data...")
    final_inner_splits = make_holdout_splits(
        len(frame), args.inner_holdout, args.inner_repeats, args.seed + 80_000_000
    )
    final_x_scaler = Standardizer.fit(x)
    final_y_scaler = Standardizer.fit(y)
    final_x_z = final_x_scaler.transform(x)
    final_y_z = final_y_scaler.transform(y)

    final_params: dict[str, dict[str, Any]] = {}
    final_refit_epochs: dict[str, Any] = {}
    final_validation_scores: dict[str, float] = {}
    classical_models: dict[str, Any] = {}
    fcnn_checkpoint: dict[str, Any] | None = None
    fcnn_single_output_checkpoint: dict[str, Any] | None = None

    for family_index, family in enumerate(args.models):
        family_seed = args.seed + 90_000_000 + family_index * 100_000
        # Final hyperparameter validation is also parallel on CPU when workers
        # are enabled. This avoids serial tiny-FCNN training on one GPU.
        final_tuning_device = (
            torch.device("cpu") if args.workers_resolved != 1 else device
        )
        best_params, best_score, _results, refit_epochs = tune_family(
            family,
            x,
            y,
            grids[family],
            final_inner_splits,
            family_seed,
            final_tuning_device,
            args.workers_resolved,
        )
        final_params[family] = best_params
        final_refit_epochs[family] = refit_epochs
        final_validation_scores[family] = best_score
        model = fit_refit_model(
            family,
            best_params,
            final_x_z,
            final_y_z,
            family_seed + 777,
            device,
            refit_epochs,
        )
        print(
            f"  {family:18s} validation MSE={best_score:.5g}; "
            f"params={jsonable_params(best_params)}"
            + (
                f"; refit_epochs={refit_epochs}"
                if family in {"fcnn", "fcnn_single_output"}
                else ""
            )
        )
        if family == "fcnn":
            model_cpu = model.cpu()
            fcnn_checkpoint = {
                "state_dict": model_cpu.state_dict(),
                "input_columns": args.inputs,
                "output_columns": args.outputs,
                "hidden_layers": list(best_params["hidden"]),
                "hyperparameters": jsonable_params(best_params),
                "refit_epochs": refit_epochs,
                "input_mean": final_x_scaler.mean,
                "input_std": final_x_scaler.scale,
                "output_mean": final_y_scaler.mean,
                "output_std": final_y_scaler.scale,
            }
        elif family == "fcnn_single_output":
            models_cpu = [m.cpu() for m in model]
            fcnn_single_output_checkpoint = {
                "state_dicts": [m.state_dict() for m in models_cpu],
                "input_columns": args.inputs,
                "output_columns": args.outputs,
                "hidden_layers": list(best_params["hidden"]),
                "hyperparameters": jsonable_params(best_params),
                "refit_epochs_by_output": list(refit_epochs),
                "input_mean": final_x_scaler.mean,
                "input_std": final_x_scaler.scale,
                "output_mean": final_y_scaler.mean,
                "output_std": final_y_scaler.scale,
                "architecture": "one independent all-input FCNN per output column",
            }
        else:
            classical_models[family] = model

    final_selected_family = min(final_validation_scores, key=final_validation_scores.get)
    print(f"Final validation-selected family: {final_selected_family}")

    artifact_bundle = {
        "input_columns": args.inputs,
        "output_columns": args.outputs,
        "input_mean": final_x_scaler.mean,
        "input_std": final_x_scaler.scale,
        "output_mean": final_y_scaler.mean,
        "output_std": final_y_scaler.scale,
        "models": classical_models,
        "hyperparameters": {
            k: jsonable_params(v) for k, v in final_params.items() if k != "fcnn"
        },
        "validation_scores": final_validation_scores,
        "selected_family": final_selected_family,
    }
    joblib.dump(artifact_bundle, output_dir / "final_classical_models.joblib")
    if fcnn_checkpoint is not None:
        torch.save(fcnn_checkpoint, output_dir / "final_fcnn.pt")
    if fcnn_single_output_checkpoint is not None:
        torch.save(
            fcnn_single_output_checkpoint,
            output_dir / "final_fcnn_single_output.pt",
        )

    (output_dir / "final_hyperparameters.json").write_text(
        json.dumps(
            {
                family: {
                    "params": jsonable_params(final_params[family]),
                    "validation_normalized_mse": final_validation_scores[family],
                    "fcnn_refit_epochs": final_refit_epochs[family],
                }
                for family in args.models
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "final_selected_model.json").write_text(
        json.dumps(
            {
                "selected_family": final_selected_family,
                "selection_metric": "mean normalized MSE over repeated inner validation holdouts",
                "validation_normalized_mse": final_validation_scores[final_selected_family],
                "params": jsonable_params(final_params[final_selected_family]),
                "fcnn_refit_epochs": final_refit_epochs[final_selected_family],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "final_preprocessing.json").write_text(
        json.dumps(
            {
                "input_scaler": final_x_scaler.as_dict(args.inputs),
                "output_scaler": final_y_scaler.as_dict(args.outputs),
                "note": "Deployment transforms are fit on all usable rows only after nested CV reporting is complete.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_empirical_report(
        output_dir / "model_report.md",
        args,
        overall,
        per_output,
        best_by_output,
        selection_frequency,
        final_params,
        final_refit_epochs,
        final_selected_family,
    )
    save_comparison_plot(overall, output_dir / "model_comparison_normalized_mse.png")

    run_summary = {
        "csv": str(args.csv),
        "usable_samples": int(len(frame)),
        "dropped_or_error_rows": dropped,
        "inputs": args.inputs,
        "outputs": args.outputs,
        "models": args.models,
        "outer_test_holdout_samples": args.holdout,
        "outer_repeats": args.cv_repeats,
        "inner_validation_holdout_samples": args.inner_holdout,
        "inner_repeats": args.inner_repeats,
        "search_profile": args.search_profile,
        "seed": args.seed,
        "device": str(device),
        "workers": ("all_cpu_cores" if args.workers_resolved == -1 else args.workers_resolved),
        "parallelization": (
            "outer split x model family jobs use one persistent process pool; "
            "multiprocess FCNN CV uses CPU and final refit may use CUDA"
        ),
        "runtime_seconds": float(time.time() - start_time),
        "normalization": "z-score fit only on training subset at every validation/test stage",
        "hyperparameter_selection_metric": "mean inner-validation normalized MSE across all six outputs",
        "model_family_selection": "lowest inner-validation normalized MSE",
        "fcnn_stopping": (
            "best inner-validation checkpoint with patience early stopping; "
            "multi-output FCNN uses one median best epoch, single-output FCNN uses "
            "one median best epoch per output network for refit"
        ),
        "final_selected_family": final_selected_family,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nOverall outer-test ranking:")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("\nValidation-selected family frequency:")
    print(selection_frequency.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")
    print(f"Report: {output_dir / 'model_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
