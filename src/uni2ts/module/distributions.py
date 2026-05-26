import torch

class QuantileKnotDistribution:
    """
    Batched quantile-knot distribution with linear interior interpolation
    and Gaussian or linear tail extrapolation.

    The fitted values can have any number of leading dimensions:
        values.shape == (..., num_quantiles)
    e.g. (B, T, K), (B, T, P, L, K), (K,) — anything with K as the last axis.

    `rvs` returns one sample per leading-axis location, so the output shape
    is exactly `values.shape[:-1]` — i.e. the batch shape with the quantile
    axis removed. This matches the v1 trajectory-sampling pattern where each
    (batch, timestep, ...) location draws an independent Monte Carlo value.

    Parameters
    ----------
    tails : {"gaussian", "linear"}, default "gaussian"

    Attributes set by `.fit`
    ------------------------
    quantiles_   : torch.tensor, shape (K,)         — shared knot positions
    values_      : torch.tensor, shape (..., K)     — sorted predicted values
    batch_shape_ : tuple                          — values_.shape[:-1]
    """

    def __init__(self, tails="gaussian"):
        if tails not in ("gaussian", "linear"):
            raise ValueError(f"tails must be 'gaussian' or 'linear', got {tails!r}")
        self.tails = tails

    # ---------- fit ----------

    def fit(self, quantiles, values):
        y = torch.as_tensor(values)
        q = torch.as_tensor(quantiles, dtype=y.dtype, device=y.device).reshape(-1)

        if q.shape[0] < 2:
            raise ValueError("need at least 2 knots")
        if y.shape[-1] != q.shape[0]:
            raise ValueError(
                f"values last dim ({y.shape[-1]}) must match num_quantiles ({q.shape[0]})"
            )
        if not torch.all((q > 0) & (q < 1)):
            raise ValueError("quantiles must lie strictly in (0, 1)")
        if not torch.all(torch.diff(q) > 0):
            raise ValueError("quantiles must be strictly increasing")

        # # Enforce per-element monotonicity along the K axis (handles
        # # quantile crossings; mirrors TOTOv2's `.sort(dim=0).values` step).
        # y = torch.sort(y, axis=-1)

        self.quantiles_   = q
        self.values_      = y
        self.batch_shape_ = y.shape[:-1]

        # Interior slopes along the last axis: shape (..., K-1)
        dq = torch.diff(q)                  # (K-1,)
        dy = torch.diff(y, axis=-1)         # (..., K-1)
        self._slopes_ = dy / dq          # (..., K-1)

        # Tail parameters — all batch-shaped
        if self.tails == "gaussian":
            z_lo = torch.special.ndtri(q[:2])
            z_hi = torch.special.ndtri(q[-2:])
            self._sigma_lo_ = (y[..., 1]  - y[..., 0])  / (z_lo[1] - z_lo[0])
            self._mu_lo_    = y[..., 0]  - self._sigma_lo_ * z_lo[0]
            self._sigma_hi_ = (y[..., -1] - y[..., -2]) / (z_hi[1] - z_hi[0])
            self._mu_hi_    = y[..., -1] - self._sigma_hi_ * z_hi[1]
        else:  # "linear"
            self._slope_lo_ = self._slopes_[..., 0]
            self._slope_hi_ = self._slopes_[..., -1]

        return self

    # ---------- ppf ----------

    def ppf(self, u):
        """
        Inverse CDF, vectorized over the batch.

        u : scalar or array, shape broadcastable with `batch_shape_`.
            For sampling, u typically has shape == batch_shape_.
            For e.g. `ppf(0.5)` (median per distribution), pass a scalar.

        Returns array of shape `np.broadcast_shapes(u.shape, batch_shape_)`.
        """
        self._check_fitted()
        u = torch.as_tensor(u, dtype=self.values_.dtype, device=self.values_.device)
        u_b = torch.broadcast_to(u, self.batch_shape_) if u.shape != self.batch_shape_ else u

        q = self.quantiles_
        K = q.shape[0]

        # Segment lookup (fully vectorized) and gather along last axis.
        idx     = torch.clip(torch.searchsorted(q, u_b, side="right") - 1, 0, K - 2)
        idx_e   = idx[..., None]
        y_left  = torch.take_along_dim(self.values_,  idx_e, axis=-1).squeeze(-1)
        slope   = torch.take_along_dim(self._slopes_, idx_e, axis=-1).squeeze(-1)
        q_left  = q[idx]
        interior = y_left + slope * (u_b - q_left)

        # Tails (computed everywhere, selected via where — cheap and
        # avoids mask-based gather complications).
        if self.tails == "gaussian":
            eps = 1e-12
            z = torch.special.ndtri(torch.clip(u_b, eps, 1 - eps))
            lower = self._mu_lo_ + self._sigma_lo_ * z
            upper = self._mu_hi_ + self._sigma_hi_ * z
        else:
            lower = self.values_[..., 0]  + self._slope_lo_ * (u_b - q[0])
            upper = self.values_[..., -1] + self._slope_hi_ * (u_b - q[-1])

        return torch.where(
            u_b < q[0],  lower,
            torch.where(u_b > q[-1], upper, interior),
        )

    # ---------- rvs ----------

    def rvs(self, random_state=42):
        """
        Draw one independent sample per fitted distribution.
        Returns array of shape `batch_shape_`.
        """
        self._check_fitted()
        device = self.values_.device
        gen = torch.Generator(device=device).manual_seed(random_state)
        u = torch.rand(
            self.batch_shape_,
            generator=gen,
            device=device,
            dtype=self.values_.dtype,
        )
        return self.ppf(u)

    # ---------- helpers ----------

    def _check_fitted(self):
        if not hasattr(self, "quantiles_"):
            raise RuntimeError("Distribution is not fitted. Call .fit(...) first.")

    def __repr__(self):
        if hasattr(self, "quantiles_"):
            return (f"QuantileKnotDistribution(tails={self.tails!r}, "
                    f"batch_shape={self.batch_shape_}, K={self.quantiles_.shape[0]})")
        return f"QuantileKnotDistribution(tails={self.tails!r}, <unfitted>)"
