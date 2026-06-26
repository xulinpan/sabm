"""
SC-ABM-NKD: Hierarchical Count-Based Spatio-Temporal Agent-Based Model
with Neural Dispersal Kernel and Multi-Source Emulated Inference

Implements:
  - Neural dispersal kernel (3-layer ReLU MLP, Eqs 10-12)
  - Density-regulated survival (Eq 4-5)
  - Poisson-superposition immigration (Eq 13)
  - Long-distance dispersal (Eq 14)
  - Doubly-stochastic observation model: NegBin counts + Bernoulli remote-sensing (Eqs 1-3)
  - Five ABC summary statistics (Eqs 19-23)
  - ABC-SMC with adaptive tolerance (Algorithm 1)

Reference: SC-ABM-NKD Research Memorandum (Hooten & Wikle 2010 / Heard 2014 synthesis)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import nbinom
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neural Dispersal Kernel  f_{θ_NN}(z_{ji})  →  (0, 1)
# ─────────────────────────────────────────────────────────────────────────────

class NeuralDispersalKernel(nn.Module):
    """
    3-layer ReLU MLP: z_dim → H1 → H2 → 1 (sigmoid).
    Maps pairwise habitat-feature vector z_{ji} to a dispersal weight in (0,1).
    Architecture follows Eqs (10)-(12).
    """

    def __init__(self, z_dim: int, H1: int = 32, H2: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, H1), nn.ReLU(),
            nn.Linear(H1,    H2), nn.ReLU(),
            nn.Linear(H2,     1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., z_dim)  →  (...,)  values in (0, 1)"""
        return self.net(z).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Spatial Grid
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGrid:
    """
    Flat-indexed 2-D grid of nrow × ncol cells.
    Cells indexed 0 … m-1 in row-major order.
    Queen's neighbourhood (≤ 8 neighbours per cell).
    """

    def __init__(self, nrow: int, ncol: int, cell_size: float = 1.0) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.m    = nrow * ncol
        self.cell_size = cell_size

        rows, cols = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
        self.coords: np.ndarray = np.column_stack(
            [rows.ravel() * cell_size, cols.ravel() * cell_size]
        ).astype(float)                         # (m, 2)

        self.dists: np.ndarray = cdist(self.coords, self.coords)   # (m, m)
        self.neighbours: List[np.ndarray] = self._queen_neighbours()
        self.max_nn: int = max(len(nb) for nb in self.neighbours)

    def _queen_neighbours(self) -> List[np.ndarray]:
        neighbours = []
        for idx in range(self.m):
            r, c = divmod(idx, self.ncol)
            nb = [
                (r + dr) * self.ncol + (c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
                and 0 <= r + dr < self.nrow
                and 0 <= c + dc < self.ncol
            ]
            neighbours.append(np.array(nb, dtype=int))
        return neighbours


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelParams:
    """
    All named parameters θ (Eq 24).
    beta, beta_K, beta_phi, gamma_ldd control regression means.
    Variance/range parameters govern GP spatial structure.
    """
    beta:       np.ndarray              # (p,)  habitat-GP mean regression
    beta_K:     np.ndarray              # (q,)  log-K regression
    beta_phi:   np.ndarray              # (p,)  logit-φ regression
    gamma_ldd:  np.ndarray              # (q,)  log-ψ regression

    sigma2_alpha: float = 1.0           # GP marginal variance
    theta_alpha:  float = 3.0           # GP range parameter
    sigma2_K:     float = 0.25          # residual log-K variance

    mu_rho:    float = 0.5              # logit-scale detection intercept
    sigma2_rho: float = 0.5             # site-level detection heterogeneity

    r:     float = 5.0                  # NegBin overdispersion
    kappa: float = 0.3                  # remote-sensing detection rate
    gamma: float = 8.0                  # half-saturation constant for g(N)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SC-ABM-NKD Simulator
# ─────────────────────────────────────────────────────────────────────────────

class SCABMNKD:
    """
    Full SC-ABM-NKD simulator.

    Population state at each step:
        N_{i,t} = S_{i,t} + I_{i,t} + L_{i,t}

    where:
        S_{i,t} ~ Binomial(N_{i,t-1}, φ_i · exp(−N_{i,t-1}/K_i))   [Eqs 4-5]
        I_{i,t} ~ Poisson(Λ_{i,t})                                   [Eq 13]
        L_{i,t} ~ Poisson(ψ_i)                                       [Eq 14]
    """

    def __init__(
        self,
        grid:   SpatialGrid,
        X:      np.ndarray,          # (m, p) environmental covariates
        W:      np.ndarray,          # (m, q) LDD / carrying-capacity covariates
        kernel: NeuralDispersalKernel,
        params: ModelParams,
    ) -> None:
        self.grid   = grid
        self.X      = X
        self.W      = W
        self.kernel = kernel
        self.params = params

        # Cell-level derived quantities (set by _derive_cell_params)
        self.alpha: np.ndarray  # (m,) latent habitat suitability
        self.phi:   np.ndarray  # (m,) baseline persistence probability
        self.K:     np.ndarray  # (m,) carrying capacity
        self.psi:   np.ndarray  # (m,) LDD rate
        self.rho:   np.ndarray  # (m,) count-survey detection probability

        self._derive_cell_params()
        self._precompute_features()

    # ── Parameter derivation ──────────────────────────────────────────────────

    def _derive_cell_params(self) -> None:
        """Sample / compute all cell-level parameters from θ."""
        p   = self.params
        m   = self.grid.m

        # Habitat suitability: Gaussian-process draw  (Eq 21)
        alpha_mu = self.X @ p.beta
        C = np.exp(-self.grid.dists / (p.theta_alpha + 1e-12))
        C += np.eye(m) * 1e-6           # numerical jitter
        L = np.linalg.cholesky(p.sigma2_alpha * C)
        self.alpha = alpha_mu + L @ np.random.randn(m)

        # Carrying capacity  log K_i = w_i'β_K + u_i  (Eq 22b)
        log_K = self.W @ p.beta_K + np.random.normal(
            0, np.sqrt(p.sigma2_K), size=m
        )
        self.K = np.exp(log_K)

        # Baseline persistence  logit(φ_i) = x_i'β_φ  (Eq 22c)
        self.phi = _sigmoid(self.X @ p.beta_phi)

        # LDD rate  log ψ_i = w_i'γ  (Eq 14)
        self.psi = np.exp(self.W @ p.gamma_ldd)

        # Detection probability  logit(ρ_i) = μ_ρ + v_i  (Eq 3)
        logit_rho = p.mu_rho + np.random.normal(
            0, np.sqrt(p.sigma2_rho), size=m
        )
        self.rho = _sigmoid(logit_rho)

    # ── Pre-compute pairwise feature tensors ──────────────────────────────────

    def _precompute_features(self) -> None:
        """
        For each target cell i and each neighbour j ∈ N_i, build:
            z_{ji} = [x_i − x_j,  d_{ji},  ã_{ji,1}, …, ã_{ji,d_N}]  (Eq 9)
        where ã_{ji,k} = (α_{N_k(j)} − α_j) / d_{N_k(j),j}  are
        directional habitat gradients of source cell j  (Eq 10).

        Stores tensors self._feat[i] of shape (|N_i|, z_dim).
        """
        p_dim = self.X.shape[1]
        z_dim = p_dim + 1 + self.grid.max_nn
        self._feat:   List[torch.Tensor] = []
        self._nb_idx: List[np.ndarray]   = []

        for i in range(self.grid.m):
            nbs   = self.grid.neighbours[i]
            n_nb  = len(nbs)
            feats = np.zeros((n_nb, z_dim), dtype=np.float32)

            for k, j in enumerate(nbs):
                feats[k, :p_dim]     = self.X[i] - self.X[j]          # habitat diff
                feats[k, p_dim]      = self.grid.dists[j, i]           # distance

                # Directional gradients of SOURCE cell j toward its own neighbours
                j_nbs = self.grid.neighbours[j]
                for k2, jn in enumerate(j_nbs[:self.grid.max_nn]):
                    d_jn = self.grid.dists[j, jn]
                    feats[k, p_dim + 1 + k2] = (
                        (self.alpha[jn] - self.alpha[j]) / d_jn if d_jn > 0 else 0.0
                    )

            self._feat.append(torch.from_numpy(feats))
            self._nb_idx.append(nbs)

    # ── Process model ─────────────────────────────────────────────────────────

    @staticmethod
    def _density_scale(N: np.ndarray, gamma: float) -> np.ndarray:
        """g(N; γ) = N / (N + γ)   (Eq 11)"""
        return N / (N + gamma + 1e-12)

    def step(self, N_prev: np.ndarray) -> np.ndarray:
        """
        Advance one time step.
        N_prev: (m,) integer counts at t-1.
        Returns N_curr: (m,) integer counts at t.
        """
        N_f   = N_prev.astype(float)
        gamma = self.params.gamma
        m     = self.grid.m

        # (A) Survival  S_i ~ Binomial(N_{i,t-1}, φ_i exp(−N/K_i))
        p_s = self.phi * np.exp(-N_f / (self.K + 1e-12))
        p_s = np.clip(p_s, 0.0, 1.0)
        S   = np.random.binomial(N_prev.astype(int), p_s)

        # (B) Neighbourhood immigration  I_i ~ Poisson(Λ_i)
        g      = self._density_scale(N_f, gamma)
        Lambda = np.zeros(m)
        with torch.no_grad():
            for i in range(m):
                nbs = self._nb_idx[i]
                if len(nbs) == 0:
                    continue
                w = self.kernel(self._feat[i]).numpy()          # (|N_i|,)
                # Λ_{i,t} = Σ_{j∈N_i} f_θ(z_{ji}) · g(N_j) · N_j  (Eq 13)
                Lambda[i] = float(np.dot(w, g[nbs] * N_f[nbs]))

        I = np.random.poisson(np.maximum(Lambda, 0.0))

        # (C) Long-distance dispersal  L_i ~ Poisson(ψ_i)
        L = np.random.poisson(self.psi)

        return (S + I + L).astype(int)

    def simulate(self, N0: np.ndarray, T: int) -> np.ndarray:
        """
        Simulate T time steps from initial counts N0.
        Returns trajectory of shape (T+1, m).
        """
        traj    = np.empty((T + 1, self.grid.m), dtype=int)
        traj[0] = N0
        for t in range(1, T + 1):
            traj[t] = self.step(traj[t - 1])
        return traj

    # ── Observation models ────────────────────────────────────────────────────

    def observe_count(
        self,
        N:   np.ndarray,
        rho: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Count-survey observations (Eq 1):
            Y^(C) | N, ρ, r  ~  NegBin(μ = ρN, r)
        """
        if rho is None:
            rho = self.rho
        mu  = rho * N.astype(float) + 1e-12
        r   = self.params.r
        p_n = r / (r + mu)
        return np.random.negative_binomial(r, p_n).astype(int)

    def observe_binary(self, N: np.ndarray) -> np.ndarray:
        """
        Binary remote-sensing observations (Eq 3):
            Y^(P) | N, κ  ~  Bernoulli(1 − exp(−κN))
        """
        prob = 1.0 - np.exp(-self.params.kappa * N.astype(float))
        return (np.random.uniform(size=N.shape) < prob).astype(int)

    # ── Log-likelihood (for given latent N) ───────────────────────────────────

    def log_likelihood(
        self,
        Y_C:   np.ndarray,   # observed counts at O_C locations
        Y_P:   np.ndarray,   # observed binary at O_P locations
        N_C:   np.ndarray,   # latent N at O_C locations
        N_P:   np.ndarray,   # latent N at O_P locations
        rho_C: np.ndarray,   # detection prob at O_C locations
    ) -> float:
        """Joint observation log-likelihood (Eq 6)."""
        r, kappa = self.params.r, self.params.kappa

        # NegBin log-pmf for count data
        mu_C = rho_C * N_C.astype(float) + 1e-12
        p_nb = r / (r + mu_C)
        ll_C = float(np.sum(nbinom.logpmf(Y_C, r, p_nb)))

        # Bernoulli log-pmf for binary data
        prob = np.clip(1.0 - np.exp(-kappa * N_P.astype(float)), 1e-12, 1 - 1e-12)
        ll_P = float(np.sum(
            Y_P * np.log(prob) + (1 - Y_P) * np.log(1 - prob)
        ))
        return ll_C + ll_P


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary Statistics  s(Y) = [s1, …, s5]   (Eqs 19-23)
# ─────────────────────────────────────────────────────────────────────────────

def compute_summaries(
    Y_C:  np.ndarray,   # (T+1, m) count observations  (NaN = unobserved)
    Y_P:  np.ndarray,   # (T+1, m) binary observations (NaN = unobserved)
    grid: SpatialGrid,
) -> np.ndarray:
    """
    Returns s = [s1, s2, s3, s4, s5].

    s1: mean observed count             (Eq 19)
    s2: relative temporal growth rate   (Eq 20)
    s3: Moran's I on last time step     (Eq 21)
    s4: remote-sensing occupancy rate   (Eq 22)
    s5: lag-1 temporal autocorrelation  (Eq 23)
    """
    T = Y_C.shape[0] - 1

    # s1 — mean observed count
    s1 = float(np.nanmean(Y_C))

    # s2 — relative temporal growth rate
    Y_bar = np.nanmean(Y_C, axis=1)   # (T+1,)
    growth = [
        (Y_bar[t] - Y_bar[t - 1]) / (Y_bar[t - 1] + 1.0)
        for t in range(1, T + 1)
        if not (np.isnan(Y_bar[t]) or np.isnan(Y_bar[t - 1]))
    ]
    s2 = float(np.mean(growth)) if growth else 0.0

    # s3 — Moran's I (spatial autocorrelation at last time step)
    y_last  = Y_C[-1]
    obs_idx = np.where(~np.isnan(y_last))[0]
    if len(obs_idx) > 1:
        y_obs  = y_last[obs_idx]
        y_mean = np.mean(y_obs)
        y_dev  = y_obs - y_mean
        sigma2 = np.var(y_obs) + 1e-12
        nb_set = {i: set(grid.neighbours[i]) for i in obs_idx}
        numer = sum(
            y_dev[a] * y_dev[b]
            for a, ia in enumerate(obs_idx)
            for b, ib in enumerate(obs_idx)
            if ib in nb_set[ia]
        )
        n_pairs = sum(1 for a, ia in enumerate(obs_idx)
                      for ib in obs_idx if ib in nb_set[ia])
        s3 = float(numer / (n_pairs * sigma2)) if n_pairs > 0 else 0.0
    else:
        s3 = 0.0

    # s4 — remote-sensing occupancy rate
    s4 = float(np.nanmean(Y_P))

    # s5 — lag-1 temporal autocorrelation of mean counts
    valid = ~np.isnan(Y_bar)
    if valid.sum() > 2:
        yt   = Y_bar[1:][valid[1:] & valid[:-1]]
        ytm1 = Y_bar[:-1][valid[1:] & valid[:-1]]
        if len(yt) > 1:
            cov_ = float(np.cov(yt, ytm1)[0, 1])
            var_ = float(np.var(yt)) + 1e-12
            s5   = cov_ / var_
        else:
            s5 = 0.0
    else:
        s5 = 0.0

    return np.array([s1, s2, s3, s4, s5], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Prior sampling and evaluation   (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def sample_prior(rng: np.random.Generator) -> Dict[str, float]:
    """Draw one scalar-parameter set from the prior (Table 2)."""
    r = rng.gamma(2.0, 1 / 0.5)         # Gamma(2, 0.5)  → mean 4
    while r <= 0:                         # guard
        r = rng.gamma(2.0, 1 / 0.5)
    kappa = rng.gamma(1.0, 1.0)          # Gamma(1, 1)    → mean 1
    gamma = rng.gamma(2.0, 1 / 0.2)     # Gamma(2, 0.2)  → mean 10

    return dict(
        phi_logit = rng.normal(0.0, 1.0),     # logit(φ) intercept
        log_K     = rng.normal(3.0, 1.0),     # log(K)    intercept
        log_psi   = rng.normal(-2.0, 1.0),    # log(ψ)    intercept
        mu_rho    = rng.normal(0.0, 1.0),     # logit(ρ)  intercept
        r         = r,
        kappa     = kappa,
        gamma     = gamma,
    )


def log_prior(theta: Dict[str, float]) -> float:
    """Log-prior density (up to additive constant)."""
    lp = 0.0
    # Normal priors on logit/log scale
    lp -= 0.5 * theta["phi_logit"] ** 2
    lp -= 0.5 * (theta["log_K"] - 3.0) ** 2
    lp -= 0.5 * (theta["log_psi"] + 2.0) ** 2
    lp -= 0.5 * theta["mu_rho"] ** 2
    # Gamma priors: log p(x) ∝ (a-1)ln(x) - b·x
    r, kappa, gamma = theta["r"], theta["kappa"], theta["gamma"]
    if r <= 0 or kappa <= 0 or gamma <= 0:
        return -np.inf
    lp += 1.0 * np.log(r)     - 2.0 * r       # Gamma(2, 0.5):  a-1=1, b=2
    lp += 0.0 * np.log(kappa) - 1.0 * kappa   # Gamma(1, 1):    a-1=0, b=1
    lp += 1.0 * np.log(gamma) - 5.0 * gamma   # Gamma(2, 0.2):  a-1=1, b=5
    return lp


def _gauss_kern(x: float, bw: float) -> float:
    return np.exp(-0.5 * (x / (bw + 1e-15)) ** 2) / (bw * np.sqrt(2 * np.pi) + 1e-15)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ABC-SMC   (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def abc_distance(
    s_sim:     np.ndarray,
    s_obs:     np.ndarray,
    var_pilot: np.ndarray,
) -> float:
    """Weighted squared Euclidean distance (Eq 18)."""
    return float(np.sum((s_sim - s_obs) ** 2 / (var_pilot + 1e-12)))


def abc_smc(
    s_obs:        np.ndarray,           # (5,) observed summary statistics
    run_sim:      Callable,             # theta_dict → np.ndarray | None
    n_particles:  int   = 300,
    n_populations: int  = 5,
    alpha_q:      float = 0.40,         # quantile for adaptive tolerance
    n_pilot:      int   = 300,          # pilot draws to estimate variance
    rng:          Optional[np.random.Generator] = None,
    verbose:      bool  = True,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Neural-emulated ABC-SMC (Algorithm 1; here uses full simulator).
    Returns (particles, weights, var_pilot).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # ── Pilot: estimate summary-statistic variances ──────────────────────────
    if verbose:
        print(f"Pilot phase: {n_pilot} prior draws …")
    pilot_s = []
    for _ in range(n_pilot):
        theta = sample_prior(rng)
        s = run_sim(theta)
        if s is not None and np.all(np.isfinite(s)):
            pilot_s.append(s)
    if len(pilot_s) < 10:
        raise RuntimeError("Too few valid pilot simulations; check run_sim.")
    pilot_arr  = np.array(pilot_s)
    var_pilot  = np.var(pilot_arr, axis=0) + 1e-12

    # ── Population 1: accept from prior ──────────────────────────────────────
    if verbose:
        print(f"Population 1/{n_populations} (sampling from prior) …")
    particles: List[Dict] = []
    all_dists: List[float] = []

    while len(particles) < n_particles:
        theta = sample_prior(rng)
        s = run_sim(theta)
        if s is None or not np.all(np.isfinite(s)):
            continue
        d = abc_distance(s, s_obs, var_pilot)
        particles.append(theta)
        all_dists.append(d)

    dists   = np.array(all_dists)
    epsilon = float(np.quantile(dists, alpha_q))
    weights = np.ones(n_particles) / n_particles
    if verbose:
        print(f"  ε₁ = {epsilon:.4f}")

    # ── Populations 2 … L ────────────────────────────────────────────────────
    keys = list(particles[0].keys())

    for ell in range(2, n_populations + 1):
        if verbose:
            print(f"Population {ell}/{n_populations}  ε = {epsilon:.4f} …")

        vals = {k: np.array([p[k] for p in particles]) for k in keys}
        bw   = {k: 1.06 * np.std(vals[k]) * n_particles ** (-0.2) + 1e-8
                for k in keys}

        new_particles: List[Dict] = []
        new_dists:     List[float] = []

        while len(new_particles) < n_particles:
            # Perturb a particle sampled from current distribution
            idx      = rng.choice(n_particles, p=weights)
            theta_n  = particles[idx]
            theta_star = {k: theta_n[k] + rng.normal(0.0, bw[k]) for k in keys}

            if np.isinf(log_prior(theta_star)):
                continue

            s = run_sim(theta_star)
            if s is None or not np.all(np.isfinite(s)):
                continue
            d = abc_distance(s, s_obs, var_pilot)
            if d <= epsilon:
                new_particles.append(theta_star)
                new_dists.append(d)

        # Importance weights  w_n ∝ π(θ*) / Σ_m w_m K(θ*|θ_m)
        new_w = np.zeros(n_particles)
        for n_idx, th in enumerate(new_particles):
            kern_sum = sum(
                weights[m_idx] * np.prod(
                    [_gauss_kern(th[k] - particles[m_idx][k], bw[k]) for k in keys]
                )
                for m_idx in range(n_particles)
            )
            new_w[n_idx] = np.exp(log_prior(th)) / (kern_sum + 1e-300)

        weights   = new_w / (new_w.sum() + 1e-300)
        particles = new_particles
        dists     = np.array(new_dists)
        epsilon   = float(np.quantile(dists, alpha_q))

    return particles, weights, var_pilot


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_model(
    theta:  Dict[str, float],
    grid:   SpatialGrid,
    X:      np.ndarray,
    W:      np.ndarray,
    z_dim:  int,
    kernel: Optional[NeuralDispersalKernel] = None,
) -> SCABMNKD:
    """
    Construct a SCABMNKD instance from a flat ABC parameter dict.
    Uses homogeneous (scalar) φ, K, ψ, ρ across cells for ABC loop.
    """
    p, q = X.shape[1], W.shape[1]

    params = ModelParams(
        beta       = np.zeros(p),
        beta_K     = np.zeros(q),
        beta_phi   = np.zeros(p),
        gamma_ldd  = np.zeros(q),
        sigma2_alpha = 0.5,
        theta_alpha  = 2.0,
        sigma2_K     = 0.01,     # near-zero: K from intercept only
        r     = max(theta["r"],     0.01),
        kappa = max(theta["kappa"], 1e-4),
        gamma = max(theta["gamma"], 1e-4),
        mu_rho = theta["mu_rho"],
        sigma2_rho = 0.01,       # near-zero: ρ from intercept only
    )
    if kernel is None:
        kernel = NeuralDispersalKernel(z_dim)

    model = SCABMNKD(grid, X, W, kernel, params)

    # Override with homogeneous values from theta
    m = grid.m
    model.phi  = np.full(m, _sigmoid(theta["phi_logit"]))
    model.K    = np.full(m, max(np.exp(theta["log_K"]),  1.0))
    model.psi  = np.full(m, max(np.exp(theta["log_psi"]), 1e-6))
    model.rho  = np.full(m, _sigmoid(theta["mu_rho"]))
    model._precompute_features()   # refresh feature tensors with updated alpha
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)

    # Grid and covariates
    NROW, NCOL = 8, 8
    T          = 10
    grid       = SpatialGrid(NROW, NCOL, cell_size=1.0)
    m          = grid.m      # 64 cells
    p          = 2           # habitat covariate dimension
    q          = 2           # LDD/K covariate dimension

    np.random.seed(0)
    X = rng.standard_normal((m, p))
    W = rng.standard_normal((m, q))

    # True parameter values
    z_dim = p + 1 + grid.max_nn   # feature dimension for neural kernel
    true_params = ModelParams(
        beta       = np.array([0.4, -0.2]),
        beta_K     = np.array([3.0,  0.2]),
        beta_phi   = np.array([0.5, -0.1]),
        gamma_ldd  = np.array([-2.0, 0.0]),
        r          = 5.0,
        kappa      = 0.30,
        gamma      = 8.0,
        mu_rho     = 0.5,
        sigma2_rho = 0.30,
    )
    true_kernel = NeuralDispersalKernel(z_dim)
    true_model  = SCABMNKD(grid, X, W, true_kernel, true_params)

    # Simulate ground-truth trajectory
    N0   = np.random.poisson(10, size=m)
    traj = true_model.simulate(N0, T)                       # (T+1, m)
    print(f"Ground-truth trajectory: mean N = {traj.mean():.2f}, max N = {traj.max()}")

    # Generate observations at all cells and all times
    Y_C = np.stack([true_model.observe_count(traj[t])  for t in range(T + 1)]).astype(float)
    Y_P = np.stack([true_model.observe_binary(traj[t]) for t in range(T + 1)]).astype(float)

    # Observed summary statistics
    s_obs = compute_summaries(Y_C, Y_P, grid)
    labels = ["mean count", "growth rate", "Moran's I", "occupancy", "lag-1 AC"]
    print("\nObserved summary statistics:")
    for lb, sv in zip(labels, s_obs):
        print(f"  {lb:16s}: {sv:8.4f}")

    # ABC-SMC
    def run_sim(theta: Dict) -> Optional[np.ndarray]:
        try:
            mod = build_model(theta, grid, X, W, z_dim)
            N0_s  = np.random.poisson(10, size=m)
            traj_s = mod.simulate(N0_s, T)
            YC_s = np.stack([mod.observe_count(traj_s[t])  for t in range(T + 1)]).astype(float)
            YP_s = np.stack([mod.observe_binary(traj_s[t]) for t in range(T + 1)]).astype(float)
            return compute_summaries(YC_s, YP_s, grid)
        except Exception:
            return None

    print("\nRunning ABC-SMC (demo: 100 particles × 3 populations) …")
    particles, weights, var_pilot = abc_smc(
        s_obs,
        run_sim,
        n_particles    = 100,
        n_populations  = 3,
        alpha_q        = 0.50,
        n_pilot        = 200,
        rng            = rng,
    )

    print("\nPosterior summary (weighted mean ± std):")
    print(f"  {'parameter':16s}  {'mean':>8s}  {'std':>8s}")
    for k in particles[0].keys():
        vals  = np.array([pp[k] for pp in particles])
        wmean = float(np.sum(weights * vals))
        wstd  = float(np.sqrt(np.sum(weights * (vals - wmean) ** 2)))
        print(f"  {k:16s}  {wmean:8.3f}  {wstd:8.3f}")


if __name__ == "__main__":
    run_demo()
