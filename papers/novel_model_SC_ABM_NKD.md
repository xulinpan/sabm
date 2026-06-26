# Hierarchical Count-Based Spatio-Temporal ABM with Neural Dispersal Kernel and Multi-Source Emulated Inference

**Working title:** SC-ABM-NKD — *Structured Count Agent-Based Model with Neural Kernel Dispersal*

---

## 1. Motivation and Positioning

### 1.1 The Three-Paper Synthesis

This model synthesises and extends three bodies of work:

| Source | Core contribution | Limitation addressed here |
|---|---|---|
| Hooten & Wikle (2010, JASA) | Hierarchical Bayesian CA for binary spatio-temporal data; anisotropic nonstationary Dirichlet dispersal; PDE limit | Binary state only; perfect detection; independent propagation assumption; stationary-only density |
| Heard (2014, Duke PhD) | Likelihood-free inference via GP emulators and ABC-SMC; ABM validation and complexity theory | No spatial count ABM; emulator not adapted to structured spatial state |
| Phase 1/3 drafts (internal notes) | Count-based CA relaxing independence; density-dependent dispersal kernel | Parametric kernel (exponential); no multi-source data fusion; no continuous-time limit |

**Key novel contributions of SC-ABM-NKD:**

1. A **doubly-stochastic count state** with joint imperfect-detection observation models for heterogeneous data sources (count surveys + binary remote sensing).
2. A **neural dispersal kernel** that replaces the Dirichlet parameterisation with a deep network taking pairwise habitat features and density as input, enabling nonparametric, nonstationary, density-dependent dispersal — without independence assumptions.
3. **Density-regulated survival** (logistic correction to persistence) yielding a carrying capacity constraint within each cell.
4. A **nonlinear advection-diffusion-reaction (ADR) PDE** as the continuous-time, continuous-space mean-field limit — a strict generalisation of the H&W linear PDE.
5. **Neural-emulated ABC-SMC inference** that trains a Neural Process surrogate on pilot ABM runs, then substitutes it for the full simulator inside the SMC loop — making posterior inference tractable at scale.

---

## 2. Notation and Domain

| Symbol | Meaning |
|---|---|
| $\mathcal{S} = \{1,\ldots,m\}$ | Discrete set of spatial cells (areal units) |
| $\mathcal{T} = \{1,\ldots,T\}$ | Discrete time steps |
| $N_{i,t} \in \mathbb{Z}_{\geq 0}$ | Latent true population count at cell $i$, time $t$ |
| $\mathcal{N}_i$ | Spatial neighbourhood of cell $i$ (Queen's, $d_\mathcal{N}=9$) |
| $d_{ji}$ | Euclidean distance between centroids of cells $j$ and $i$ |
| $\mathbf{x}_i \in \mathbb{R}^p$ | Habitat/environment covariates at cell $i$ |
| $\alpha_i \in \mathbb{R}$ | Latent habitat suitability at cell $i$ |
| $\phi_i \in (0,1)$ | Baseline persistence/survival probability at cell $i$ |
| $K_i > 0$ | Local carrying capacity at cell $i$ |
| $\psi_i \geq 0$ | Long-distance dispersal (LDD) rate into cell $i$ |
| $\rho_i \in (0,1]$ | Detection probability for count surveys at cell $i$ |
| $\kappa > 0$ | Remote-sensing detection rate parameter |
| $\theta_{NN}$ | Parameters of the neural dispersal kernel |

---

## 3. Observation Model (Layer 1)

We accommodate two heterogeneous data streams that are observed over potentially different subsets of the domain.

### 3.1 Count Survey Observations

Let $\mathcal{O}_C \subseteq \mathcal{S} \times \mathcal{T}$ index cells and times with count survey data (e.g., eBird checklists, field transects). For $(i,t) \in \mathcal{O}_C$:

$$Y_{i,t}^{(C)} \mid N_{i,t},\, \rho_i,\, r \;\sim\; \text{NegBin}\!\left(\mu_{i,t}^{(C)},\; r\right), \qquad \mu_{i,t}^{(C)} = \rho_i \, N_{i,t} \tag{1}$$

with probability mass function

$$\Pr\!\left(Y_{i,t}^{(C)} = y\right) = \binom{y+r-1}{y} \left(\frac{r}{r + \mu_{i,t}^{(C)}}\right)^{\!r} \left(\frac{\mu_{i,t}^{(C)}}{r+\mu_{i,t}^{(C)}}\right)^{\!y} \tag{2}$$

$r > 0$ controls overdispersion; $r \to \infty$ recovers the Poisson. This generalises the Poisson thinning model of Phase 1 to handle real-count variability beyond the Poisson mean-variance equality.

When survey effort data $\mathbf{e}_{i,t}$ (duration, observers, distance) are available (as in eBird), extend the detection model:

$$\text{logit}(\rho_{i,t}) = \mu_\rho + \boldsymbol{\delta}'\mathbf{e}_{i,t} + v_i, \qquad v_i \stackrel{iid}{\sim} \mathcal{N}(0,\sigma_\rho^2) \tag{3}$$

### 3.2 Binary Remote Sensing Observations

Let $\mathcal{O}_P \subseteq \mathcal{S} \times \mathcal{T}$ index cells with binary presence/absence from remote sensing (e.g., VIIRS active fire detection, satellite occupancy). For $(i,t) \in \mathcal{O}_P$:

$$Y_{i,t}^{(P)} \mid N_{i,t},\, \kappa \;\sim\; \text{Bernoulli}\!\left(1 - e^{-\kappa N_{i,t}}\right) \tag{4}$$

This is the **zero-truncated complement** of a $\text{Poisson}(\kappa N_{i,t})$ model: the probability of detecting at least one individual (or signal) when $N_{i,t}$ individuals are present and each is independently detected with infinitesimal probability in a Poisson process with total rate $\kappa N_{i,t}$.

*Key properties:*
- When $N_{i,t} = 0$: $\Pr(Y_{i,t}^{(P)}=1) = 0$ (no false positives)
- When $N_{i,t} \to \infty$: $\Pr(Y_{i,t}^{(P)}=1) \to 1$ (saturating)
- Reduces exactly to Hooten & Wikle's $y_{i,t} \mid \theta_{i,t} \sim \text{Bern}(\theta_{i,t})$ when $\kappa N_{i,t} = \theta_{i,t}$ (linking binary probability to latent count)

### 3.3 Joint Observation Likelihood

$$\mathcal{L}\!\left(\mathbf{Y}^{(C)}, \mathbf{Y}^{(P)} \mid \mathbf{N}, \boldsymbol{\rho}, r, \kappa\right) = \prod_{(i,t)\in\mathcal{O}_C} \text{NegBin}\!\left(Y_{i,t}^{(C)};\, \rho_{i,t} N_{i,t},\, r\right) \cdot \prod_{(i,t)\in\mathcal{O}_P} \text{Bern}\!\left(Y_{i,t}^{(P)};\, 1-e^{-\kappa N_{i,t}}\right) \tag{5}$$

The two data streams are conditionally independent given $\mathbf{N}$, which is the latent true process.

---

## 4. Process Model: Latent Population Dynamics (Layer 2)

The latent count at cell $i$, time $t$ decomposes into three additive mechanisms:

$$N_{i,t} = S_{i,t} + I_{i,t} + L_{i,t} \tag{6}$$

where $S_{i,t}$, $I_{i,t}$, and $L_{i,t}$ are mutually independent given $\{N_{j,t-1}\}_{j \in \mathcal{S}}$.

### 4.1 Density-Regulated Persistence $S_{i,t}$

Individuals at cell $i$ survive from time $t-1$ to $t$ with probability that declines as density increases, capturing intra-specific competition:

$$S_{i,t} \mid N_{i,t-1},\, \phi_i,\, K_i \;\sim\; \text{Binomial}\!\left(N_{i,t-1},\; p_{s,i,t-1}\right) \tag{7}$$

$$p_{s,i,t-1} = \phi_i \exp\!\left(-\frac{N_{i,t-1}}{K_i}\right) \tag{8}$$

*Interpretation:*
- When $N_{i,t-1} \ll K_i$: $p_{s,i,t-1} \approx \phi_i$ (near-constant survival, recovering H&W's $\phi$)
- When $N_{i,t-1} = K_i$: $p_{s,i,t-1} = \phi_i / e \approx 0.368\,\phi_i$
- $K_i \to \infty$ collapses to the binary H&W persistence parameter

Conditional mean of $S_{i,t}$:

$$\mathbb{E}[S_{i,t} \mid N_{i,t-1}] = N_{i,t-1} \phi_i \exp\!\left(-\frac{N_{i,t-1}}{K_i}\right) \tag{9}$$

This is the discrete analogue of the logistic growth term $\phi N(1 - N/K)$ that appears in reaction-diffusion ecology models.

### 4.2 Neural Kernel Neighbourhood Immigration $I_{i,t}$

#### 4.2.1 The Neural Dispersal Kernel

Define the dispersal *intensity* from source cell $j$ to target cell $i$ as:

$$\lambda_{j \to i}(N_{j,t-1};\, \theta_{NN},\, \boldsymbol{\alpha}) = f_{\theta_{NN}}\!\left(\mathbf{z}_{ji}\right) \cdot g(N_{j,t-1};\, \gamma) \tag{10}$$

where $\mathbf{z}_{ji}$ is the pairwise feature vector and $g$ is a saturating density function.

**Pairwise feature vector:**

$$\mathbf{z}_{ji} = \left[\mathbf{x}_i - \mathbf{x}_j,\;\; d_{ji},\;\; \tilde{a}_{ji,1}, \ldots, \tilde{a}_{ji,d_\mathcal{N}}\right]' \in \mathbb{R}^{p+1+d_\mathcal{N}} \tag{11}$$

where $\tilde{a}_{ji,k} = (\alpha_{N_k,j} - \alpha_j)/d_{N_k,j}$ is the habitat suitability gradient in direction $k$ (as in H&W Eq. 8), evaluated at source cell $j$.

**Neural kernel architecture** (3-layer ReLU MLP, output in $(0,1)$):

$$h^{(1)} = \text{ReLU}(\mathbf{W}_1 \mathbf{z}_{ji} + \mathbf{b}_1) \in \mathbb{R}^{H_1}$$
$$h^{(2)} = \text{ReLU}(\mathbf{W}_2 h^{(1)} + \mathbf{b}_2) \in \mathbb{R}^{H_2}$$
$$f_{\theta_{NN}}(\mathbf{z}_{ji}) = \sigma(\mathbf{w}_3' h^{(2)} + b_3) \in (0,1) \tag{12}$$

where $\sigma(x) = (1+e^{-x})^{-1}$ is the sigmoid, and $\theta_{NN} = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{w}_3, b_3\}$.

**Density-scaling function** (saturating in source population):

$$g(N;\, \gamma) = \frac{N}{N + \gamma} \tag{13}$$

- $g(0;\gamma) = 0$: no dispersal from empty cells
- $g \to 1$ as $N \to \infty$: kernel contribution saturates (prevents explosive propagation)
- $\gamma > 0$ controls the half-saturation constant

*Connection to H&W:* Setting $f_{\theta_{NN}}(\mathbf{z}_{ji}) = p_{N_j \to i}$ (a constant directional probability from the Dirichlet model) and $g(N;\gamma) \to \mathbf{1}[N>0]$ recovers the binary H&W neighbourhood dispersal term.

#### 4.2.2 Immigration Count

The total immigration into cell $i$ from its neighbourhood at time $t$ is:

$$I_{i,t} \mid \{N_{j,t-1}\}_{j \in \mathcal{N}_i},\, \theta_{NN},\, \boldsymbol{\alpha} \;\sim\; \text{Poisson}\!\left(\Lambda_{i,t}\right) \tag{14}$$

$$\Lambda_{i,t} = \sum_{j \in \mathcal{N}_i} \lambda_{j \to i}(N_{j,t-1};\, \theta_{NN},\, \boldsymbol{\alpha}) \cdot N_{j,t-1} \tag{15}$$

**Why Poisson and not Binomial?** By the **Poisson superposition theorem**, if individuals from source $j$ arrive at cell $i$ as independent Poisson processes each with rate $\lambda_{j\to i} N_{j,t-1}$, then the total arrivals from all neighbours is $\text{Poisson}(\sum_j \lambda_{j\to i} N_{j,t-1})$.

This is precisely the step that **relaxes the H&W independence assumption** (their Eq. 5, p. 238): H&W required that propagules from different source cells act independently when computing $\tilde{p}_{i,t} = 1 - \prod_j (1-p_{j\to i})^{y_{j,t-1}}$. In the count model, the Poisson superposition handles multi-source inputs without requiring that interactions between sources be ignored — the total rate $\Lambda_{i,t}$ absorbs contributions from all occupied neighbours jointly, and the Poisson distribution correctly accounts for their additive effect.

### 4.3 Long-Distance Dispersal $L_{i,t}$

$$L_{i,t} \mid \psi_i \;\sim\; \text{Poisson}(\psi_i) \tag{16}$$

$$\log \psi_i = \mathbf{w}_i' \boldsymbol{\gamma} \tag{17}$$

where $\mathbf{w}_i$ includes covariates governing long-range connectivity (proximity to roads, rivers, migratory flyways, etc.). This matches H&W's $\psi$ parameter but allows spatial variation.

### 4.4 Full Process Distribution

Because $S_{i,t}$, $I_{i,t}$, $L_{i,t}$ are conditionally independent given $\{N_{j,t-1}\}$, the probability generating function (PGF) of $N_{i,t}$ factorises:

$$G_{N_{i,t}}(z) = G_{S_{i,t}}(z) \cdot G_{I_{i,t}}(z) \cdot G_{L_{i,t}}(z) \tag{18}$$

$$= \left(q_{s} + p_{s,i,t-1}\, z\right)^{N_{i,t-1}} \cdot \exp\!\left[\Lambda_{i,t}(z-1)\right] \cdot \exp\!\left[\psi_i(z-1)\right] \tag{19}$$

where $q_s = 1 - p_{s,i,t-1}$. Setting $z=1$ confirms $G_{N_{i,t}}(1)=1$. The conditional mean and variance are:

$$\mathbb{E}[N_{i,t} \mid \mathbf{N}_{t-1}] = N_{i,t-1}\, p_{s,i,t-1} + \Lambda_{i,t} + \psi_i \tag{20}$$

$$\text{Var}[N_{i,t} \mid \mathbf{N}_{t-1}] = N_{i,t-1}\, p_{s,i,t-1}\,(1-p_{s,i,t-1}) + \Lambda_{i,t} + \psi_i \tag{21}$$

The variance exceeds the mean when $N_{i,t-1}\, p_s (1-p_s) > 0$, i.e., whenever the binomial survival contributes variance — this overdispersion is correctly captured in the NegBin observation model (Layer 1).

---

## 5. Spatial Random Effects and Covariate Model (Layer 3)

### 5.1 Habitat Suitability Field

Following H&W Section 2.3, model latent suitability $\boldsymbol{\alpha} = [\alpha_1, \ldots, \alpha_m]'$ as a spatial Gaussian process:

$$\boldsymbol{\alpha} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}_m\!\left(\mathbf{0},\; \sigma_\alpha^2\, \mathbf{C}(\theta_\alpha)\right) \tag{22}$$

where $\mathbf{X} \in \mathbb{R}^{m \times p}$ is the environmental covariate matrix, $\boldsymbol{\beta} \in \mathbb{R}^p$ the regression coefficients, and

$$\mathbf{C}(\theta_\alpha)_{ij} = \exp\!\left(-\frac{\|s_i - s_j\|}{\theta_\alpha}\right) \tag{23}$$

is the exponential correlation function with range parameter $\theta_\alpha > 0$.

### 5.2 Carrying Capacity Model

Carrying capacities vary spatially according to:

$$\log K_i = \mathbf{x}_i' \boldsymbol{\beta}_K + u_i, \qquad u_i \stackrel{iid}{\sim} \mathcal{N}(0, \sigma_K^2) \tag{24}$$

### 5.3 Persistence Model

Baseline persistence probabilities follow a logistic-linear model:

$$\text{logit}(\phi_i) = \mathbf{x}_i' \boldsymbol{\beta}_\phi \tag{25}$$

---

## 6. Continuous-Time Mean-Field Limit: Nonlinear ADR PDE

A key theoretical result is that SC-ABM-NKD converges to a **nonlinear advection-diffusion-reaction PDE** in the continuum limit, extending H&W's linear PDE (their Eq. 14) to the density-regulated count setting.

### 6.1 Setup

Let $u(s, t) = \mathbb{E}[N_{s,t}]$ be the mean population density at location $s \in \mathbb{R}^2$ and continuous time $t$. Let the lattice spacing be $\Delta s$ and time step $\Delta t$. From Eq. (20), the mean-field recursion is:

$$u(s, t) = u(s, t-\Delta t)\, \phi(s) \exp\!\left(-\frac{u(s,t-\Delta t)}{K(s)}\right) + \int_{\mathcal{N}(s)} \lambda(s',s)\, u(s',t-\Delta t)\, ds' + \psi(s) \tag{26}$$

where $\lambda(s', s) = f_{\theta_{NN}}(\mathbf{z}_{s's}) \cdot g(u(s',t-\Delta t))$ is the continuum kernel.

### 6.2 Taylor Expansion of the Dispersal Term

Expand $u(s', t-\Delta t)$ around $s' = s$ for a local kernel supported on $\|s'-s\| \leq \Delta s$:

$$u(s', t-\Delta t) = u(s,t-\Delta t) + (s'-s)'\nabla u + \frac{1}{2}(s'-s)' \nabla^2 u\, (s'-s) + O(\Delta s^3)$$

Substituting into the dispersal integral and letting $\bar{K}(s) = \int_{\mathcal{N}(s)} \lambda(s',s) \, ds'$ (total dispersal probability from any occupied neighbour):

$$\int_{\mathcal{N}(s)} \lambda(s',s) u(s', t-\Delta t) \, ds' = \bar{K}(s) u(s,t-\Delta t) - \mathbf{v}(s)' \nabla u + \nabla \cdot \!\left(\mathbf{D}(s) \nabla u\right) + O(\Delta s^3)$$

where:

$$\mathbf{v}(s) = -\int_{\mathcal{N}(s)} (s'-s)\, \lambda(s', s)\, ds' \qquad \text{(drift velocity, } 2 \times 1 \text{)} \tag{27}$$

$$\mathbf{D}(s) = \frac{1}{2} \int_{\mathcal{N}(s)} (s'-s)(s'-s)'\, \lambda(s', s)\, ds' \qquad \text{(diffusion tensor, } 2\times 2 \text{)} \tag{28}$$

### 6.3 Taylor Expansion of the Persistence Term

Expand the density-regulated persistence for small $\Delta t$:

$$u(s,t-\Delta t) \phi(s) \exp\!\left(-\frac{u(s,t-\Delta t)}{K(s)}\right) \approx u(s,t) \phi(s) \left(1 - \frac{u(s,t)}{K(s)}\right) - \Delta t\, \phi(s) \frac{\partial u}{\partial t} + O(\Delta t^2) \tag{29}$$

### 6.4 The ADR PDE

Collecting Eqs. (26)–(29), rearranging, and taking $\Delta s, \Delta t \to 0$ with $\Delta s^2 / \Delta t = c$ (constant diffusive scaling):

$$\boxed{
\frac{\partial u}{\partial t} = \underbrace{r(s)\, u\!\left(1 - \frac{u}{K(s)}\right)}_{\textbf{(A) Logistic reaction}} - \underbrace{\nabla \cdot (\mathbf{v}(s)\, u)}_{\textbf{(B) Advection}} + \underbrace{\nabla \cdot (\mathbf{D}(s) \nabla u)}_{\textbf{(C) Anisotropic diffusion}} + \underbrace{\psi(s)}_{\textbf{(D) Source}}
} \tag{30}$$

where $r(s) = \phi(s) + \bar{K}(s) - 1$ is the net intrinsic growth rate.

**Term interpretations:**

- **(A) Logistic reaction**: density-regulated population growth with carrying capacity $K(s)$. Setting $K(s) \to \infty$ recovers linear growth; $K(s) = 0$ yields extinction dynamics.
- **(B) Advection**: directed movement of the population toward areas of higher habitat suitability, with velocity $\mathbf{v}(s)$ determined by the gradient of the neural kernel (and hence implicitly by $\nabla \alpha$).
- **(C) Anisotropic diffusion**: random dispersal with spatially varying, directionally asymmetric diffusion tensor $\mathbf{D}(s)$. When the neural kernel places unequal weight on different directions (habitat gradient), $\mathbf{D}$ is non-diagonal, encoding anisotropy. This generalises H&W's isotropic diffusion coefficient.
- **(D) Source**: long-distance colonisation rate $\psi(s)$.

**Reduction to H&W:** Setting $K(s) \to \infty$ and using a symmetric local kernel ($\mathbf{v}(s) = \mathbf{0}$, $\mathbf{D}(s) = D\mathbf{I}$) recovers the linear advection-diffusion equation of H&W Eq. (14) with drift $\delta_1, \delta_2$ and diffusion $D_1, D_2$. Thus Eq. (30) is a strict generalisation.

---

## 7. Full Hierarchical Model

Let $\Theta$ collect all model parameters:

$$\Theta = \left(\boldsymbol{\beta},\, \boldsymbol{\beta}_K,\, \boldsymbol{\beta}_\phi,\, \sigma_\alpha^2,\, \theta_\alpha,\, \sigma_K^2,\, \boldsymbol{\gamma},\, \mu_\rho,\, \sigma_\rho^2,\, \boldsymbol{\delta},\, r,\, \kappa,\, \gamma_{sat},\, \theta_{NN}\right) \tag{31}$$

The full joint posterior distribution is:

$$\left[\mathbf{N}_{1:T},\, \boldsymbol{\alpha},\, \boldsymbol{\rho},\, \Theta \mid \mathbf{Y}^{(C)},\, \mathbf{Y}^{(P)}\right] \propto \underbrace{\mathcal{L}\!\left(\mathbf{Y}^{(C)},\, \mathbf{Y}^{(P)} \mid \mathbf{N},\, \boldsymbol{\rho},\, r,\, \kappa\right)}_{\text{Observation layer}} \cdot \underbrace{\left[\mathbf{N}_{2:T} \mid \mathbf{N}_1,\, \Theta,\, \boldsymbol{\alpha}\right]}_{\text{Process layer}} \cdot \underbrace{\left[\boldsymbol{\alpha} \mid \boldsymbol{\beta},\, \sigma_\alpha^2,\, \theta_\alpha\right]}_{\text{Spatial GP}} \cdot \underbrace{\pi(\Theta)}_{\text{Priors}} \tag{32}$$

where the process layer factorises as:

$$\left[\mathbf{N}_{2:T} \mid \mathbf{N}_1,\, \Theta,\, \boldsymbol{\alpha}\right] = \prod_{t=2}^{T} \prod_{i=1}^{m} \left[N_{i,t} \mid \{N_{j,t-1}\}_{j \in \mathcal{S}},\, \Theta,\, \alpha_i\right] \tag{33}$$

and $[N_{i,t} | \cdot]$ is determined by the PGF in Eq. (19) with $p_{s,i,t-1}$ from Eq. (8) and $\Lambda_{i,t}$ from Eq. (15).

### 7.1 Prior Specifications

| Parameter | Prior | Justification |
|---|---|---|
| $\boldsymbol{\beta}$ | $\mathcal{N}_p(\mathbf{0},\, 10^2 \mathbf{I})$ | Weakly informative, habitat regression |
| $\sigma_\alpha^2$ | $\text{InvGamma}(2,\, 1)$ | Mode at 1, heavy right tail |
| $\theta_\alpha$ | $\text{Gamma}(2,\, 0.1)$ | Mean spatial range 20 units |
| $\boldsymbol{\beta}_K$ | $\mathcal{N}_q(\mathbf{0},\, 4\mathbf{I})$ | Carrying capacity regression |
| $\sigma_K^2$ | $\text{InvGamma}(2,\, 0.5)$ | Residual log-K variation |
| $\boldsymbol{\beta}_\phi$ | $\mathcal{N}_p(\mathbf{0},\, 1^2 \mathbf{I})$ | Logistic persistence (keep $\phi \in (0,1)$) |
| $\boldsymbol{\gamma}$ | $\mathcal{N}_q(\mathbf{0},\, 1^2 \mathbf{I})$ | LDD log-rate covariates |
| $\mu_\rho$ | $\mathcal{N}(0,\, 1)$ | Logit-scale detection intercept |
| $\sigma_\rho^2$ | $\text{InvGamma}(2,\, 1)$ | Site-level detection heterogeneity |
| $r$ | $\text{Gamma}(2,\, 0.5)$ | NegBin overdispersion; mean 4 |
| $\kappa$ | $\text{Gamma}(1,\, 1)$ | Remote sensing detection rate |
| $\gamma_{sat}$ | $\text{Gamma}(2,\, 0.2)$ | Half-saturation; mean 10 |
| $\theta_{NN}$ | $L_2$ regularisation with $\lambda_{reg}$ | Weight decay, MAP for neural params |

---

## 8. Inference: Neural-Emulated ABC-SMC

The likelihood $[\mathbf{Y}|\Theta]$ is intractable because it requires marginalising over $\mathbf{N}_{1:T}$ and $\boldsymbol{\alpha}$ — a high-dimensional sum/integral. Additionally, the neural kernel $\theta_{NN}$ makes the process non-conjugate. We use a **two-stage likelihood-free inference** approach extending Heard (2014) Chapters 3–4.

### 8.1 Summary Statistics

Define a vector of informative summary statistics $\mathbf{s}(\mathbf{Y}) = [s_1, \ldots, s_5]'$:

$$s_1 = \frac{1}{|\mathcal{O}_C|} \sum_{(i,t)\in\mathcal{O}_C} Y_{i,t}^{(C)} \qquad \text{(mean observed count)} \tag{34}$$

$$s_2 = \frac{1}{T-1} \sum_{t=2}^{T} \frac{\bar{Y}_t^{(C)} - \bar{Y}_{t-1}^{(C)}}{\bar{Y}_{t-1}^{(C)} + 1} \qquad \text{(relative temporal growth rate)} \tag{35}$$

$$s_3 = \frac{\sum_{(i,j) \in E} (Y_i^{(C)} - \bar{Y})(Y_j^{(C)} - \bar{Y})}{|E| \cdot \hat{\sigma}^2_Y} \qquad \text{(Moran's } I \text{, spatial autocorrelation)} \tag{36}$$

$$s_4 = \frac{1}{|\mathcal{O}_P|} \sum_{(i,t)\in\mathcal{O}_P} Y_{i,t}^{(P)} \qquad \text{(occupancy rate from remote sensing)} \tag{37}$$

$$s_5 = \frac{\text{Cov}(\bar{Y}_t^{(C)},\, \bar{Y}_{t-1}^{(C)})}{\text{Var}(\bar{Y}_t^{(C)})} \qquad \text{(temporal autocorrelation, lag-1)} \tag{38}$$

These capture: (i) abundance level, (ii) population trajectory, (iii) spatial clustering, (iv) occupancy complementary to counts, and (v) temporal persistence. Together they are informative for the key biological parameters $(\phi, K, \Lambda, \psi)$.

### 8.2 Neural Process (NP) Emulator

Following Heard (2014) Ch. 3, but replacing the Treed GP with a **Neural Process** (Garnelo et al. 2018), we train a surrogate that maps parameters to summary statistic distributions.

The NP defines a predictive distribution $q_\omega(\mathbf{s} | \Theta) \approx [\mathbf{s}|\Theta]$, where $\omega$ are learned weights. Architecture:

**Encoder** (context set from pilot simulations $\{(\Theta^{(k)}, \mathbf{s}^{(k)})\}_{k=1}^{K}$):

$$\mathbf{r}^{(k)} = h_\omega\!\left(\Theta^{(k)}, \mathbf{s}^{(k)}\right) \in \mathbb{R}^d, \qquad \bar{\mathbf{r}} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{r}^{(k)} \tag{39}$$

**Latent variable** (uncertainty in emulator):

$$\mathbf{z} \mid \bar{\mathbf{r}} \;\sim\; \mathcal{N}_d(\boldsymbol{\mu}_\omega(\bar{\mathbf{r}}),\; \text{diag}(\boldsymbol{\sigma}^2_\omega(\bar{\mathbf{r}}))) \tag{40}$$

**Decoder** (predictive distribution for new $\Theta^*$):

$$q_\omega(\mathbf{s} \mid \Theta^*, \mathbf{z}) = \mathcal{N}_5\!\left(m_\omega(\Theta^*, \mathbf{z}),\; \text{diag}(v_\omega(\Theta^*, \mathbf{z}))\right) \tag{41}$$

Training: maximise the ELBO over the pilot set.

### 8.3 ABC-SMC Algorithm with NP Emulator

The ABC-SMC scheme (Heard 2014, Sec. 4.3; Beaumont et al. 2009) proceeds through $L$ populations with decreasing tolerances $\epsilon_1 > \epsilon_2 > \cdots > \epsilon_L$.

**Distance metric:**

$$\rho(\mathbf{s}, \mathbf{s}') = \sum_{k=1}^{5} \frac{(s_k - s_k')^2}{\widehat{\text{Var}}_{\text{pilot}}(s_k)} \tag{42}$$

**Algorithm:**

```
Pilot phase (offline, run once):
  1. Draw K_0 = 2000 parameters from prior: Theta_k ~ pi(Theta)
  2. Run full SC-ABM-NKD for each k, compute s_k = s(Y_k)
  3. Train NP emulator q_omega on {(Theta_k, s_k)}: min ELBO loss
  4. Set epsilon_1 = 90th percentile of {rho(s_k, s_obs)}

ABC-SMC phase:
  Initialize: particle set P_1 = empty, weights W = {}
  
  For ell = 1, ..., L:
    accepted = 0
    While accepted < N_particles:
      If ell == 1:
        Theta* ~ pi(Theta)              [sample from prior]
      Else:
        Pick particle Theta_n from P_{ell-1} with prob W_n
        Theta* ~ K_ell(Theta | Theta_n) [perturb via transition kernel]
      
      Draw emulated summary: s_tilde ~ q_omega(. | Theta*)
      
      If rho(s_tilde, s_obs) <= epsilon_ell:
        Add Theta* to P_ell
        accepted += 1
    
    Compute importance weights:
      W_n = pi(Theta_n) / sum_{m} W_m K_ell(Theta_n | Theta_m)
    Normalise: W_n = W_n / sum(W)
    
    Set epsilon_{ell+1} = alpha-quantile of {rho(s_tilde_n, s_obs)}
    Update transition kernel K_{ell+1} from P_ell (Gaussian, bandwidth by Silverman)
  
  Return: {(Theta_n, W_n)}_{n=1}^N ~ [Theta | rho(s(Y_obs), s) <= epsilon_L]
```

where $\alpha \in (0.3, 0.5)$ (adaptive tolerance schedule).

**Computational advantage over Heard (2014) treed GP:** The NP emulator captures multi-modal and high-dimensional relationships between $\Theta$ and $\mathbf{s}$ without tree partitioning, and uncertainty in the emulator is propagated through the latent variable $\mathbf{z}$ — a form of *emulator-aware* ABC that avoids over-accepting under emulator uncertainty.

---

## 9. Model Identifiability and Validation

### 9.1 Identifiability

The joint observation model (count + binary) provides **complementary information** to identify the triple $(\rho_i, \kappa, N_{i,t})$:

- From count data alone: only $\rho_i N_{i,t}$ is identifiable, not $\rho_i$ or $N_{i,t}$ separately.
- Binary data provides $\kappa N_{i,t}$ through the Bernoulli logit, which under the log-linear link is independent of $\rho_i$.
- Together, the system identifies $N_{i,t}$ via:

$$\hat{N}_{i,t} \approx \frac{Y_{i,t}^{(C)}}{\hat{\rho}_i} \approx \frac{-\log(1 - \hat{p}_{i,t}^{(P)})}{\hat{\kappa}}$$

Equating these two expressions provides a moment equation that identifies $\kappa / \rho_i$, and repeated observations over time identify them separately.

Formal identifiability of the neural kernel requires regularisation (weight decay) and sufficient spatial replication — each direction of dispersal must have observed source-target pairs at multiple density levels. This is ensured when $|\mathcal{O}_C| \gg \dim(\theta_{NN})$.

### 9.2 Internal Validation (Heard Ch. 2)

Following Heard's internal validation protocol, we measure:

**Sensitivity** via Sobol' indices $S_k = \text{Var}_{\Theta_{-k}}[\mathbb{E}[s_j|\Theta_k]] / \text{Var}[s_j]$ for each summary statistic $s_j$ and each parameter $\Theta_k$.

**Model complexity** via the compression-ratio measure (Heard 2014, Sec. 2.4):

$$C = 1 - \frac{\text{length of compressed ABM output}}{\text{raw output length}}$$

with irrelevant rules identified by the criterion $R^2_{\text{adj}}(\gamma_0)$: a parameter $\Theta_k$ is irrelevant if removing it reduces $R^2$ by less than $\gamma_0 = 0.05$.

**Posterior predictive check:** Simulate $B = 500$ replications $\mathbf{Y}^{(rep)} \sim [\mathbf{Y}|\Theta^{(b)}]$ for $\Theta^{(b)}$ drawn from the ABC posterior, then compare $\mathbf{s}(\mathbf{Y}^{(rep)})$ to $\mathbf{s}(\mathbf{Y}^{obs})$ using the Kolmogorov-Smirnov test.

---

## 10. Connection to Dataset Context

This model is designed for **joint analysis of the VIIRS fire archive and eBird Basic Dataset** assembled in this research project:

| Dataset | Role in SC-ABM-NKD |
|---|---|
| `fire_viirs_2018.parquet` (19.2M VIIRS detections) | $\mathbf{Y}^{(P)}$ — binary remote sensing layer; FRP as proxy for population intensity $\kappa N_{i,t}$ |
| `ebd_observations.parquet` (eBird counts, Tuscaloosa AL) | $\mathbf{Y}^{(C)}$ — count survey layer with effort covariates for $\rho_{i,t}$ |
| `data/processed/ebd_checklist_summary.csv` | Effort variables $\mathbf{e}_{i,t}$ for detection model (Eq. 3) |
| `data/processed/ebd_species_summary.csv` | Species-level frequency for species-specific $\phi_i$, $K_i$ |
| MODIS TPW (when downloaded) | Environmental covariate $\mathbf{x}_i$ for habitat model (Eq. 22) — atmospheric moisture as bird habitat proxy |

The eBird + VIIRS combination instantiates the multi-source data fusion model directly. The fire FRP in the VIIRS data provides an analogue for population "intensity" detectable by remote sensing; eBird count surveys provide the discrete count layer. In an avian influenza application (Phase 3 notes), VIIRS fire would be replaced by VIIRS active disease signals, and eBird counts by bird abundance surveys — the mathematical structure is identical.

---

## 11. Summary of Mathematical Novelty

| Component | H&W (2010) | Heard (2014) | Phase 1 Draft | **SC-ABM-NKD (this model)** |
|---|---|---|---|---|
| State | Binary $y_{i,t} \in \{0,1\}$ | Any (emulated) | Count $N_{i,t}$ | **Count $N_{i,t}$** |
| Survival | Constant $\phi$ | — | Const. $\phi$ | **Density-regulated $\phi e^{-N/K}$** |
| Dispersal | Dirichlet Dir($\mathbf{a}_i$) | Black-box | Exponential kernel | **Neural kernel $f_{\theta_{NN}}(\mathbf{z}_{ji})$** |
| Density dependence | None | None | Saturating $g(N)$ | **Saturating $g(N)$ + logistic survival** |
| Independence assumption | Yes (products in Eq. 5) | — | Relaxed | **Relaxed (Poisson superposition, Eq. 14)** |
| Detection model | Perfect | Discrepancy $\delta$ | Poisson thinning | **NegBin count + Bernoulli binary (joint)** |
| PDE limit | Linear advection-diffusion (Eq. 14) | None | None | **Nonlinear ADR (Eq. 30) with logistic term** |
| Inference | Gibbs + MH MCMC | Treed GP + ABC | ABC-SMC | **Neural Process emulator + ABC-SMC** |
| Data sources | Binary only | Any | Count only | **Multi-source: count + binary** |
| Spatial heterogeneity | Via $\boldsymbol{\alpha}$ + Dir($\mathbf{a}_i$) | N/A | Implicit | **Via $\boldsymbol{\alpha}$ GP + neural kernel jointly** |

---

## 12. Notation Index

| Symbol | Defined | Meaning |
|---|---|---|
| $N_{i,t}$ | Eq. (6) | Latent count at cell $i$, time $t$ |
| $Y_{i,t}^{(C)}$ | Eq. (1) | Observed count (survey) |
| $Y_{i,t}^{(P)}$ | Eq. (4) | Observed binary (remote sensing) |
| $S_{i,t}$ | Eq. (7) | Surviving individuals |
| $I_{i,t}$ | Eq. (14) | Neighbourhood immigrants |
| $L_{i,t}$ | Eq. (16) | Long-distance dispersal arrivals |
| $p_{s,i,t}$ | Eq. (8) | Density-regulated survival probability |
| $\Lambda_{i,t}$ | Eq. (15) | Total neighbourhood dispersal rate |
| $\lambda_{j\to i}$ | Eq. (10) | Pairwise dispersal intensity |
| $f_{\theta_{NN}}$ | Eq. (12) | Neural dispersal kernel |
| $g(N;\gamma)$ | Eq. (13) | Density-scaling function |
| $\alpha_i$ | Eq. (22) | Latent habitat suitability |
| $\tilde{a}_{ji,k}$ | Eq. (11) | Directional habitat gradient |
| $\mathbf{v}(s)$ | Eq. (27) | Advection velocity (PDE) |
| $\mathbf{D}(s)$ | Eq. (28) | Diffusion tensor (PDE) |
| $G_{N_{i,t}}(z)$ | Eq. (18) | Probability generating function |
| $\rho_{i,t}$ | Eq. (3) | Detection probability (survey) |
| $\kappa$ | Eq. (4) | Remote sensing detection rate |
| $r$ | Eq. (1) | NegBin overdispersion |
| $\mathbf{s}(\mathbf{Y})$ | Eq. (34)–(38) | ABC summary statistics |
| $q_\omega(\cdot)$ | Eq. (41) | Neural Process emulator |
| $\epsilon_\ell$ | Sec. 8.3 | ABC tolerance at population $\ell$ |

---

*References: Hooten & Wikle (2010, JASA 105:236-248); Heard (2014, Duke PhD Dissertation); Garnelo et al. (2018, Neural Processes, arXiv:1807.01622); Beaumont et al. (2009, Biometrika); Turchin (1998, Quantitative Analysis of Movement).*
