"""
train_3d.py — Stratos PINN v3
==============================
3D Transient Heat Conduction through a Cylindrical TPS Core Sample.
Implemented with NVIDIA PhysicsNeMo (physicsnemo.sym) for Google Colab T4 GPU.

Physics
-------
PDE:   dT/dt = alpha * (d²T/dx² + d²T/dy² + d²T/dz²)
alpha  = 5e-6 m²/s  (thermal diffusivity of TPS material)
t      in [0, 60] s

Geometry
--------
Cylinder: radius=0.05 m, height=0.10 m, axis along z.
  z=0      — bottom face (plasma-facing hot side)
  z=0.10   — top face    (cabin-facing insulated side)

Boundary Conditions
-------------------
  z=0,    t>0  : T = 4000 K                [Dirichlet — plasma temperature]
  z=0.10, t≥0  : dT/dz = 0                [Neumann   — insulated cabin wall]
  r=0.05, t≥0  : dT/dn = 0                [Neumann   — adiabatic outer wall]
  t=0,    ∀x,z : T = 300 K                [Initial Condition]

Key Design Decisions
--------------------
  Normalization : All inputs (x,y,z,t) and the output (T) are min-max scaled
                  to [0,1] before touching the network.  The PDE is rederived
                  in normalized coordinates via the chain rule, yielding a
                  single dimensionless coefficient alpha_hat = 0.030.  This
                  prevents exploding gradients from the large temperature range
                  (300-4000 K) and the space/time scale mismatch.

  Architecture  : FourierNetArch — Fourier-feature encoding + SiLU fully-
                  connected backbone.  Fourier features are essential here
                  because plain MLPs exhibit spectral bias and are slow to
                  learn the steep thermal gradient concentrated near z=0.

  OOM Guard     : Interior batch_size=2000, boundary batch_size=1000.  These
                  are the empirically safe ceilings for the 16 GB T4 GPU when
                  computing second-order autodiff through a 6-layer network.

Environment
-----------
  Requires the Colab setup cell to have been run first:
    - nvidia-modulus installed with --no-deps
    - physicsnemo symlinked from the cloned modulus-sym repo
"""

import os
import sys
import glob
import numpy as np
import torch

# ---------------------------------------------------------------------------
# numpy compatibility shim
# ---------------------------------------------------------------------------
# numpoly (pulled in by chaospy, used for geometry point sampling) calls
# numpy.core.multiarray.set_legacy_print_mode, which was removed in numpy
# 1.25.  We patch it back as a no-op before any physicsnemo import.
#
# numpy 2.x renamed numpy.core → numpy._core (numpy.core still exists but
# emits DeprecationWarning).  We probe the correct location to avoid noise.
def _noop(*args, **kwargs):
    pass

try:
    _marray = np._core.multiarray          # numpy 2.x path (no deprecation)
except AttributeError:
    _marray = np.core.multiarray           # numpy 1.x path

if not hasattr(_marray, "set_legacy_print_mode"):
    _marray.set_legacy_print_mode = _noop

# Also patch the legacy np.core path, but ONLY on numpy 1.x.
# On numpy 2.x, merely accessing np.core.multiarray emits a DeprecationWarning,
# and numpoly on numpy 2.x will reach np._core.multiarray (already patched above).
if int(np.__version__.split(".")[0]) < 2:
    try:
        if not hasattr(np.core.multiarray, "set_legacy_print_mode"):
            np.core.multiarray.set_legacy_print_mode = _noop
    except AttributeError:
        pass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, open_dict

# PhysicsNeMo is the renamed successor to NVIDIA modulus-sym.
# All modulus.sym.* imports become physicsnemo.sym.*
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Cylinder
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint,
    PointwiseBoundaryConstraint,
)
from physicsnemo.sym.node import Node
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.models.arch import Arch   # base class for all PhysicsNeMo models

# Config classes — pulled directly from the physicsnemo structured-config hierarchy
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.hydra.training import DefaultTraining, DefaultStopCriterion
from physicsnemo.sym.hydra.optimizer import AdamConf
from physicsnemo.sym.hydra.scheduler import ExponentialLRConf

SimConfig = PhysicsNeMoConfig

from sympy import Symbol, Function, diff


# =============================================================================
# NETWORK ARCHITECTURE  (pure PyTorch — no physicsnemo.nn dependency)
# =============================================================================
# physicsnemo.nn (FCLayer etc.) lives in the NVIDIA/modulus CORE repo which is
# a separate package from nvidia-modulus (the PINN branch).  It is not
# pip-installable in Python 3.12 / Colab.  We implement the equivalent
# Fourier-feature network directly in PyTorch so the rest of the PINN
# infrastructure (Node, Solver, Domain, constraints) is unaffected.

class FourierFeatureNet(Arch):
    """
    Fourier-feature encoding + SiLU fully-connected backbone.

    Inherits from Arch (physicsnemo.sym.models.arch.Arch) so that the
    PhysicsNeMo Solver recognises this module as a trainable network and
    correctly registers its parameters with the Adam optimizer.  A plain
    torch.nn.Module is silently excluded from the parameter search.

    Architecture
    ____________
    1.  Random Fourier projection: x_enc = [sin(Bx), cos(Bx)]  (fixed, not learned)
        B ~ N(0, freq_scale²),  shape (n_inputs, n_frequencies)
    2.  nr_layers × (Linear → SiLU) with weight normalization
    3.  Final linear projection to n_outputs (no activation)

    Input / output convention (matches PhysicsNeMo Node evaluate signature):
        forward(in_vars: dict[str, Tensor(batch,1)]) → dict[str, Tensor(batch,1)]
    """

    def __init__(
        self,
        input_keys:  list,           # list of Key objects
        output_keys: list,           # list of Key objects
        layer_size:  int   = 256,
        nr_layers:   int   = 6,
        n_frequencies: int = 8,      # number of Fourier frequencies
        freq_scale:  float = 1.0,
    ):
        # Arch.__init__ stores input_keys / output_keys and calls
        # torch.nn.Module.__init__ — must come before any nn.Module ops.
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self._in_keys  = [k.name for k in input_keys]
        self._out_keys = [k.name for k in output_keys]
        n_in  = len(input_keys)
        n_out = len(output_keys)

        # Fixed random Fourier projection matrix
        B = torch.randn(n_in, n_frequencies) * freq_scale
        self.register_buffer("_B", B)

        # After encoding: sin + cos → 2 * n_frequencies features
        enc_dim = 2 * n_frequencies

        layers: list[torch.nn.Module] = []
        in_dim = enc_dim
        for _ in range(nr_layers):
            lin = torch.nn.Linear(in_dim, layer_size)
            lin = torch.nn.utils.weight_norm(lin)
            layers += [lin, torch.nn.SiLU()]
            in_dim = layer_size
        layers.append(torch.nn.Linear(layer_size, n_out))

        self.net = torch.nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, in_vars: dict) -> dict:
        x = torch.cat([in_vars[k] for k in self._in_keys], dim=-1)  # (B, n_in)
        x_proj = x @ self._B                                         # (B, n_freq)
        x_enc  = torch.cat([torch.sin(x_proj),
                             torch.cos(x_proj)], dim=-1)             # (B, 2*n_freq)
        out = self.net(x_enc)                                         # (B, n_out)
        return {k: out[:, i : i + 1]
                for i, k in enumerate(self._out_keys)}

    # make_node() is inherited from Arch.  Arch.make_node(name, jit, optimize)
    # creates Node(self.input_keys, self.output_keys, self, name=name,
    # optimize=optimize) with optimize=True by default — exactly what the
    # Solver needs to register this network's parameters with the optimizer.


# =============================================================================
# BEST-WEIGHT SOLVER  (save-best checkpointing + auto-resume)
# =============================================================================

class BestWeightSolver(Solver):
    """
    Thin Solver subclass that saves network weights to outputs/models/ on every
    PhysicsNeMo checkpoint interval (save_network_freq steps).

    Design
    ------
    Attempting to track total loss by wrapping compute_gradients is unreliable:
    PhysicsNeMo's _train_loop may capture that callable as a local variable
    before our wrapper is in place, leaving the tracker permanently at inf and
    causing every save_checkpoint call to return early without writing anything.

    The robust alternative: always save on every checkpoint call, keep only the
    latest _KEEP_LAST files (default 3) so disk usage stays bounded.

    File format
    -----------
    outputs/models/best_weights_step_NNNNNN.pth
    payload: {"states": {"heat_network": <state_dict>}, "step": N}

    Auto-resume
    -----------
    Before the Solver is constructed, the caller scans outputs/models/ for
    existing best_weights_step_*.pth files and warm-starts the network from
    the latest one.  PhysicsNeMo's native checkpoint (outputs/networks/)
    governs optimizer + scheduler state for a full training-state resume.
    """

    _BEST_DIR:  str = os.path.join("outputs", "models")
    _KEEP_LAST: int = 3   # number of recent checkpoint files to retain

    def __init__(self, cfg: "SimConfig", domain: Domain) -> None:
        os.makedirs(self._BEST_DIR, exist_ok=True)
        super().__init__(cfg, domain)

    def save_checkpoint(self, step: int) -> None:
        # Run the native PhysicsNeMo checkpoint first (optimizer + scheduler state).
        # This writes heat_network.0.pth to the current working directory.
        super().save_checkpoint(step)

        # Copy the just-written native checkpoint into our models/ folder with a
        # step-numbered name.  This avoids navigating the Graph/Node API (which
        # changed between modulus-sym versions and caused AttributeError on .nodes).
        native = "heat_network.0.pth"
        if not os.path.exists(native):
            return

        sd = torch.load(native, map_location="cpu")
        ckpt_path = os.path.join(self._BEST_DIR, f"best_weights_step_{step:06d}.pth")
        torch.save({"states": {"heat_network": sd}, "step": step}, ckpt_path)
        print(f"\n  ✓ Saved  {os.path.basename(ckpt_path)}")

        # Keep only the latest _KEEP_LAST checkpoints.
        existing = sorted(
            glob.glob(os.path.join(self._BEST_DIR, "best_weights_step_*.pth"))
        )
        for old in existing[: -self._KEEP_LAST]:
            os.remove(old)
            print(f"  ✗ Removed {os.path.basename(old)}")


# =============================================================================
# 1.  PHYSICAL & NORMALIZATION CONSTANTS
# =============================================================================

# --- Geometry ----------------------------------------------------------------
RADIUS = 0.05   # m  — cylinder radius
HEIGHT = 0.10   # m  — cylinder height (z-axis)

# --- Thermal -----------------------------------------------------------------
ALPHA = 5e-6    # m²/s  — thermal diffusivity

# --- Time domain -------------------------------------------------------------
T_END = 60.0    # s

# --- Temperature range -------------------------------------------------------
T_INITIAL = 300.0               # K  — initial condition / ambient
T_PLASMA  = 4000.0              # K  — front-face Dirichlet BC
T_RANGE   = T_PLASMA - T_INITIAL  # 3700 K

# --- Min-max scaling to [0, 1] -----------------------------------------------
#   x_hat = (x - x_min) / x_scale
X_MIN,   X_SCALE = -RADIUS, 2.0 * RADIUS   # [-0.05, 0.05] → [0, 1]
Y_MIN,   Y_SCALE = -RADIUS, 2.0 * RADIUS   # [-0.05, 0.05] → [0, 1]
Z_MIN,   Z_SCALE =  0.0,    HEIGHT         # [0.0,  0.10]  → [0, 1]
T_MIN_T, T_SCALE =  0.0,    T_END          # [0.0,  60.0]  → [0, 1]

# --- Normalized diffusivity (chain-rule derivation) --------------------------
#
# Starting from the physical PDE:
#   dT/dt = alpha * (d²T/dx² + d²T/dy² + d²T/dz²)
#
# Substituting T = T_RANGE * T_hat + T_INITIAL and x = X_SCALE * x_hat + X_MIN:
#   dT/dt   = T_RANGE / T_SCALE  * dT_hat/dt_hat
#   d²T/dx² = T_RANGE / X_SCALE² * d²T_hat/dx_hat²  (same for y, z)
#
# Dividing through by T_RANGE and multiplying by T_SCALE:
#   dT_hat/dt_hat = (alpha * T_SCALE / X_SCALE²) * Laplacian(T_hat)
#
# Since X_SCALE = Y_SCALE = Z_SCALE = 0.1 m, all three spatial coefficients
# are equal:
#   alpha_hat = 5e-6 * 60 / (0.1)² = 0.030   (dimensionless)
#
ALPHA_HAT = ALPHA * T_SCALE / (X_SCALE ** 2)  # = 0.030


# =============================================================================
# 2.  NORMALIZED HEAT EQUATION PDE
# =============================================================================

class NormalizedHeatEquation3D(PDE):
    """
    Transient heat equation expressed in normalized [0,1] coordinates.

    Residual enforced at interior collocation points:
        R = dT_hat/dt_hat
            - alpha_hat * (d²T_hat/dx_hat² + d²T_hat/dy_hat² + d²T_hat/dz_hat²)
          = 0
    """

    name = "NormalizedHeatEquation3D"

    def __init__(self, alpha_hat: float = ALPHA_HAT):
        x_hat = Symbol("x_hat")
        y_hat = Symbol("y_hat")
        z_hat = Symbol("z_hat")
        t_hat = Symbol("t_hat")

        # Declare output as a Sympy function of the normalized inputs
        T_hat = Function("T_hat")(x_hat, y_hat, z_hat, t_hat)

        residual = (
            diff(T_hat, t_hat)
            - alpha_hat * (
                diff(T_hat, x_hat, 2)
                + diff(T_hat, y_hat, 2)
                + diff(T_hat, z_hat, 2)
            )
        )

        # PhysicsNeMo/Modulus minimizes every entry of self.equations to zero
        self.equations = {"heat_equation": residual}


# =============================================================================
# 3.  HYDRA CONFIG STORE
#     Registers PhysicsNeMoConfig so @hydra.main works without a conf/
#     directory on disk — required for single-file Colab usage.
# =============================================================================

cs = ConfigStore.instance()
cs.store(name="config", node=PhysicsNeMoConfig)


# =============================================================================
# 4.  MAIN TRAINING FUNCTION
# =============================================================================

@hydra.main(version_base="1.2", config_path=None, config_name="config")
def run(cfg: SimConfig) -> None:

    # --- Training hyperparameters --------------------------------------------
    # PhysicsNeMoConfig declares training/optimizer/scheduler/stop_criterion
    # as MISSING sentinels — they must be populated before Solver reads them.
    # open_dict lifts struct validation; OmegaConf.structured() ensures the
    # assigned value is the exact subclass the typed field expects.
    with open_dict(cfg):
        cfg.training = OmegaConf.structured(DefaultTraining(
            max_steps=50_000,
            save_network_freq=500,   # checkpoint every 500 steps for best-weight tracking
            print_stats_freq=100,
            summary_freq=1_000,
        ))
        cfg.stop_criterion = OmegaConf.structured(DefaultStopCriterion())
        cfg.optimizer      = OmegaConf.structured(AdamConf())
        cfg.scheduler      = OmegaConf.structured(ExponentialLRConf(gamma=0.9999))

    # -------------------------------------------------------------------------
    # 4.1  Geometry
    # -------------------------------------------------------------------------
    # Cylinder axis along z:
    #   z=0      — plasma-facing hot side   (Dirichlet BC)
    #   z=HEIGHT — insulated cabin side     (Neumann BC)
    # center=(0, 0, HEIGHT/2) places faces exactly at z=0 (hot side)
    # and z=HEIGHT (insulated side).  center=(0,0,0) would span z=−0.05…+0.05
    # and leave the z=0.1 back-face BC hunting empty space → AssertionError.
    cylinder = Cylinder(
        center=(0, 0, HEIGHT / 2),
        radius=RADIUS,
        height=HEIGHT,
    )

    # -------------------------------------------------------------------------
    # 4.2  Preprocessing / Postprocessing Nodes
    # -------------------------------------------------------------------------

    # Coordinate normalization: physical (x,y,z,t) → normalized (x_hat,…,t_hat)
    # Node.from_sympy(expr, out_name) — one equation per node in physicsnemo.
    # PhysicsNeMo samples points in physical space; these nodes map them into
    # the [0,1] space that the network operates in.
    x_hat_node = Node.from_sympy((Symbol("x") - X_MIN)   / X_SCALE, "x_hat")
    y_hat_node = Node.from_sympy((Symbol("y") - Y_MIN)   / Y_SCALE, "y_hat")
    z_hat_node = Node.from_sympy((Symbol("z") - Z_MIN)   / Z_SCALE, "z_hat")
    t_hat_node = Node.from_sympy((Symbol("t") - T_MIN_T) / T_SCALE, "t_hat")

    # Temperature denormalization: T_hat ∈ [0,1] → T in Kelvin
    T_denorm_node = Node.from_sympy(
        Symbol("T_hat") * T_RANGE + T_INITIAL, "T"
    )

    # Neumann wall expression: radial heat flux on the lateral surface.
    # PhysicsNeMo injects normal_x and normal_y for every sampled boundary point.
    # Physical condition:  dT/dn = normal_x*(dT/dx) + normal_y*(dT/dy) = 0
    # In normalized space (X_SCALE == Y_SCALE cancels):
    #   neumann_wall = normal_x*(dT_hat/dx_hat) + normal_y*(dT_hat/dy_hat) = 0
    # Double-underscore notation: "T_hat__x_hat" = dT_hat/dx_hat (autodiff).
    neumann_wall_node = Node.from_sympy(
        Symbol("normal_x") * Symbol("T_hat__x_hat")
        + Symbol("normal_y") * Symbol("T_hat__y_hat"),
        "neumann_wall",
    )

    # -------------------------------------------------------------------------
    # 4.3  Network Architecture
    # -------------------------------------------------------------------------
    # Pure-PyTorch FourierFeatureNet: random Fourier feature encoding →
    # weight-normed SiLU FC backbone.  No physicsnemo.nn dependency — avoids
    # the FCLayer / Conv1dFCLayer modules that are only in the NVIDIA/modulus
    # core repo (not pip-installable on Python 3.12).
    network = FourierFeatureNet(
        input_keys=[Key("x_hat"), Key("y_hat"), Key("z_hat"), Key("t_hat")],
        output_keys=[Key("T_hat")],
        layer_size=256,
        nr_layers=6,
        n_frequencies=8,
        freq_scale=1.0,
    )

    # jit=False: safer on Colab — CUDA-graph JIT compilation spikes VRAM at
    # startup and can exceed the T4 budget before training begins.
    network_node = network.make_node(name="heat_network", jit=False)

    # -------------------------------------------------------------------------
    # 4.4  PDE Nodes
    # -------------------------------------------------------------------------
    heat_pde  = NormalizedHeatEquation3D(alpha_hat=ALPHA_HAT)
    pde_nodes = heat_pde.make_nodes()

    # -------------------------------------------------------------------------
    # 4.5  Full Node Graph
    # -------------------------------------------------------------------------
    # Order matters: normalization → network → residuals / neumann.
    all_nodes = (
        [x_hat_node, y_hat_node, z_hat_node, t_hat_node]  # (x,y,z,t) → normalized
        + [network_node]      # (x_hat,…,t_hat) → T_hat
        + pde_nodes           # T_hat → heat_equation residual
        + [T_denorm_node]     # T_hat → T  [Kelvin, for diagnostics]
        + [neumann_wall_node] # (normal_x, dT_hat/dx_hat, …) → neumann_wall
    )

    # -------------------------------------------------------------------------
    # 4.6  Domain & Constraints
    # -------------------------------------------------------------------------
    domain = Domain()

    # --- 1) Interior PDE constraint ------------------------------------------
    # Enforces the normalized heat equation at 2000 random collocation points
    # per step throughout the cylinder volume × time domain.
    # batch_size=2000 is the OOM-safe ceiling for the T4 16 GB GPU.
    interior_pde = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=cylinder,
        outvar={"heat_equation": 0},
        batch_size=2000,
        bounds={
            Symbol("x"): (-RADIUS,  RADIUS),
            Symbol("y"): (-RADIUS,  RADIUS),
            Symbol("z"): (0,        HEIGHT),
            Symbol("t"): (0,        T_END),
        },
        lambda_weighting={"heat_equation": 1.0},
        fixed_dataset=False,
        shuffle=True,
    )
    domain.add_constraint(interior_pde, name="interior_pde")

    # --- 2) Front face Dirichlet BC: T=4000 K at z=0, for t>0 ---------------
    # T_hat target = (4000 - 300) / 3700 = 1.0
    # Weight=10.0: strictly enforce the plasma temperature BC.
    # parameterization starts at t=0.1 to avoid the corner discontinuity at
    # (z=0, t=0) where Dirichlet (4000 K) and IC (300 K) conflict.
    front_face_bc = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=cylinder,
        outvar={"T_hat": 1.0},
        batch_size=1000,
        criteria=lambda invar, params: np.isclose(invar["z"], 0.0, atol=1e-4),
        parameterization={Symbol("t"): (0.1, T_END)},
        lambda_weighting={"T_hat": 10.0},
    )
    domain.add_constraint(front_face_bc, name="bc_front_dirichlet")

    # --- 3) Back face Neumann BC: dT/dz=0 at z=HEIGHT -----------------------
    # Insulated cabin wall — zero heat flux through the back face.
    # "T_hat__z_hat" = dT_hat/dz_hat  (PhysicsNeMo double-underscore notation)
    back_face_bc = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=cylinder,
        outvar={"T_hat__z_hat": 0},
        batch_size=1000,
        criteria=lambda invar, params: np.isclose(invar["z"], HEIGHT, atol=1e-4),
        parameterization={Symbol("t"): (0, T_END)},
        lambda_weighting={"T_hat__z_hat": 1.0},
    )
    domain.add_constraint(back_face_bc, name="bc_back_neumann")

    # --- 4) Lateral wall Neumann BC: dT/dn=0 at r=RADIUS --------------------
    # Adiabatic outer wall — models an infinite-plane TPS with no lateral loss.
    # Criteria excludes the top/bottom faces to avoid double-sampling.
    lateral_wall_bc = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=cylinder,
        outvar={"neumann_wall": 0},
        batch_size=1000,
        criteria=lambda invar, params: (
            (np.sqrt(invar["x"] ** 2 + invar["y"] ** 2) >= RADIUS - 1e-4)
            & ~np.isclose(invar["z"], 0.0,    atol=1e-4)
            & ~np.isclose(invar["z"], HEIGHT, atol=1e-4)
        ),
        parameterization={Symbol("t"): (0, T_END)},
        lambda_weighting={"neumann_wall": 1.0},
    )
    domain.add_constraint(lateral_wall_bc, name="bc_wall_neumann")

    # --- 5) Initial condition: T=300 K everywhere at t=0 ---------------------
    # T_hat target = (300 - 300) / 3700 = 0.0
    initial_condition = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=cylinder,
        outvar={"T_hat": 0.0},
        batch_size=2000,
        bounds={
            Symbol("x"): (-RADIUS,  RADIUS),
            Symbol("y"): (-RADIUS,  RADIUS),
            Symbol("z"): (0,        HEIGHT),
        },
        parameterization={Symbol("t"): 0.0},   # pins t=0 without collapsing domain volume
        lambda_weighting={"T_hat": 5.0},
        fixed_dataset=False,
    )
    domain.add_constraint(initial_condition, name="ic_t0")

    # -------------------------------------------------------------------------
    # 4.7  Solver
    # -------------------------------------------------------------------------
    # The Solver owns Adam (lr=1e-3 default), the training loop, checkpoint
    # saving, and TensorBoard logging.
    #
    # Monitor in Colab:
    #   %load_ext tensorboard && %tensorboard --logdir outputs/
    #
    # Expected convergence:
    #   - Front-face BC loss  → ~0 within   5 000 steps  (weight = 10)
    #   - IC loss             → converged by 10 000 steps (weight =  5)
    #   - Interior PDE loss   → below 1e-3 by ~30 000 steps
    #   - Neumann BC losses   → small throughout (soft constraints)
    # -------------------------------------------------------------------------
    # 4.8  Inject Loss Config & Launch Solver
    # -------------------------------------------------------------------------
    # Resume is handled automatically by PhysicsNeMo: on startup the Trainer
    # scans the working directory for heat_network.0.pth / optim_checkpoint.0.pth
    # and restores weights + optimizer state if found.
    # LossConf must be a typed dataclass (not a raw dict) to pass PhysicsNeMo's
    # schema validation.  _target_ points at the concrete aggregator class.
    from physicsnemo.sym.hydra.config import LossConf
    OmegaConf.set_struct(cfg, False)
    loss_cfg = LossConf()
    loss_cfg._target_ = "physicsnemo.sym.loss.aggregator.Sum"
    cfg.loss = loss_cfg

    # BestWeightSolver wraps compute_gradients to track total loss and
    # overrides save_checkpoint to persist the best network weights.
    slv = BestWeightSolver(cfg=cfg, domain=domain)
    slv.solve()


# =============================================================================
# 5.  POST-TRAINING VERIFICATION
# =============================================================================

def plot_temperature_profile(network, checkpoint_path: str = None):
    """
    Sample the trained network along the cylinder axis (x=0, y=0) at
    t = {0, 10, 30, 60} seconds and plot T (Kelvin) vs z.

    Physical sanity checks
    ----------------------
    1. t=0   : T ≈ 300 K everywhere               — IC satisfied
    2. z=0   : T ≈ 4000 K for all t > 0           — Dirichlet BC satisfied
    3. z=0.1 : dT/dz ≈ 0 (profile flattens)       — back-face Neumann BC
    4. Penetration depth at t=60 s:
         δ = √(alpha·t) ≈ √(5e-6 × 60) ≈ 0.017 m ≈ 1.7 cm from z=0
    """
    import matplotlib.pyplot as plt

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cuda")
        network.load_state_dict(state)

    network.eval()
    device = next(network.parameters()).device
    z_phys = np.linspace(0.0, HEIGHT, 200, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 5))

    for t_phys in [0.0, 10.0, 30.0, 60.0]:
        x_hat = np.full_like(z_phys, (0.0 - X_MIN) / X_SCALE)
        y_hat = np.full_like(z_phys, (0.0 - Y_MIN) / Y_SCALE)
        z_hat = z_phys / Z_SCALE
        t_hat = np.full_like(z_phys, t_phys / T_SCALE)

        inputs = {
            "x_hat": torch.tensor(x_hat[:, None], device=device),
            "y_hat": torch.tensor(y_hat[:, None], device=device),
            "z_hat": torch.tensor(z_hat[:, None], device=device),
            "t_hat": torch.tensor(t_hat[:, None], device=device),
        }

        with torch.no_grad():
            T_hat_pred = network(inputs)["T_hat"]

        T_pred = T_hat_pred.cpu().numpy().flatten() * T_RANGE + T_INITIAL
        ax.plot(z_phys * 100.0, T_pred, label=f"t = {t_phys:.0f} s")

    ax.axhline(T_INITIAL, color="gray", linestyle="--", linewidth=0.8,
               label=f"T₀ = {T_INITIAL:.0f} K")
    ax.axhline(T_PLASMA,  color="red",  linestyle="--", linewidth=0.8,
               label=f"T_plasma = {T_PLASMA:.0f} K")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Stratos PINN v3 — Axial Temperature Profile")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("temperature_profile.png", dpi=150)
    plt.show()
    print("Plot saved to temperature_profile.png")


# =============================================================================
# 6.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
