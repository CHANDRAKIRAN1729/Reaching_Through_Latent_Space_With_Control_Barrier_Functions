"""
CBF Model — Neural Control Barrier Function B_θ(z, o).

Defines the barrier network and the CBF safety correction function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BarrierNet(nn.Module):
    """
    Neural Control Barrier Function B_θ(z, o).

    Input:  concat(z, o) where z is the latent code and o is the obstacle descriptor.
    Output: Scalar barrier value B (signed, unbounded).
            B(z, o) ≥ 0  →  Safe
            B(z, o) < 0   →  Unsafe

    Architecture mirrors the classifier head in vae_obs.py (fc32 → fc_obs → fc42)
    but is a standalone module with no sigmoid — output is the raw signed barrier value.
    """

    def __init__(self, latent_dim=7, obs_dim=4, hidden_units=2048, num_hidden=4):
        super(BarrierNet, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Input layer: concat(z, o) → hidden
        self.fc_in = nn.Linear(latent_dim + obs_dim, hidden_units)

        # Hidden layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )

        # Output layer: hidden → scalar barrier value
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z, obs):
        """
        Compute barrier value B(z, o).

        Args:
            z:   (batch, latent_dim) latent codes
            obs: (batch, obs_dim) obstacle descriptors [x, y, h, r]

        Returns:
            B: (batch,) scalar barrier values
        """
        x = torch.cat([z.view(-1, self.latent_dim), obs.view(-1, self.obs_dim)], dim=-1)
        h = F.elu(self.fc_in(x))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


def cbf_safety_correction(cbf_net, z_current, z_nominal, obs, alpha, delta_t,
                          lambda_max=1.0, safe_threshold=None, max_iters=5):
    """
    Single-step closed-form safe latent update.

    Given a nominal next state z_nom from the Goal+Prior optimizer, compute
    the minimum correction to satisfy the CBF decrease constraint:

        B(z_safe) >= (1 - α·Δ) · B(z_k)

    Solution (via first-order Taylor approximation of B around z_nom):

        z_safe = z_nom + λ · ∇B(z_nom)
        λ = max(0, (B_target - B_nom) / ||∇B||²)

    When λ = 0, the nominal step already satisfies the constraint and
    z_safe = z_nom (zero-cost when safe).

    Args:
        cbf_net:    Trained BarrierNet
        z_current:  (1, latent_dim) tensor — z_k (current state)
        z_nominal:  (1, latent_dim) tensor — z_{k+1}^nom (nominal next state)
        obs:        (1, obs_dim) tensor — obstacle parameters
        alpha:      float — barrier decay rate
        delta_t:    float — time step
        lambda_max: float — maximum λ (caps correction magnitude)
        safe_threshold: float or None — (unused, kept for API compatibility)
        max_iters:  int — (unused, kept for API compatibility)

    Returns:
        z_safe:     (1, latent_dim) tensor — z_{k+1}^safe
        info:       dict with correction diagnostics
    """
    # =========================================================================
    # Step 1: Evaluate B(z_k) — no gradient needed
    # =========================================================================
    with torch.no_grad():
        B_current = cbf_net(z_current.detach(), obs)    # B(z_k)

    # Target barrier value: B_target = (1 - α·Δ) · B(z_k)
    B_target = (1.0 - alpha * delta_t) * B_current

    # =========================================================================
    # Step 2: Evaluate B(z_nom) and ∇B(z_nom) — Eqs. 3, 5-8
    # =========================================================================
    z_nom = z_nominal.detach().clone().requires_grad_(True)
    B_nom = cbf_net(z_nom, obs)
    B_nom_val = B_nom.item()

    # Check: is correction needed? (Eq. 10: λ > 0 only when B_nom < B_target)
    if B_nom_val >= B_target.item():
        # Nominal step already satisfies the constraint — no correction needed
        return z_nominal.detach().clone(), {
            'lambda_val': 0.0,
            'B_current': B_current.item(),
            'B_nominal': B_nom_val,
            'B_target': B_target.item(),
            'B_safe': B_nom_val,
            'correction_applied': False,
            'constraint_satisfied': True,
            'rejected': False,
            'grad_norm': 0.0,
        }

    # =========================================================================
    # Step 3: Compute ∇B(z_nom) — Eq. 3
    # =========================================================================
    grad_B = torch.autograd.grad(
        B_nom, z_nom, grad_outputs=torch.ones_like(B_nom),
        create_graph=False, retain_graph=False
    )[0]

    d_norm_sq = torch.sum(grad_B ** 2) + 1e-8
    grad_norm_val = torch.sqrt(d_norm_sq).item()

    # =========================================================================
    # Step 4: Compute λ — Eq. 9
    #   λ = (B_target - B(z_nom)) / ||∇B(z_nom)||²
    # =========================================================================
    lambda_linear = ((B_target - B_nom) / d_norm_sq).item()

    # Eq. 10: λ >= 0 (and cap at lambda_max for stability)
    lambda_val = min(max(0.0, lambda_linear), lambda_max)

    # =========================================================================
    # Step 5: Compute z_safe — Eq. 4
    #   z_safe = z_nom + λ · ∇B(z_nom)
    # =========================================================================
    z_safe = z_nom.detach() + lambda_val * grad_B.detach()

    # Evaluate B at the corrected state for diagnostics
    with torch.no_grad():
        B_safe_val = cbf_net(z_safe, obs).item()

    info = {
        'lambda_val': lambda_val,
        'B_current': B_current.item(),
        'B_nominal': B_nom_val,
        'B_target': B_target.item(),
        'B_safe': B_safe_val,
        'correction_applied': lambda_val > 0.0,
        'constraint_satisfied': B_safe_val >= B_target.item(),
        'rejected': False,
        'grad_norm': grad_norm_val,
    }

    return z_safe.detach(), info


