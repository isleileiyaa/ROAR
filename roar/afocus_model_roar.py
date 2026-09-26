"""ROAR ablation: same host, same 2K-candidate construction, same pinball/BCE/scale^2 loss as
combo4 (afocus_model_combo4.AFocusCombo4Model) -- the ONLY thing this file changes is how psi's
outputs combine into a correction. combo4 uses one softmax over {host, 2K candidates}
((2K+1)-way). ROAR instead splits that into two independent heads (this is the original
ROAR-style parameterization, being re-tested here against the h_linear_head host instead of the
v3-decoupled host she originally tested it against, to separate "does ROAR's split
parameterization itself work" from "was it the host choice that made ROAR look bad"):
  p_ij = softmax_j(psi_cand(cand_in)_j)   over the 2K candidates ONLY (no host slot in this
                                            softmax -- p always sums to 1 over just the candidates)
  u_i  = sigmoid(psi_base(base_f))        independent scalar gate on correction magnitude
  r_i  = sum_j p_ij * (Z_j - y0)          full candidate-mix correction (unscaled)
  y_hat_i = y0 + u_i * r_i

Everything else -- host, candidate construction (K level-aligned + K anchor-aligned, same
InstanceNorm attributions), psi_cand/psi_base input features (identical 3-dim cand_in / 3-dim
base_f), w_i formula ((1+beta*h*g)*scale^2), 9-quantile pinball main loss in raw coordinates, and
the auxiliary BCE abstention loss's role -- is copy-identical to combo4, not reimplemented or
altered. This file does not import from or modify afocus_model_combo4.py; it duplicates the
shared candidate-construction block on purpose so combo4 stays completely untouched.

Two explicit design decisions NOT dictated verbatim by the ROAR spec message (flagging both for
review before the diff is approved, same as the Z_j-InstanceNorm-attribution question asked
before combo4 was written):

1. psi_base bias init: combo4's bias=5.2 was tuned so softmax's host slot starts at a0~=0.90 (a
   softmax logit prior). Here psi_base's output goes through sigmoid to directly BECOME u_i (the
   correction-magnitude gate), a different role -- reusing 5.2 would make sigmoid(5.2)=0.994,
   i.e. ~full-strength correction from step 0, inverting the "start close to host, let the
   correction earn its gradient" curriculum both AFocusModel and combo4 use. To preserve that
   same curriculum under ROAR's split, initialized instead to bias=-2.197 = logit(0.10), so
   u_i starts at sigmoid(-2.197)=0.10 (small correction, host-dominated start) -- the mirror of
   combo4's a0~=0.90 "mostly host" prior, translated into ROAR's multiplicative-gate role.
2. Auxiliary BCE target: combo4 zeroes a0 and renormalizes the remaining candidate weights to
   get y_tilde (the "fully forced correction" prediction), then trains (1-a0) via BCE to predict
   whether that helps. Under ROAR, p already IS a softmax over candidates only (no host slot to
   zero/renormalize) -- so y_tilde = y0 + r_i directly, no renormalization step needed. u_i
   fills (1-a0)'s role in the BCE target (both mean "how much to trust the full candidate
   correction"): L_aux = BCE(u_i, label), label_i = 1[MSE(y_tilde_i, y_i) < l0_i], same label
   definition as combo4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from afocus_model import AFocusModel, ZCLIP
from models.ChronosBolt import ChronosBoltOutput


class AFocusRoarModel(AFocusModel):
    def __init__(self, host: nn.Module, beta: float = 0.5, t_h: float = float('nan'), eps: float = 1e-6,
                 lambda_gate: float = 1.0):
        # weight_mode unused (same as combo4: fixed w_i formula below, not the parent's dispatch) --
        # pass 'joint' only to satisfy the parent's assert.
        super().__init__(host, beta=beta, t_h=t_h, eps=eps, weight_mode='joint')
        # weight on the auxiliary abstention BCE (L_aux) in the total loss; 1.0 == previous hardcoded value.
        self.lambda_gate = lambda_gate
        # psi_cand: identical shape/inputs to combo4's (3-dim: [ld_j, dist_f_j, flag_j]), now
        # produces p_ij logits (softmax over 2K only, no host-slot competitor).
        self.psi_cand = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        # psi_base: identical shape/inputs to combo4's (3-dim base_f), now produces u_i's sigmoid
        # logit instead of a softmax host-slot logit. Bias re-init: see docstring point 1.
        nn.init.constant_(self.psi_base[2].bias, -2.197)
        # Same IQR quantile-pair resolution as combo4 (0.2/0.8, tie-break confirmed).
        q = host.quantiles.detach().cpu()
        self.iq25 = int(torch.argmin((q - 0.2).abs()))
        self.iq75 = int(torch.argmin((q - 0.8).abs()))

    def forward(self, context, mask=None, target=None, target_mask=None, retrieved_seq=None, distances=None):
        qp = self.base_forecast(context, retrieved_seq, distances, mask)  # (B, Q, H) raw units
        with torch.no_grad():
            _, (loc, scale) = self.host.instance_norm(context.float())        # (B,1) each, query coords
            y0 = (qp[:, self.i50] - loc) / scale                              # (B,H) normalized
            iqr = ((qp[:, self.iq75] - qp[:, self.iq25]).abs() / scale).mean(-1)  # (B,)

            H = self.H
            rx, ry = retrieved_seq[..., :-H].float(), retrieved_seq[..., -H:].float()  # (B,K,Lctx),(B,K,H)
            B, K, _ = rx.shape
            _, (lj, sj) = self.host.instance_norm(rx.reshape(B * K, -1))      # (B*K,1) each
            sj2 = sj.reshape(B, K)

            # branch 0: level-aligned (identical to combo4/AFocusModel)
            Z_level = ((ry.reshape(B * K, -1) - lj) / sj).reshape(B, K, H).clamp(-ZCLIP, ZCLIP)

            # branch 1: anchor/delta-aligned (identical to combo4)
            x_tilde_T = (context[..., -1].float() - loc.squeeze(-1)) / scale.squeeze(-1)  # (B,)
            rx_last = rx[..., -1]                                              # (B,K) raw
            Z_anchor = (x_tilde_T.view(B, 1, 1) + (ry - rx_last.unsqueeze(-1)) / sj2.unsqueeze(-1))
            Z_anchor = Z_anchor.clamp(-ZCLIP, ZCLIP)                           # (B,K,H)

            Z = torch.cat([Z_level, Z_anchor], dim=1)                          # (B,2K,H)
            flag = torch.cat([torch.zeros(B, K, device=Z.device), torch.ones(B, K, device=Z.device)], dim=1)

            ld1 = torch.log1p(distances.float().clamp_min(0))                  # (B,K)
            ld = torch.cat([ld1, ld1], dim=1)                                  # (B,2K)
            dist_f = torch.log1p((Z - y0.unsqueeze(1)).abs().mean(-1))         # (B,2K)
            base_f = torch.stack([torch.log1p(iqr), ld.mean(1), torch.log1p(Z.std(1).mean(-1))], -1)  # (B,3)

        cand_in = torch.stack([ld, dist_f, flag], -1)                          # (B,2K,3)
        p = torch.softmax(self.psi_cand(cand_in).squeeze(-1), dim=-1)          # (B,2K), ROAR: candidates-only softmax
        u = torch.sigmoid(self.psi_base(base_f).squeeze(-1))                   # (B,), ROAR: independent magnitude gate

        r = (p.unsqueeze(-1) * (Z - y0.unsqueeze(1))).sum(1)                   # (B,H) full candidate-mix correction
        corr = u.unsqueeze(-1) * r                                             # (B,H) magnitude-gated correction

        loss = None
        if target is not None:
            t_norm = (target.float() - loc) / scale                           # (B,H) normalized, w_i's coord system
            valid = ~torch.isnan(t_norm) if target_mask is None else target_mask.bool()
            t_norm = torch.nan_to_num(t_norm, nan=0.0)
            v = valid.float()
            n_valid = v.sum(-1).clamp_min(1.0)
            mse = lambda p_: (((p_ - t_norm) ** 2) * v).sum(-1) / n_valid

            with torch.no_grad():
                l0_ = mse(y0)                                                  # (B,)
                l_cands = (((Z - t_norm.unsqueeze(1)) ** 2) * v.unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)  # (B,2K)
                l_best = l_cands.min(1).values
                g = (l0_ - l_best).clamp_min(0) / (l0_ + self.eps)
                h = (l0_ > self.t_h).float() if torch.isfinite(self.t_h) else torch.zeros_like(l0_)
                w = (1.0 + self.beta * h * g) * (scale.squeeze(-1) ** 2)       # identical to combo4

            # main loss: mean pinball over all 9 quantiles, raw coordinates (identical to combo4).
            y_hat_raw = qp + (corr * scale).unsqueeze(1)                       # (B,Q,H) raw
            target_raw = torch.nan_to_num(target.float(), nan=0.0)             # (B,H) raw, same NaN mask `v`
            tau = self.quantiles.to(Z.device).view(1, -1, 1)                   # (1,Q,1)
            diff = target_raw.unsqueeze(1) - y_hat_raw                         # (B,Q,H)
            ind = (target_raw.unsqueeze(1) <= y_hat_raw).float()
            pinball = 2 * (diff * (ind - tau)).abs() * v.unsqueeze(1)          # (B,Q,H)
            ell = (pinball.sum(-1) / n_valid.unsqueeze(1)).mean(1)             # (B,) mean over Q

            w_sg = w.detach()
            main_loss = (w_sg * ell).sum() / w_sg.sum()

            # auxiliary abstention BCE, ROAR mapping (see docstring point 2): p is already the
            # fully-forced candidate mix (softmax over 2K only, no host slot to renormalize away)
            # -- y_tilde = y0 + r directly. u_i fills (1-a0)'s role as the BCE target.
            y_tilde = y0 + r                                                    # (B,H) normalized
            with torch.no_grad():
                l_tilde = mse(y_tilde)
                label = (l_tilde < l0_).float()
            p_correct = torch.clamp(u, self.eps, 1.0 - self.eps)
            L_aux = F.binary_cross_entropy(p_correct, label)

            loss = main_loss + self.lambda_gate * L_aux

        shifted = qp + (corr * scale).unsqueeze(1)                             # (B,Q,H) raw, final output
        self.last_a0 = (1.0 - u).detach()  # kept for interface parity with AFocusModel/combo4 (a0 == "host share")
        if loss is not None:
            self.last_w = w.detach()
            self.last_l0, self.last_g, self.last_h = l0_.detach(), g.detach(), h.detach()
        return ChronosBoltOutput(loss=loss, loss_forecast=loss, quantile_preds=shifted)
