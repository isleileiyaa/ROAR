"""ROAR v1 gate-only precision fine-tune (an independent new method -- NOT a retry of
ROARHEAD v2, which changed the gate's INPUT and retrained everything from scratch and regressed;
this instead starts from an already-fully-trained ROAR v1 checkpoint, freezes candidate
combination and host completely, and only re-tunes the SCALAR blend weight u_i under a new,
more-direct loss that also regularizes against straying far from the original checkpoint).

Base checkpoint: roar_b05_k10_seed2021 (the already-trained AFocusRoarModel, beta=0.5, top_k=10,
h_linear paperhp host) -- loaded whole via --init_from_checkpoint (see pretrain_A20.py wiring),
NOT via --afocus_host_ckpt (that only loads the host; here host+psi_cand+psi_base all come
pre-trained from this one checkpoint file, since AFocusRoarModel.state_dict() already saves the
complete model, host included -- confirmed by reading pretrain_A20.py's final torch.save call).

Frozen: host (as always) + psi_cand (candidate network, entirely). Trainable: psi_base (the gate,
produces u_i) ONLY. Hooked into the existing layers_to_unfreeze substring-match mechanism as
layers_to_unfreeze=['psi_base'] (narrowed from the existing afocus family's ['psi_'] prefix) --
see pretrain_A20.py wiring. IMPORTANT naming note: the frozen reference-gate snapshot below is
named `gate_ref`, deliberately NOT containing "psi_base" as a substring -- layers_to_unfreeze
matching is `any(layer in name for layer in layers_to_unfreeze)`, and if the snapshot were named
e.g. `psi_base_ref`, the substring 'psi_base' would match it too and accidentally unfreeze it.

Prediction formula (candidate mix r_i^ref and host y0 both fixed/stopgraded; only u_i trainable):
  y0_i     = host's own raw median forecast (qp[:, i50]), sg()'d (host runs under
             base_forecast()'s @torch.no_grad() anyway, so already ungraphed)
  r_i^ref  = y0_i + d_i * scale_i   (raw units), where d_i = sum_j p_ij*(Z_j - y0_norm) is
             AFocusRoarModel's own `d`/`r` quantity, computed via the FROZEN psi_cand -- same
             candidate construction (Z_level/Z_anchor) as every other ROAR/combo4 variant,
             unchanged, wrapped in torch.no_grad() here since nothing needs its gradient anymore
  u_i(psi_u) = sigmoid(psi_base(base_f))          -- the ONLY differentiable part
  y_hat_i  = sg(y0_i) + u_i(psi_u) * sg(r_i^ref - y0_i)
All 9 output quantiles are shifted by the same (y_hat_i - y0_i) raw delta, matching every other
afocus_model*.py variant's `shifted = qp + (corr*scale).unsqueeze(1)` pattern (broadcast the
point-forecast correction across the quantile axis, preserving the host's own uncertainty width).

Loss (COMPLETELY replaces the pinball+weighted+BCE loss for this fine-tune stage -- nothing else
is added, matching the explicit design choice "just this one loss, avoid the ASH trap of mixing objectives"):
  L_cal = mean_i[ MSE(y_hat_i, y_i) + kappa * MSE(y_hat_i, sg(y_hat_i^ref)) ]
  kappa = 20 (confirmed value, hardcoded default -- not swept, so not exposed as a new CLI flag)
  y_hat_i^ref = this checkpoint's own ORIGINAL prediction, i.e. sg(y0_i) + u_i^ref * sg(r_i^ref -
                y0_i) computed with a FROZEN SNAPSHOT of psi_base taken at the moment
                freeze_for_gatecal() is called (right after the checkpoint loads, before any
                fine-tuning step) -- NOT recomputed from the live (being-trained) psi_base, so it
                stays a fixed anchor throughout the whole fine-tune run, exactly like the spec.
Both MSE terms are computed in RAW/final-eval-scale units (target is passed to forward() already
raw -- see how every other variant does `t_norm = (target-loc)/scale`, meaning `target` itself is
raw to begin with; there's no separately-named "denormalize" utility anywhere in this codebase --
checked afocus_model.py and dataset.py -- the raw-scale reconstruction is always just this inline
`host's raw qp + normalized_correction*scale` pattern, reused verbatim here, not reimplemented).

Design decisions made beyond the literal spec (flagging both, same as every prior file here):
1. NaN-masking added to L_cal (spec's formula doesn't mention it) -- the underlying pretrain data
   can have NaN targets (every other loss in this codebase masks them: v1/v2/combo4 all do
   `valid = ~torch.isnan(...)`), so this reuses that same masking convention rather than skipping
   it, since skipping would silently corrupt the loss whenever a NaN slips through.
2. Training data: "复用当初算calibrate()用的同一份定义" is satisfied by running through
   pretrain_A20.py's REGULAR training loop/train_loader unmodified for this augment_mode -- its
   CustomPretrainDataset(...).shuffle(...) construction is already textually identical to
   calibrate()'s own `_calib_loader` construction (same class, same args), so no new loader needed
   -- calibrate() itself is simply never invoked for this augment_mode (L_cal doesn't use t_h/
   mu_h/mu_g at all, so there was nothing to calibrate for).
3. Step-count sweep: run as 4 INDEPENDENT constant-LR fine-tunes (--train_steps N
   --evaluation_steps N for N in 100/500/1000/2000, matching every other afocus-family run in
   this project, which always sets evaluation_steps==train_steps -- meaning the scheduler only
   ever fires once, at the very end, after training is already done, so LR is effectively CONSTANT
   throughout every prior run here), rather than one continuous 2000-step run with periodic
   snapshots (which would introduce a live mid-training cosine decay -- new behavior not used
   anywhere else in this project, and not something the original spec specified). Both are defensible;
   chose the one consistent with established practice. Same PRETRAIN_SEED + init_from_checkpoint +
   identical code path across all 4 runs -> identical RNG consumption before the data loop starts
   -> first N steps' batches are the same sequence across all 4 runs.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from afocus_model import ZCLIP
from afocus_model_roar import AFocusRoarModel
from models.ChronosBolt import ChronosBoltOutput


class AFocusRoarGateCalModel(AFocusRoarModel):
    # __init__ deliberately NOT overridden: state_dict keys must match AFocusRoarModel exactly so
    # a roar_b05_k10_seed2021 checkpoint loads via --init_from_checkpoint with 0 missing/unexpected
    # keys. gate_ref does not exist until freeze_for_gatecal() runs (see module docstring for why
    # it can't be created in __init__: it needs the TRAINED weights, which don't exist yet then).
    kappa: float = 20.0

    def freeze_for_gatecal(self):
        if not hasattr(self, 'gate_ref'):
            self.gate_ref = copy.deepcopy(self.psi_base)
        for p in self.gate_ref.parameters():
            p.requires_grad_(False)
        for p in self.psi_cand.parameters():
            p.requires_grad_(False)  # belt-and-suspenders on top of layers_to_unfreeze=['psi_base']

    def forward(self, context, mask=None, target=None, target_mask=None, retrieved_seq=None, distances=None):
        assert hasattr(self, 'gate_ref'), 'call freeze_for_gatecal() before forward()'
        with torch.no_grad():
            qp = self.base_forecast(context, retrieved_seq, distances, mask)  # (B,Q,H) raw
            _, (loc, scale) = self.host.instance_norm(context.float())        # (B,1) each, query coords
            y0_raw = qp[:, self.i50]                                          # (B,H) == \hat y_i^0
            y0 = (y0_raw - loc) / scale                                       # normalized, for base_f/dist_f only
            iqr = ((qp[:, self.iq75] - qp[:, self.iq25]).abs() / scale).mean(-1)  # (B,)

            H = self.H
            rx, ry = retrieved_seq[..., :-H].float(), retrieved_seq[..., -H:].float()
            B, K, _ = rx.shape
            _, (lj, sj) = self.host.instance_norm(rx.reshape(B * K, -1))
            sj2 = sj.reshape(B, K)

            Z_level = ((ry.reshape(B * K, -1) - lj) / sj).reshape(B, K, H).clamp(-ZCLIP, ZCLIP)
            x_tilde_T = (context[..., -1].float() - loc.squeeze(-1)) / scale.squeeze(-1)
            rx_last = rx[..., -1]
            Z_anchor = (x_tilde_T.view(B, 1, 1) + (ry - rx_last.unsqueeze(-1)) / sj2.unsqueeze(-1))
            Z_anchor = Z_anchor.clamp(-ZCLIP, ZCLIP)
            Z = torch.cat([Z_level, Z_anchor], dim=1)                          # (B,2K,H)
            flag = torch.cat([torch.zeros(B, K, device=Z.device), torch.ones(B, K, device=Z.device)], dim=1)

            ld1 = torch.log1p(distances.float().clamp_min(0))
            ld = torch.cat([ld1, ld1], dim=1)
            dist_f = torch.log1p((Z - y0.unsqueeze(1)).abs().mean(-1))
            base_f = torch.stack([torch.log1p(iqr), ld.mean(1), torch.log1p(Z.std(1).mean(-1))], -1)  # (B,3)

            cand_in = torch.stack([ld, dist_f, flag], -1)                      # (B,2K,3)
            p = torch.softmax(self.psi_cand(cand_in).squeeze(-1), dim=-1)      # frozen psi_cand
            d = (p.unsqueeze(-1) * (Z - y0.unsqueeze(1))).sum(1)               # (B,H) normalized candidate mix
            r_ref_raw = y0_raw + d * scale                                     # (B,H) raw == r_i^ref

            u_ref = torch.sigmoid(self.gate_ref(base_f).squeeze(-1))           # (B,), frozen snapshot
            y_hat_ref = y0_raw + u_ref.unsqueeze(-1) * (r_ref_raw - y0_raw)    # (B,H) raw == \hat y_i^ref

        u = torch.sigmoid(self.psi_base(base_f).squeeze(-1))                   # (B,), TRAINABLE gate
        y_hat = y0_raw + u.unsqueeze(-1) * (r_ref_raw - y0_raw)                # y0_raw/r_ref_raw both
                                                                                 # already ungraphed (built
                                                                                 # entirely inside no_grad above)

        loss = None
        if target is not None:
            target_raw = target.float()
            valid = ~torch.isnan(target_raw) if target_mask is None else target_mask.bool()
            target_filled = torch.nan_to_num(target_raw, nan=0.0)
            v = valid.float()
            n_valid = v.sum(-1).clamp_min(1.0)

            def masked_mse(a, b):
                return (((a - b) ** 2) * v).sum(-1) / n_valid                  # (B,), per-sample mean over H

            l_data = masked_mse(y_hat, target_filled)                          # MSE(y_hat_i, y_i)
            l_anchor = masked_mse(y_hat, y_hat_ref.detach())                   # MSE(y_hat_i, sg(y_hat_ref_i))
            loss = (l_data + self.kappa * l_anchor).mean()                     # mean_i[...] == (1/B)*sum_i[...]

        shifted = qp + (y_hat - y0_raw).unsqueeze(1)                           # (B,Q,H) raw, all quantiles shifted
        self.last_a0 = (1.0 - u).detach()
        return ChronosBoltOutput(loss=loss, loss_forecast=loss, quantile_preds=shifted)
