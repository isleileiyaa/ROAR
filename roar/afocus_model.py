"""Method A inside the unchanged TS-RAG pipeline.

Everything except method A itself is the host model and its inputs, untouched:
  * retrieval  = whatever the TS-RAG data pipeline hands the model (Chronos-T5 L2 top-k windows
                 from the TS-RAG knowledge base; pretraining: precomputed indices of
                 pretrain_pairs_ctx512, zero-shot: the precomputed retrieval CSV);
  * base y0    = the frozen host's median forecast (host = Chronos-Bolt via augment 'baseline',
                 or a trained idf_clean_dis_v3 / idf_h_linear_head checkpoint), host kept in eval
                 mode and under no_grad.
Method A (the only new part):
    y_hat = y0 + sum_{j=1..K} a_j (Z_j - y0),  (a_0..a_K) = softmax(psi logits)
    Z_j   = neighbour future aligned with the neighbour's own context statistics (InstanceNorm of
            its 512-point context), expressed in the query's InstanceNorm coordinates
    L     = sum_i sg(w_i) l_i / sum_i sg(w_i)
            l = MSE over valid (non-NaN) target elements, in the query's InstanceNorm coordinates
            (the same scale Chronos-Bolt's own training loss uses)
            h_i = [l0_i > t_h],  g_i = [l0_i - min_j l(Z_j)]_+ / (l0_i + eps)
            mu_h, mu_g, mu_hg = E[h], E[g], E[h*g] over a calibration pass (see calibrate()),
            c = beta * mu_hg
            w_i depends on weight_mode: uniform -> 1, joint -> 1 + beta * h_i * g_i (default,
            beta = 1; the original A20 formula), hard -> 1 + c * h_i / mu_h, gain -> 1 + c * g_i /
            mu_g (Hard/Gain rescaled so their average weight over the calibration set matches
            Joint's, 1 + c; if mu_h or mu_g is 0 then c is 0 too since mu_hg <= min(mu_h, mu_g)
            for h, g >= 0, so the ratio is defined as 0 in that case -- degenerates to uniform)
psi: candidate score MLP(2->16->1) on (log1p distance_j, log1p mean|Z_j - y0|) and base score
MLP(3->16->1) on (log1p IQR, mean log1p distance, log1p candidate spread); base bias init 3.
Output quantile_preds = host quantiles shifted by the same correction, so the median (the value
TS-RAG's test code scores) equals y_hat.  All trainable parameter names start with 'psi_'.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from models.ChronosBolt import ChronosBoltOutput

ZCLIP = 50.0  # numerical guard for near-constant neighbour contexts (tiny scale)


WEIGHT_MODES = ('uniform', 'hard', 'gain', 'joint')


class AFocusModel(nn.Module):
    def __init__(self, host: nn.Module, beta: float = 1.0, t_h: float = float('nan'), eps: float = 1e-6,
                 weight_mode: str = 'joint'):
        super().__init__()
        assert weight_mode in WEIGHT_MODES, f'unknown weight_mode: {weight_mode}'
        self.host = host
        for p in self.host.parameters():
            p.requires_grad = False
        self.beta, self.eps = beta, eps
        self.weight_mode = weight_mode
        self.register_buffer('t_h', torch.tensor(float(t_h)))
        # mu_h, mu_g, c = beta * mu_hg: fixed by calibrate() before training, used to rescale
        # the Hard/Gain formulas (see module docstring); NaN/0 until calibrated.
        self.register_buffer('mu_h', torch.tensor(float('nan')))
        self.register_buffer('mu_g', torch.tensor(float('nan')))
        self.register_buffer('c', torch.tensor(float('nan')))
        self.psi_cand = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.psi_base = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.psi_base[2].bias, 3.0)
        q = host.quantiles.detach().cpu()
        self.i10, self.i50, self.i90 = [int(torch.argmin((q - v).abs())) for v in (0.1, 0.5, 0.9)]
        # ChronosBolt exposes prediction_length only via chronos_config; some other host types
        # set it directly as an attribute -- try the ChronosBolt path first, fall back to a
        # direct attribute lookup so this class isn't hard-locked to one host family.
        self.H = getattr(host, 'chronos_config', host).prediction_length

    def train(self, mode: bool = True):
        super().train(mode)
        self.host.eval()  # host is frozen: never dropout / train-mode behaviour
        return self

    @property
    def quantiles(self):
        return self.host.quantiles

    @torch.no_grad()
    def base_forecast(self, context, retrieved_seq, distances, mask=None):
        """Host forecast; the dummy all-zero target only satisfies the host's loss bookkeeping
        (its forecast is formed before the target is read), so y0 cannot depend on the truth."""
        dummy = torch.zeros(context.shape[0], self.H, device=context.device, dtype=context.dtype)
        # Some host families' forward() has no `mask` parameter at all -- only pass it through
        # when given one, so calling such a host (mask is always None from a normal training
        # loop) doesn't TypeError.
        host_kwargs = dict(context=context, target=dummy, retrieved_seq=retrieved_seq, distances=distances)
        if mask is not None:
            host_kwargs['mask'] = mask
        out = self.host(**host_kwargs)
        return out.quantile_preds.float()  # (B, Q, H), original units

    @torch.no_grad()
    def calibrate(self, loader, retriever, device, n_batches: int = 50) -> dict:
        """Fix t_h, mu_h, mu_g and c = beta * mu_hg from n_batches of an independent iterator over
        this run's training distribution, before training starts; buffers stay fixed afterwards.
        Must be re-run per host (mu_h/mu_g/mu_hg depend on the host's own error distribution) and
        per training run (loader is a fresh independent iterator, not shared/cached across runs).
        Restores the RNG state on exit so calibration doesn't perturb the training sample order.
        """
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        self.t_h.fill_(float('nan'))  # h must read as 0 during this l0/g collection pass
        l0s, gs = [], []
        for j, b in enumerate(loader):
            if j >= n_batches:
                break
            rs = torch.tensor(retriever.whole_seq[b['indices']]).float().to(device)
            x, y = b['x'].float().to(device), b['y'].float().to(device)
            self(context=x, target=y, retrieved_seq=rs, distances=b['distances'].float().to(device))
            l0s.append(self.last_l0.cpu())
            gs.append(self.last_g.cpu())
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])

        l0, g = torch.cat(l0s), torch.cat(gs)
        self.t_h.fill_(l0.median().item())
        h = (l0 > self.t_h.item()).float()
        mu_hg = (h * g).mean().item()
        self.mu_h.fill_(h.mean().item())
        self.mu_g.fill_(g.mean().item())
        self.c.fill_(self.beta * mu_hg)
        return dict(t_h=self.t_h.item(), mu_h=self.mu_h.item(), mu_g=self.mu_g.item(), mu_hg=mu_hg,
                    c=self.c.item(), n=l0.numel())

    def _calib_ratio(self, mu: torch.Tensor) -> torch.Tensor:
        """c / mu, guarded to 0 when mu == 0 (only possible when c == 0 too, since h,g >= 0 gives
        mu_hg <= min(mu_h, mu_g)) -- Hard/Gain then degenerate to uniform, matching Joint's w=1
        for the same (uncalibratable) degenerate case."""
        return torch.where(mu > 0, self.c / mu.clamp_min(self.eps), torch.zeros_like(mu))

    def forward(self, context, mask=None, target=None, target_mask=None, retrieved_seq=None, distances=None):
        qp = self.base_forecast(context, retrieved_seq, distances, mask)
        with torch.no_grad():
            _, (loc, scale) = self.host.instance_norm(context.float())             # query coords
            y0 = (qp[:, self.i50] - loc) / scale                                   # (B, H)
            iqr = ((qp[:, self.i90] - qp[:, self.i10]).abs() / scale).mean(-1)     # (B,)
            H = self.H
            rx, ry = retrieved_seq[..., :-H].float(), retrieved_seq[..., -H:].float()
            B, K, _ = rx.shape
            _, (lj, sj) = self.host.instance_norm(rx.reshape(B * K, -1))
            Z = ((ry.reshape(B * K, -1) - lj) / sj).reshape(B, K, H).clamp(-ZCLIP, ZCLIP)
            ld = torch.log1p(distances.float().clamp_min(0))                       # (B, K)
            dist_f = torch.log1p((Z - y0.unsqueeze(1)).abs().mean(-1))             # (B, K)
            base_f = torch.stack([torch.log1p(iqr), ld.mean(1), torch.log1p(Z.std(1).mean(-1))], -1)

        lc = self.psi_cand(torch.stack([ld, dist_f], -1)).squeeze(-1)             # (B, K)
        l0 = self.psi_base(base_f)                                                 # (B, 1)
        a = torch.softmax(torch.cat([l0, lc], -1), -1)
        corr = (a[:, 1:].unsqueeze(-1) * (Z - y0.unsqueeze(1))).sum(1)            # sg(Z - y0): no grad path
        y_hat = y0 + corr                                                          # (B, H), query coords

        loss = None
        if target is not None:
            t = (target.float() - loc) / scale                                    # same coords as y0
            valid = ~torch.isnan(t) if target_mask is None else target_mask.bool()
            t = torch.nan_to_num(t, nan=0.0)
            v = valid.float()
            n_valid = v.sum(-1).clamp_min(1.0)
            mse = lambda p: (((p - t) ** 2) * v).sum(-1) / n_valid
            l_hat = mse(y_hat)
            with torch.no_grad():
                l0_ = mse(y0)
                l_best = ((((Z - t.unsqueeze(1)) ** 2) * v.unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)).min(1).values
                g = (l0_ - l_best).clamp_min(0) / (l0_ + self.eps)
                h = (l0_ > self.t_h).float() if torch.isfinite(self.t_h) else torch.zeros_like(l0_)
                if self.weight_mode == 'uniform':
                    w = torch.ones_like(l0_)
                elif self.weight_mode == 'hard':
                    w = 1.0 + self._calib_ratio(self.mu_h) * h
                elif self.weight_mode == 'gain':
                    w = 1.0 + self._calib_ratio(self.mu_g) * g
                else:  # joint (original A20 formula, unaffected by calibration)
                    w = 1.0 + self.beta * h * g
            loss = (w * l_hat).sum() / w.sum()

        shifted = qp + (corr * scale).unsqueeze(1)                                 # original units
        self.last_a0 = a[:, 0].detach()
        if loss is not None:
            self.last_w, self.last_l_hat = w.detach(), l_hat.detach()
            self.last_l0, self.last_g, self.last_h = l0_.detach(), g.detach(), h.detach()
        return ChronosBoltOutput(loss=loss, loss_forecast=loss, quantile_preds=shifted)
