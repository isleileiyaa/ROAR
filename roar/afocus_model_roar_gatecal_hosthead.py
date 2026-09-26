"""ROAR v1 gate + host-head joint precision fine-tune (本方在gate-only精修"没有信号"之后，自己
设计的下一步实验，不是原方案指定的 -- 刻意避免"host整体解冻+标准loss重新训"这种最像原方案当初
"联合训练"的做法，只新增一个干净变量：解冻host自己的可训练检索融合头（idf_h_linear_head原本训练
时放开的那4个模块：encode_mlp/ret_score_head/fuse_gate/h_pred_head -- 不是"骨干"，Chronos-Bolt
主干部分继续冻结；确认过这4个是host自己pretrain时真正训练的模块，不是猜的，见下方"host头模块名
确认"部分），让它去接管y0（"不修正"分支）的输出，检索修正那一支（候选构造/psi_cand打分/r_ref/
gate的输入特征）继续完全用原始checkpoint的冻结快照算，不随host训练而移动。kappa锚定正则继续保留
（约束最终预测别离原始checkpoint的预测太远），这点跟原方案当初"联合训练"最可能的做法（没有这种锚定
约束）有本质区别，不是简单重复她已经测过的失败结果。

host头模块名确认（不能想当然假设是"h_pred_head"）：查了pretrain_A20.py里augment_mode=='idf_h_
linear_head'这条路径自己训练时放开的layers_to_unfreeze，实际是4个模块：encode_mlp、
ret_score_head、fuse_gate、h_pred_head（不只是h_pred_head一个），在实例化的host上用
named_parameters()验证过，这4个substring精确匹配到10个参数（4+2+2+2），互不重叠，也不跟Chronos-
Bolt主干的279个参数里的其余269个有任何交集。

两个必须精心处理的"冻结/子串匹配"坑（第二个是gate-only精修那次已经踩过一次的同类问题，这次因为
新增了host_ref这份完整的host快照，同样的坑会以新的形式再出现一次，必须重新过一遍）：

1. host_ref命名/子串碰撞：这次新增了一份完整的host深拷贝快照self.host_ref，用来算候选构造/
   psi_cand打分/gate输入特征/y_hat_ref锚点 -- 全部不随训练中的host移动。但host_ref的参数名是
   `host_ref.encode_mlp.0.weight`这样的，'encode_mlp'/'ret_score_head'/'fuse_gate'/'h_pred_head'
   这几个子串在layers_to_unfreeze里出现时，会同时匹配到self.host(想解冻的)和self.host_ref(必须
   保持冻结)两边 -- 不能靠改host_ref内部模块名来避免（那些名字是ChronosBolt类自己定义死的，不能
   按实例改）。解决方式：pretrain_A20.py的freeze/unfreeze主循环跑完之后，额外调用一次
   freeze_for_gatecal_hosthead()方法，显式把self.host_ref整个重新冻结一遍(requires_grad_(False))
   -- 这一步是必须的、不是防御性冗余，跳过就会导致host_ref被意外训练、r_ref/gate输入特征/y_hat_ref
   全部跟着host训练偷偷漂移，"候选那支完全不变"这个实验设计前提就被破坏了。
2. 优化器参数分组同理：给host头一个单独的(更小的)学习率时，筛选"host头参数"同样要用
   `name.startswith('host.')`精确限定，不能只看'encode_mlp'这类子串是否出现在参数名里，否则会把
   host_ref(冻结、且本来就没梯度)的同名参数也混进那个参数组 -- 虽然因为requires_grad=False、
   .grad恒为None，optimizer.step()本身会自动跳过它们，实际不影响训练结果，但保留这个筛选是为了
   代码本身逻辑自洽，不依赖"反正冻结了所以无所谓"这种巧合。

base_forecast()不能直接复用：AFocusModel.base_forecast()被@torch.no_grad()装饰(定义在afocus_
model.py，不改这个共享文件)，且硬编码调用self.host(...)，两点都不满足这次需求(host_live分支需要
梯度、host_ref分支需要显式指定用哪个host模块)。这里改用self._forecast(host_module, ...,
no_grad=True/False)，是base_forecast()内部逻辑的逐字重写(only 参数化了host_module + no_grad开关)，
不是重新设计了一套新算法。

train()/eval()模式：没有override AFocusModel.train()里"self.host.eval()"这行 -- 查过
encode_mlp/ret_score_head/fuse_gate/h_pred_head这4个模块的定义(models/ChronosBolt.py:903-912)，
全部是plain nn.Linear/nn.ReLU，没有Dropout/BatchNorm，train()/eval()模式对它们的前向数值完全没
区别；host其余部分(主干等)保持eval()模式，跟这个项目其余所有实验的一贯做法一致(确定性、无dropout
噪声)。所以这里不需要新的train()override，直接继承。

公式(gate-only版本的直接推广，只多了一个"y0现在可微分、梯度通向host头"的变化)：
  frozen分支(host_ref, gate_ref, 冻结psi_cand): 候选构造Z_level/Z_anchor/psi_cand打分/base_f
    (gate输入)/r_ref_raw/y_hat_ref(anchor) -- 跟gate-only版本逐字一致，只是把self.host换成
    self.host_ref(instance_norm本身无可训练参数，数值上其实无所谓用哪个，但这里刻意显式用
    host_ref，不依赖"instance_norm恰好没参数"这个巧合)
  live分支(host, 可训练的4个头模块): y0_live_raw = self._forecast(self.host,...,no_grad=False)[:,i50]
    -- 唯一一处梯度通向host头的地方
  u_i = sigmoid(psi_base(base_f))  -- base_f来自frozen分支，gate"感知不到"host在移动
  y_hat_i = y0_live_raw + u_i * (r_ref_raw - y0_live_raw)  -- r_ref_raw是常数(sg)，y0_live_raw
    可微分(通向host头)，u_i可微分(通向psi_base)
  L_cal = mean_i[MSE(y_hat_i, y_i) + kappa*MSE(y_hat_i, sg(y_hat_i^ref))]  -- 跟gate-only版本
    的loss结构完全一样，kappa=20不变，y_hat_i^ref完全来自frozen分支，训练全程不变

输出quantile_preds: shifted = qp_live + (y_hat - y0_live_raw)，用LIVE host自己当前的qp(会随
训练变化)做9个分位数的基准，保留host自己的不确定性宽度估计，跟其余每个variant的"host raw qp +
correction*scale"这套内联写法保持同一个模式(这里由于kappa/y_hat都已经是raw units，不需要再乘
scale)。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from afocus_model import ZCLIP
from afocus_model_roar_gatecal import AFocusRoarGateCalModel
from models.ChronosBolt import ChronosBoltOutput

HOST_HEAD_MODULE_NAMES = ('encode_mlp', 'ret_score_head', 'fuse_gate', 'h_pred_head')


class AFocusRoarGateCalHostHeadModel(AFocusRoarGateCalModel):
    # gamma: post-hoc scalar rescaling of the gate-selected correction magnitude, applied only to
    # the OUTPUT (quantile_preds), never to the training loss (l_data/l_anchor/y_hat below are
    # unaffected -- gamma calibration is a pure inference-time exploration on an already-trained
    # checkpoint, it does not retrain anything). Default 1.0 = unmodified behaviour, exactly
    # reproducing every step500/1000/2000/3000/... result already reported (none of those runs
    # ever touch self.gamma). Only the gamma-calibration step (see roar_gatecal_hosthead_gamma_scan.py)
    # sets this to something else, and only on a copy used purely for eval, never during training.
    gamma: float = 1.0

    def freeze_for_gatecal_hosthead(self):
        self.freeze_for_gatecal()  # gate_ref快照 + 冻结psi_cand (复用不变)
        if not hasattr(self, 'host_ref'):
            self.host_ref = copy.deepcopy(self.host)
        for p in self.host_ref.parameters():
            p.requires_grad_(False)
        self.host_ref.eval()

    def _forecast(self, host_module, context, retrieved_seq, distances, mask=None, no_grad=True):
        """base_forecast()的逐字重写：参数化host_module + no_grad开关，因为
        AFocusModel.base_forecast()被硬编码成@torch.no_grad() + self.host，两点都不满足这里的
        需求(host_ref分支要能传入不同的host模块，host_live分支需要梯度)。"""
        dummy = torch.zeros(context.shape[0], self.H, device=context.device, dtype=context.dtype)
        host_kwargs = dict(context=context, target=dummy, retrieved_seq=retrieved_seq, distances=distances)
        if mask is not None:
            host_kwargs['mask'] = mask
        if no_grad:
            with torch.no_grad():
                out = host_module(**host_kwargs)
        else:
            out = host_module(**host_kwargs)
        return out.quantile_preds.float()

    def forward(self, context, mask=None, target=None, target_mask=None, retrieved_seq=None, distances=None):
        assert hasattr(self, 'gate_ref') and hasattr(self, 'host_ref'), \
            'call freeze_for_gatecal_hosthead() before forward()'

        # ---- frozen分支(host_ref)：候选构造/psi_cand打分/gate输入特征/r_ref/y_hat_ref锚点，
        # 全部不随训练中的host移动，逐字复用AFocusRoarGateCalModel.forward()对应部分的逻辑，
        # 只是把self.host换成self.host_ref。 ----
        with torch.no_grad():
            qp_ref = self._forecast(self.host_ref, context, retrieved_seq, distances, mask)  # (B,Q,H) raw
            _, (loc_ref, scale_ref) = self.host_ref.instance_norm(context.float())
            y0_ref_raw = qp_ref[:, self.i50]                                   # (B,H)
            y0_ref = (y0_ref_raw - loc_ref) / scale_ref                        # normalized
            iqr = ((qp_ref[:, self.iq75] - qp_ref[:, self.iq25]).abs() / scale_ref).mean(-1)

            H = self.H
            rx, ry = retrieved_seq[..., :-H].float(), retrieved_seq[..., -H:].float()
            B, K, _ = rx.shape
            _, (lj, sj) = self.host_ref.instance_norm(rx.reshape(B * K, -1))
            sj2 = sj.reshape(B, K)

            Z_level = ((ry.reshape(B * K, -1) - lj) / sj).reshape(B, K, H).clamp(-ZCLIP, ZCLIP)
            x_tilde_T = (context[..., -1].float() - loc_ref.squeeze(-1)) / scale_ref.squeeze(-1)
            rx_last = rx[..., -1]
            Z_anchor = (x_tilde_T.view(B, 1, 1) + (ry - rx_last.unsqueeze(-1)) / sj2.unsqueeze(-1))
            Z_anchor = Z_anchor.clamp(-ZCLIP, ZCLIP)
            Z = torch.cat([Z_level, Z_anchor], dim=1)                          # (B,2K,H)
            flag = torch.cat([torch.zeros(B, K, device=Z.device), torch.ones(B, K, device=Z.device)], dim=1)

            ld1 = torch.log1p(distances.float().clamp_min(0))
            ld = torch.cat([ld1, ld1], dim=1)
            dist_f = torch.log1p((Z - y0_ref.unsqueeze(1)).abs().mean(-1))
            base_f = torch.stack([torch.log1p(iqr), ld.mean(1), torch.log1p(Z.std(1).mean(-1))], -1)  # (B,3)

            cand_in = torch.stack([ld, dist_f, flag], -1)
            p = torch.softmax(self.psi_cand(cand_in).squeeze(-1), dim=-1)      # 冻结psi_cand
            d = (p.unsqueeze(-1) * (Z - y0_ref.unsqueeze(1))).sum(1)           # normalized candidate mix (w.r.t. y0_ref)
            r_ref_raw = y0_ref_raw + d * scale_ref                             # (B,H) raw == r_i^ref

            u_ref = torch.sigmoid(self.gate_ref(base_f).squeeze(-1))
            y_hat_ref = y0_ref_raw + u_ref.unsqueeze(-1) * (r_ref_raw - y0_ref_raw)  # (B,H) raw, 固定anchor

        # ---- live分支(host)：y0，唯一梯度通向host头(encode_mlp/ret_score_head/fuse_gate/
        # h_pred_head)的地方。 ----
        qp_live = self._forecast(self.host, context, retrieved_seq, distances, mask, no_grad=False)
        y0_live_raw = qp_live[:, self.i50]                                     # (B,H), 可微分

        u = torch.sigmoid(self.psi_base(base_f).squeeze(-1))                   # base_f来自frozen分支
        y_hat = y0_live_raw + u.unsqueeze(-1) * (r_ref_raw - y0_live_raw)      # r_ref_raw是sg常量

        loss = None
        if target is not None:
            target_raw = target.float()
            valid = ~torch.isnan(target_raw) if target_mask is None else target_mask.bool()
            target_filled = torch.nan_to_num(target_raw, nan=0.0)
            v = valid.float()
            n_valid = v.sum(-1).clamp_min(1.0)

            def masked_mse(a, b):
                return (((a - b) ** 2) * v).sum(-1) / n_valid

            l_data = masked_mse(y_hat, target_filled)
            l_anchor = masked_mse(y_hat, y_hat_ref.detach())
            loss = (l_data + self.kappa * l_anchor).mean()

        delta = y_hat - y0_live_raw                                             # (B,H) raw, 这个模型自己的
                                                                                 # 未修正预测y0_live_raw相对
                                                                                 # 修正后预测的差值 -- gamma
                                                                                 # 校准(见roar_gatecal_hosthead_
                                                                                 # gamma_scan.py)重新缩放的就是
                                                                                 # 这一项，不是loss用的y_hat
        shifted = qp_live + (self.gamma * delta).unsqueeze(1)                  # (B,Q,H) raw, 用live host自己的qp做基准
        self.last_a0 = (1.0 - u).detach()
        self.last_y0_live = y0_live_raw.detach()                               # 暴露给gamma校准脚本用
        self.last_delta = delta.detach()                                       # 同上
        return ChronosBoltOutput(loss=loss, loss_forecast=loss, quantile_preds=shifted)
