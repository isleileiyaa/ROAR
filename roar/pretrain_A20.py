import os
import time
import wandb
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.moment import MOMENTPipelineWithRetrieval
from dataset import (
    CustomPretrainDataset,
    EnvironmentBalancedPretrainDataset,
    STAGE4_TINY_CHUNK_BASENAMES,
    Retriever_for_pretrain,
)
from models.ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval
from models.Moirai2 import Moirai2ModelForForecastingWithRetrieval, Moirai2HLinearHeadModelForForecastingWithRetrieval, Moirai2MoEModelForForecastingWithRetrieval
from models.TimesFM25 import TimesFM25ModelForForecastingWithRetrieval, TimesFM25HLinearHeadModelForForecastingWithRetrieval, TimesFM25MoEModelForForecastingWithRetrieval
    
warnings.filterwarnings('ignore')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')

fix_seed = int(os.environ.get('PRETRAIN_SEED', 2021))
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='ChronosBoltRetrieve')

parser.add_argument('--model_id', type=str, default='ChronosBoltRetrieve_Pretrain')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--retrieve_lookback_length', type=int, default=64)
parser.add_argument('--retrieval_database_path', type=str, default='../database/pretrain/retrieval_database_512.parquet')

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')
parser.add_argument('--debug_shapes', action='store_true', help='print key tensor shapes once for debug')
parser.add_argument('--rho1', type=float, default=0.0)
parser.add_argument('--rho2', type=float, default=0.0)
parser.add_argument('--tau_dis', type=float, default=0.17)
parser.add_argument('--lambda_sem', type=float, default=0.0)
parser.add_argument("--fusion_mode", type=str, default="learned",
                     choices=["learned", "additive"],
                     help="idf_clean_dis_v4专属：最终融合方式。learned=现有final_pred_head"
                          "(cat(y_inv,y_dyn))（默认，等价于不加这个参数时的行为）；"
                          "additive=y_inv+y_dyn直接相加，跳过final_pred_head。"
                          "对v3/其他augment_mode无效。")
parser.add_argument("--head_mode", type=str, default="learned", choices=["learned", "frozen_native"],
                     help="idf_clean_dis_v3/v4专属：y_inv/y_dyn的投影头。learned=现有的"
                          "inv_pred_head/dyn_pred_head_clean（默认，等价于不加这个参数时的行为）；"
                          "frozen_native=复用冻结的原生output_patch_embedding对z_inv/z_dyn各投影一次，"
                          "必须搭配--fusion_mode additive。对其他augment_mode无效。")
parser.add_argument("--disable_ci", action="store_true", default=False, help="c_i fixed to 1 for all samples, ignoring tau confidence weighting (v4 only)")
parser.add_argument("--weight_dis_by_ci", action="store_true", default=False, help="weight L_dis (rho2 decoupling loss) per-sample by c_i, so decoupling pressure concentrates on samples with confident retrieval instead of being applied uniformly (idf_clean_dis_v4 only, no-op otherwise)")
parser.add_argument('--lambda_ord', type=float, default=0.0)
parser.add_argument("--lambda_delta", type=float, default=0.0, help="weight for L_Delta (final-y first-difference Huber loss vs ground truth, v3/v4 only)")
# FINAL_MSE_OBJECTIVE_PATCH_V1: direct accuracy term on the fused median forecast.
parser.add_argument("--lambda_final_mse", type=float, default=0.0,
                    help="weight for fused-output median MSE; 0 preserves the old objective")
parser.add_argument("--calibrate_final_mse_grad_norm", action="store_true", default=False,
                    help="one-batch gradient calibration for lambda_final_mse, then exit")
parser.add_argument("--final_mse_grad_ratio", type=float, default=0.20,
                    help="target ||lambda*g_mse|| / ||g_pinball|| for calibration")
parser.add_argument("--huber_kappa", type=float, default=1.0, help="Huber loss delta/threshold parameter for L_Delta")
parser.add_argument("--gatecal_kappa", type=float, default=20.0,
                    help="afocus_roar_gatecal(_hosthead) only: weight on the anchor MSE term in "
                         "L_cal = mean[MSE(y_hat,y) + gatecal_kappa*MSE(y_hat, sg(y_hat_ref))]. "
                         "Not the same as --huber_kappa (that's L_Delta's threshold). kappa is a "
                         "class attribute on AFocusRoarGateCalModel (not an __init__ param), so "
                         "this is applied via model.kappa = args.gatecal_kappa after construction.")
parser.add_argument("--calibrate_grad_norm", action="store_true", default=False,
                     help="诊断专用：第一步forward后分别对loss_forecast和loss_delta单独backward，"
                          "打印||grad(loss_forecast)||/||grad(loss_delta)||比值和建议的lambda_delta"
                          "取值(5%%/10%%梯度范数占比)，然后退出，不进行真正训练。")
parser.add_argument('--lambda_xcov', type=float, default=0.0)
parser.add_argument('--rho3', type=float, default=0.0)
parser.add_argument('--rho4', type=float, default=0.0)
# output-level disentanglement ablation (idf_clean_dis / RIDDE only):
# dis_mode selects which L_dis term(s) feed into the total loss.
#   latent  -> only rho2 * L_dis(z_inv, z_dyn)              [original RIDDE / Table 4]
#   output  -> only rho_dis_output * L_dis(y_hat_inv, y_hat_dyn)
#   both    -> rho2 * L_dis(z_inv, z_dyn) + rho_dis_output * L_dis(y_hat_inv, y_hat_dyn)
parser.add_argument('--dis_mode', type=str, default='latent', choices=['latent', 'output', 'both'])
parser.add_argument('--rho_dis_output', type=float, default=0.0)
# RIDDE "Training Objective ver 2.0" (idf_ridde_v2 only, paper Eq. 15-23):
# L = L_pred + rho_sem*L_sem + rho_xcov*L_xcov + rho_ord*L_ord.
parser.add_argument('--rho_sem', type=float, default=0.0)
parser.add_argument('--rho_xcov', type=float, default=0.0)
parser.add_argument('--rho_ord', type=float, default=0.0)
parser.add_argument('--ord_margin', type=float, default=0.0)
# Alternative/complementary decorrelation terms proposed after the L_xcov
# ablation (RIDDE_ver2.0_xcov消融实验报告.md): rho_cos penalizes within-sample
# cos_sim(z_inv, z_dyn) directly; rho_gbal penalizes gamma's global mean
# drifting away from 0.5 (anti-collapse). Both idf_ridde_v2-only.
parser.add_argument('--rho_cos', type=float, default=0.0)
parser.add_argument('--rho_gbal', type=float, default=0.0)
parser.add_argument('--lambda1', type=float, default=0.0)
parser.add_argument('--lambda2', type=float, default=0.0)
# RIDDE_新版目标函数与最终实验方案 Stage 3 (idf_trr_dualpath only): L_sep =
# L_xcov + beta_var*L_var, weighted into the total loss by lambda_sep.
# gamma0_var 是防塌缩方差下界里那个 gamma_0 阈值(文档4.5节)。L_TRR/CVaR还
# 没接(下一步Stage 4才加)，所以这里没有 lambda_trr/cvar_alpha 这些flag。
parser.add_argument('--lambda_sep', type=float, default=0.0)
parser.add_argument('--beta_var', type=float, default=1.0)
parser.add_argument('--gamma0_var', type=float, default=0.1)
# RIDDE_新版目标函数与最终实验方案 Stage 4 (idf_trr_dualpath only): L_TRR/CVaR
# (文档4.2-4.6节)。lambda_trr=0(默认)时完全不影响 Stage 3 的行为——不会构造
# model_B、不会切换成环境平衡采样、不会新增 nu_tilde 参数，训练循环跟之前
# 完全一样。只有显式传 --lambda_trr > 0 时才会启用下面这一整套逻辑。
parser.add_argument('--lambda_trr', type=float, default=0.0)
parser.add_argument('--cvar_alpha', type=float, default=0.6)
parser.add_argument('--trr_reference_model_path', type=str, default=None,
                     help='Query-only 参考模型(方案文档4.2节模型B)的 state_dict 路径，'
                          '仅在 --lambda_trr > 0 时需要，必须是配方对齐(同backbone/数据/'
                          '步数/优化器)、augment_mode=baseline 训出来的 checkpoint。')
parser.add_argument('--env_group_size', type=int, default=8,
                     help='方案文档5.3节的 G：每个 step 从多少个环境里抽样')
parser.add_argument('--env_batch_size', type=int, default=32,
                     help='方案文档5.3节的 m：每个环境每个 step 抽取的样本数')
parser.add_argument('--env_shuffle_buffer_length', type=int, default=2000,
                     help='每个环境自己的局部shuffle缓冲区大小(环境数多，单个不宜设太大)')
parser.add_argument('--nu_init', type=float, default=0.0,
                     help='nu_tilde 的初始值(softplus之前)，默认softplus(0)=log(2)=0.693起步')
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--dyn_margin', type=float, default=1.0)
parser.add_argument('--aux_loss_detach_ret', type=str2bool, default=True)

# model
parser.add_argument('--model', type=str, default='ChronosBoltRetrieve')
parser.add_argument('--freeze_chronos_bolt', action='store_true', help="freeze the params of chronos-bolt.")
# ---- method A (augment_mode='afocus') ----
parser.add_argument('--afocus_host_aug', type=str, default='baseline', help='augment of the frozen host (baseline = Chronos-Bolt)')
parser.add_argument('--afocus_host_ckpt', type=str, default='', help='trained host checkpoint (v3 / h_linear_head); empty for Bolt')
parser.add_argument('--afocus_beta', type=float, default=1.0)
parser.add_argument('--lambda_gate', type=float, default=1.0,
                    help='afocus_roar only: weight on L_aux (abstention BCE) in loss = main_loss + lambda_gate * L_aux')
parser.add_argument('--afocus_weight_mode', type=str, default='joint', choices=['uniform', 'hard', 'gain', 'joint'],
                     help='L_focus sample weight: uniform=1, hard=1+c*h/mu_h, gain=1+c*g/mu_g, joint=1+beta*h*g '
                          '(c=beta*mu_hg; mu_h/mu_g/mu_hg calibrated by AFocusModel.calibrate before training)')
parser.add_argument('--th_calib_batches', type=int, default=50)
parser.add_argument('--host_head_lr', type=float, default=3e-5,
                     help='afocus_roar_gatecal_hosthead only: separate (smaller) LR for the host '
                          'retrieval-fusion head (encode_mlp/ret_score_head/fuse_gate/h_pred_head), '
                          'psi_base keeps --learning_rate; default 3e-5 matches idf_h_linear_head\'s '
                          'own original paperhp pretraining LR (a first guess, not tuned)')
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base/')
parser.add_argument('--context_length', type=int, default=512)
parser.add_argument('--prediction_length', type=int, default=64)

# pretrain
parser.add_argument('--data_path', type=str, default='../datasets/pretrain/50m-with-retrieval_512', help='pretrain data path')
parser.add_argument('--train_steps', type=int, default=200_000)
parser.add_argument('--evaluation_steps', type=int, default=10_000)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--shuffle_buffer_length', type=int, default=100_000)
parser.add_argument('--grad_clip_value', type=float, default=1.0)
# Base-arm ablation: keep the exact idf_clean_dis architecture, freeze list and
# training budget, but shuffle retrieved_seq across the batch dimension so each
# query is paired with someone else's retrieved neighbors instead of its own.
# This isolates "value of real retrieval content" from "value of having a
# trained fusion head at all" -- a plain augment_mode='baseline' run has no
# trainable fusion head to begin with, so it is not a fair matched control.
parser.add_argument('--kill_retrieval', action='store_true',
                     help='shuffle retrieved_seq across the batch dim so retrieval carries no real signal')

# RIDDE_检索扰动相对风险目标 (rob_doc): L = L_pred + lambda_rob*L_rob
parser.add_argument('--lambda_rob', type=float, default=0.0,
                     help='rho_rob；0则完全跳过L_rob，行为跟改动前一模一样')
parser.add_argument('--kl_radius', type=float, default=0.02, help='epsilon_rob，omega周围的KL球半径')
parser.add_argument('--rob_inner_steps', type=int, default=3)
parser.add_argument('--rob_step_size', type=float, default=0.2)
parser.add_argument('--rob_random_restarts', type=int, default=1)
parser.add_argument('--rob_bisection_steps', type=int, default=24)
parser.add_argument('--f0_checkpoint_path', type=str, default='',
                     help='Query-only参考模型f_0的checkpoint(augment_mode=baseline，即TrueBase)，lambda_rob>0时必填')
parser.add_argument('--init_from_checkpoint', type=str, default='',
                     help='鲁棒训练要接着哪个已经训好的checkpoint继续训(比如Pure-ERM+学习型融合头那个)，不填就是从头初始化')

# gpu
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--gpu_loc', type=int, default=0, help='main gpu location')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)


args = parser.parse_args()

if args.head_mode == "frozen_native":
    if args.fusion_mode != "additive":
        raise ValueError(f"--head_mode frozen_native 必须搭配 --fusion_mode additive，当前fusion_mode={args.fusion_mode}")
    if args.augment_mode not in ("idf_clean_dis_v3", "idf_clean_dis_v4"):
        raise ValueError(f"--head_mode frozen_native 目前只在idf_clean_dis_v3/idf_clean_dis_v4白名单内验证过，当前augment_mode={args.augment_mode}")

# init wandb project
wandb.init(project=f'{args.model}_Pretrain', name=args.model_id)
wandb.config.update(args)


if torch.cuda.is_available():
    device = 'cuda:' + str(args.gpu_loc)
else:
    device = 'cpu'

time_now = time.time()


def build_afocus_host(args, config, tag):
    """AFocus系列wrapper(afocus/afocus_roar/afocus_roar_gatecal_hosthead等)共用的
    backbone三选一host构造：按--model在ChronosBoltRetrieve/Moirai2Retrieve/
    TimesFM25Retrieve之间切换。从原来'afocus'分支里各自写死的host构造代码抽出来，
    afocus_roar/afocus_roar_gatecal_hosthead不再各自硬编码ChronosBoltRetrieve。
    AFocusRoarModel/AFocusRoarGateCalHostHeadModel等wrapper类本身不读tau/ord_margin/
    debug_shapes这几个属性(只有ChronosBolt自己的forward()内部会用到，Moirai2/TimesFM25
    的host class没有这个内部分支)，所以只在ChronosBoltRetrieve分支设置。
    """
    if args.model == 'ChronosBoltRetrieve':
        host = ChronosBoltModelForForecastingWithRetrieval(config=config, augment=args.afocus_host_aug)
        host.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
        if args.afocus_host_ckpt:
            _sd = torch.load(args.afocus_host_ckpt, map_location='cpu')
            _msg = host.load_state_dict({k.replace('module.', ''): v for k, v in _sd.items()}, strict=False)
            print(f'[{tag}] host ckpt {args.afocus_host_ckpt}: missing={len(_msg.missing_keys)} unexpected={len(_msg.unexpected_keys)}')
        host.debug_shapes = False
        host._debug_shapes_printed = False
        host.tau = 0.1
        host.ord_margin = 0.0
        return host
    elif args.model in ('Moirai2Retrieve', 'TimesFM25Retrieve'):
        # 跟ChronosBolt不同：这两个backbone自己的骨干在__init__里就已经加载并冻结好了，
        # RIDDE/h_linear_head融合头必须来自一次独立的、针对该backbone的骨干级预训练产物，
        # 不存在"不训头、纯baseline"这个选项，所以--afocus_host_ckpt是必填的。
        assert args.afocus_host_ckpt, f"{args.model} afocus host requires --afocus_host_ckpt"
        host_cls_map = {
            'Moirai2Retrieve': {
                'idf_h_linear_head': Moirai2HLinearHeadModelForForecastingWithRetrieval,
                'idf_clean_dis': Moirai2ModelForForecastingWithRetrieval,
                'moe': Moirai2MoEModelForForecastingWithRetrieval,
            },
            'TimesFM25Retrieve': {
                'idf_h_linear_head': TimesFM25HLinearHeadModelForForecastingWithRetrieval,
                'idf_clean_dis': TimesFM25ModelForForecastingWithRetrieval,
                'moe': TimesFM25MoEModelForForecastingWithRetrieval,
            },
        }[args.model]
        host_cls = host_cls_map[args.afocus_host_aug]
        host = host_cls(context_length=args.context_length, prediction_length=args.prediction_length)
        _sd = torch.load(args.afocus_host_ckpt, map_location='cpu')
        _msg = host.load_state_dict({k.replace('module.', ''): v for k, v in _sd.items()}, strict=False)
        print(f'[{tag}] host ckpt {args.afocus_host_ckpt}: missing={len(_msg.missing_keys)} unexpected={len(_msg.unexpected_keys)}')
        return host
    else:
        raise ValueError(f"build_afocus_host: unsupported args.model={args.model!r}")


## load model, optimizer
config = AutoConfig.from_pretrained(args.pretrained_model_path)
if hasattr(config, "chronos_config"):
    config.chronos_config["context_length"] = args.context_length
    config.chronos_config["prediction_length"] = args.prediction_length
if args.model == 'ChronosBolt':
    model = ChronosBoltModelForForecasting.from_pretrained(args.pretrained_model_path, config=config)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
elif args.model in ('ChronosBoltRetrieve', 'Moirai2Retrieve', 'TimesFM25Retrieve') and args.augment_mode == 'afocus':
    from afocus_model_k20 import AFocusSplitModel as AFocusModel
    host = build_afocus_host(args, config, tag='afocus')
    model = AFocusModel(host, beta=args.afocus_beta, weight_mode=args.afocus_weight_mode)
elif args.model == 'ChronosBoltRetrieve' and args.augment_mode == 'afocus_combo4':
    # combo4: independent cross-validation implementation (see roar/afocus_model_combo4.py
    # docstring). Unlike afocus/AFocusSplitModel, host and candidates share the same K (no K=20/
    # host_k=10 split) -- the reference combo4 setup uses K=10 for both, so --top_k directly controls
    # both host and the 2K candidate count here.
    from afocus_model_combo4 import AFocusCombo4Model
    host = ChronosBoltModelForForecastingWithRetrieval(config=config, augment=args.afocus_host_aug)
    host.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if args.afocus_host_ckpt:
        _sd = torch.load(args.afocus_host_ckpt, map_location='cpu')
        _msg = host.load_state_dict({k.replace('module.', ''): v for k, v in _sd.items()}, strict=False)
        print(f'[afocus_combo4] host ckpt {args.afocus_host_ckpt}: missing={len(_msg.missing_keys)} unexpected={len(_msg.unexpected_keys)}')
    host.debug_shapes = False
    host._debug_shapes_printed = False
    host.tau = 0.1
    host.ord_margin = 0.0
    model = AFocusCombo4Model(host, beta=args.afocus_beta)
elif args.model in ('ChronosBoltRetrieve', 'Moirai2Retrieve', 'TimesFM25Retrieve') and args.augment_mode == 'afocus_roar':
    # ROAR ablation: same host/candidates/loss as afocus_combo4, only the psi->correction
    # combination logic differs (split p_ij softmax + independent u_i sigmoid gate, instead of
    # combo4's single (2K+1)-way softmax) -- see roar/afocus_model_roar.py docstring.
    # backbone换成Moirai2/TimesFM25时host构造走build_afocus_host，AFocusRoarModel本身不用改。
    from afocus_model_roar import AFocusRoarModel
    host = build_afocus_host(args, config, tag='afocus_roar')
    model = AFocusRoarModel(host, beta=args.afocus_beta, lambda_gate=args.lambda_gate)
elif args.model == 'ChronosBoltRetrieve' and args.augment_mode == 'afocus_roar_v2':
    # ROARHEAD v2: same host/candidates/loss as afocus_roar, only psi_base's gate input widens
    # 3->7 dims (adds proposal-conditioned features) -- see roar/afocus_model_roar_v2.py.
    from afocus_model_roar_v2 import AFocusRoarV2Model
    host = ChronosBoltModelForForecastingWithRetrieval(config=config, augment=args.afocus_host_aug)
    host.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if args.afocus_host_ckpt:
        _sd = torch.load(args.afocus_host_ckpt, map_location='cpu')
        _msg = host.load_state_dict({k.replace('module.', ''): v for k, v in _sd.items()}, strict=False)
        print(f'[afocus_roar_v2] host ckpt {args.afocus_host_ckpt}: missing={len(_msg.missing_keys)} unexpected={len(_msg.unexpected_keys)}')
    host.debug_shapes = False
    host._debug_shapes_printed = False
    host.tau = 0.1
    host.ord_margin = 0.0
    model = AFocusRoarV2Model(host, beta=args.afocus_beta)
elif args.model == 'ChronosBoltRetrieve' and args.augment_mode == 'afocus_roar_gatecal':
    # ROAR v1 gate-only精修：host/psi_cand权重全部来自下面的--init_from_checkpoint整体覆盖
    # (roar_b05_k10_seed2021已训练好的完整checkpoint)，这里的构造只是占位对齐state_dict形状
    # -- 见 roar/afocus_model_roar_gatecal.py 文档。
    from afocus_model_roar_gatecal import AFocusRoarGateCalModel
    host = ChronosBoltModelForForecastingWithRetrieval(config=config, augment=args.afocus_host_aug)
    host.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    host.debug_shapes = False
    host._debug_shapes_printed = False
    host.tau = 0.1
    host.ord_margin = 0.0
    model = AFocusRoarGateCalModel(host, beta=args.afocus_beta)
    model.kappa = args.gatecal_kappa  # kappa is a class attribute, not an __init__ param -- see afocus_model_roar_gatecal.py:89
elif args.model in ('ChronosBoltRetrieve', 'Moirai2Retrieve', 'TimesFM25Retrieve') and args.augment_mode == 'afocus_roar_gatecal_hosthead':
    # ROAR v1 gate+host头联合精修：同样通过--init_from_checkpoint整体覆盖，占位构造对齐state_dict
    # -- 见 roar/afocus_model_roar_gatecal_hosthead.py 文档。
    # backbone换成Moirai2/TimesFM25时host构造走build_afocus_host，AFocusRoarGateCalHostHeadModel
    # 本身不用改；--afocus_host_ckpt在Moirai2/TimesFM25下是必填的(build_afocus_host内部assert)，
    # 跟ChronosBolt不同的是Moirai2/TimesFM25这里不再有"占位构造"的说法(没有base autogluon可以先垫底)，
    # afocus_host_ckpt就是Stage1训出来的那个host，--init_from_checkpoint仍然整体覆盖wrapper自己的
    # psi_cand/gate_ref等参数。
    from afocus_model_roar_gatecal_hosthead import AFocusRoarGateCalHostHeadModel
    host = build_afocus_host(args, config, tag='afocus_roar_gatecal_hosthead')
    model = AFocusRoarGateCalHostHeadModel(host, beta=args.afocus_beta)
    model.kappa = args.gatecal_kappa  # kappa is a class attribute, not an __init__ param -- see afocus_model_roar_gatecal.py:89
elif args.model == 'ChronosBoltRetrieve':
    model = ChronosBoltModelForForecastingWithRetrieval(config=config, augment=args.augment_mode)
    model.debug_shapes = args.debug_shapes
    model._debug_shapes_printed = False
    model.rho1 = args.rho1
    model.rho2 = args.rho2
    model.tau_dis = args.tau_dis
    model.lambda_sem = args.lambda_sem
    model.fusion_mode = args.fusion_mode
    model.head_mode = args.head_mode
    model.disable_ci = args.disable_ci
    model.weight_dis_by_ci = args.weight_dis_by_ci
    model.lambda_ord = args.lambda_ord
    model.lambda_delta = args.lambda_delta
    model.lambda_final_mse = args.lambda_final_mse
    model.huber_kappa = args.huber_kappa
    model.lambda_xcov = args.lambda_xcov
    model.rho3 = args.rho3
    model.rho4 = args.rho4
    model.dis_mode = args.dis_mode
    model.rho_dis_output = args.rho_dis_output
    model.rho_sem = args.rho_sem
    model.rho_xcov = args.rho_xcov
    model.rho_ord = args.rho_ord
    model.ord_margin = args.ord_margin
    model.rho_cos = args.rho_cos
    model.rho_gbal = args.rho_gbal
    model.lambda1 = args.lambda1
    model.lambda2 = args.lambda2
    model.lambda_sep = args.lambda_sep
    model.beta_var = args.beta_var
    model.gamma0_var = args.gamma0_var
    model.tau = args.tau
    model.dyn_margin = args.dyn_margin
    model.aux_loss_detach_ret = args.aux_loss_detach_ret
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer])
    if args.augment_mode == 'moe_disentangle':
        model.init_extra_weights([model.disentangle_gate, model.final_pred_head, model.inv_aux_backproj, model.dyn_aux_backproj])
    if 'gate' in args.augment_mode:
        model.init_extra_weights([model.gate_layer, model.gate_linear1, model.gate_linear2])
    if args.augment_mode == 'idf':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_transition,
            model.dyn_transition,
            model.inv_head,
            model.dyn_head,
            model.final_head,
        ])
    if args.augment_mode in ['idf_branch', 'idf_x']:
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.inv_residual_head,
            model.dyn_pred_head,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_clean_dis':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_ridde_v2':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_trr_dualpath':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.P_inv,
            model.P_dyn,
            model.f_inv,
            model.f_dyn,
        ])
    if args.augment_mode == 'idf_trr_dualpath_learnfuse':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.P_inv,
            model.P_dyn,
            model.f_inv,
            model.f_dyn,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_clean_dis_deepmlp':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_clean_dis_ts3align':
        model.init_extra_weights([
            model.encode_mlp,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_h_linear_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.h_pred_head,
        ])
    if args.augment_mode == 'idf_h_native_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
        ])
    if args.augment_mode == 'idf_y_linear_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.h_pred_head,
            model.y_linear_head,
        ])
    if args.augment_mode == 'idf_branch_gru':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.inv_residual_head,
            model.dyn_gru,
            model.dyn_out_head,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_branch_gru_q':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.inv_residual_head,
            model.dyn_gru,
            model.dyn_out_head,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_dual_direct_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_hidden_proj,
            model.dyn_ret_hidden_proj,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_residual':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head,
        ])
    if args.augment_mode == 'idf_dual_projector':
        model.init_extra_weights([
            model.fuse_gate,
            model.g_inv,
            model.g_dyn,
            model.f_inv,
            model.f_dyn,
        ])
    if args.augment_mode == 'idf_dual_projector_mlp':
        model.init_extra_weights([
            model.retrieved_x_encoder_mlp,
            model.fuse_gate,
            model.g_inv,
            model.g_dyn,
            model.f_inv,
            model.f_dyn,
        ])
elif args.model == 'Moirai2Retrieve':
    # Backbone (Salesforce/moirai-2.0-R-small) is loaded and frozen inside the
    # model class itself (requires_grad=False set in __init__); no checkpoint
    # load / init_extra_weights needed here, unlike ChronosBoltRetrieve.
    # args.augment_mode selects which fusion head to attach on top of the
    # frozen backbone -- both output a 9-quantile distribution (pinball loss),
    # matching the original Chronos-Bolt path's output format.
    if args.augment_mode == 'idf_clean_dis':
        model = Moirai2ModelForForecastingWithRetrieval(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    elif args.augment_mode == 'moe':
        model = Moirai2MoEModelForForecastingWithRetrieval(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    else:
        raise ValueError(
            f"Moirai2Retrieve only supports augment_mode in ['idf_clean_dis', 'moe'], got {args.augment_mode!r}"
        )
elif args.model == 'TimesFM25Retrieve':
    # Backbone (google/timesfm-2.5-200m-pytorch, native timesfm package) is
    # loaded and frozen inside the model class itself; same 9-quantile /
    # pinball-loss heads as Moirai2Retrieve, just wired to a different backbone.
    if args.augment_mode == 'idf_clean_dis':
        model = TimesFM25ModelForForecastingWithRetrieval(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    elif args.augment_mode == 'moe':
        model = TimesFM25MoEModelForForecastingWithRetrieval(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    else:
        raise ValueError(
            f"TimesFM25Retrieve only supports augment_mode in ['idf_clean_dis', 'moe'], got {args.augment_mode!r}"
        )
elif args.model == 'MOMENTRetrieve':
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer, model.project_before_fusion, model.project_after_fusion])
    criterion = nn.MSELoss().to(device)
else:
    print('model error')
    exit()
print(f'{args.model} model loaded')

# RIDDE_检索扰动相对风险目标 (rob_doc)：鲁棒训练接着已经训好的checkpoint(比如
# Pure-ERM+学习型融合头)继续训，不是从随机初始化的P_inv/P_dyn/f_inv/f_dyn/
# final_pred_head/encode_mlp/ret_score_head开始。上面的init_extra_weights只是把
# 这些模块随机初始化好、把shape注册上，这里再整体覆盖成真正训好的权重。
if args.init_from_checkpoint:
    assert args.augment_mode in ('idf_trr_dualpath_learnfuse', 'idf_clean_dis_v3', 'idf_clean_dis_v4', 'afocus_roar_gatecal', 'afocus_roar_gatecal_hosthead'), \
        "--init_from_checkpoint 目前只为鲁棒训练(idf_trr_dualpath_learnfuse)、" \
        "带容差门控重叠(idf_clean_dis_v3)、v4(idf_clean_dis_v4)、ROAR gate-only精修" \
        "(afocus_roar_gatecal)和ROAR gate+host头联合精修(afocus_roar_gatecal_hosthead)这几条路径验证过"
    _init_sd = torch.load(args.init_from_checkpoint, map_location='cpu')
    _missing, _unexpected = model.load_state_dict(_init_sd, strict=False)
    print(f"[RIDDE_rob] 从 {args.init_from_checkpoint} 加载已训练权重完成 "
          f"(missing={len(_missing)}, unexpected={len(_unexpected)})")
    print(f"[RIDDE_rob] missing keys 示例: {_missing[:10]}")
    print(f"[RIDDE_rob] unexpected keys 示例: {_unexpected[:10]}")
    assert len(_unexpected) == 0, \
        "unexpected keys不应该出现——说明checkpoint里有当前模型结构不认识的键，先人工核对再继续"

if args.augment_mode == 'afocus_roar_gatecal':
    assert args.init_from_checkpoint, \
        "--augment_mode afocus_roar_gatecal 必须搭配 --init_from_checkpoint(已训练好的ROAR v1 checkpoint)"
    model.freeze_for_gatecal()  # 必须在checkpoint加载完之后调用，快照的是训练好的psi_base权重
    print('[afocus_roar_gatecal] gate_ref已从加载的checkpoint快照，psi_cand已冻结，只有psi_base可训练')

if args.augment_mode == 'afocus_roar_gatecal_hosthead':
    assert args.init_from_checkpoint, \
        "--augment_mode afocus_roar_gatecal_hosthead 必须搭配 --init_from_checkpoint(已训练好的ROAR v1 checkpoint)"
    model.freeze_for_gatecal_hosthead()  # 必须在checkpoint加载完之后调用，host_ref快照的是训练好的host权重
    print('[afocus_roar_gatecal_hosthead] gate_ref/host_ref已从加载的checkpoint快照，'
          'psi_cand已冻结，host的encode_mlp/ret_score_head/fuse_gate/h_pred_head和psi_base可训练')

model.to(device)
if args.use_multi_gpu:
    args.devices = [int(i) for i in args.devices.split(',')]
    model = nn.DataParallel(model, device_ids=args.devices)

# RIDDE_新版目标函数与最终实验方案 Stage 4：Query-only 参考模型 B(文档4.2节)。
# B 是完全独立、推理时永不更新的一个模型实例——直接加载已经训练好、配方对齐的
# augment_mode='baseline' checkpoint(不走 autogluon_model.pth + init_extra_weights
# 那套"从头初始化训练"的流程，因为 B 不需要在这里训练，只需要产出冻结的参考预测)。
model_B = None
nu_tilde = None
if args.lambda_trr > 0:
    assert args.model == 'ChronosBoltRetrieve', \
        "Stage 4 的 L_TRR 目前只接了 ChronosBoltRetrieve 这条路径"
    assert args.trr_reference_model_path, \
        "--lambda_trr > 0 时必须提供 --trr_reference_model_path(Query-only 参考模型B的checkpoint)"
    config_B = AutoConfig.from_pretrained(args.pretrained_model_path)
    if hasattr(config_B, "chronos_config"):
        config_B.chronos_config["context_length"] = args.context_length
        config_B.chronos_config["prediction_length"] = args.prediction_length
    model_B = ChronosBoltModelForForecastingWithRetrieval(config=config_B, augment='baseline')
    model_B.debug_shapes = False
    model_B._debug_shapes_printed = True
    b_state_dict = torch.load(args.trr_reference_model_path, map_location='cpu')
    missing, unexpected = model_B.load_state_dict(b_state_dict, strict=False)
    print(f"[Stage4] 参考模型B从 {args.trr_reference_model_path} 加载完成 "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model_B.to(device)
    model_B.eval()
    for p in model_B.parameters():
        p.requires_grad = False
    # nu_tilde 是方案文档4.3节 CVaR 的分位变量 nu 的 softplus 前参数化，是一个跟
    # model 参数完全独立的标量，需要手动加进优化器的参数列表(不在 model.parameters()里)。
    nu_tilde = nn.Parameter(torch.tensor(float(args.nu_init), dtype=torch.float32, device=device))

# RIDDE_检索扰动相对风险目标(rob_doc)："Query-only模型仅用于提供参考预测损失"。
# 加载方式完全照抄上面model_B那一套(augment='baseline'，eval，requires_grad=False)，
# 只是checkpoint换成TrueBase(script/pretrain_truebase.sh训出来的那个)，不是TRR的B。
model_f0 = None
if args.lambda_rob > 0:
    assert args.model == 'ChronosBoltRetrieve', "L_rob目前只接了ChronosBoltRetrieve这条路径"
    assert args.f0_checkpoint_path, "--lambda_rob > 0 时必须提供 --f0_checkpoint_path(TrueBase/Query-only参考模型checkpoint)"
    config_f0 = AutoConfig.from_pretrained(args.pretrained_model_path)
    if hasattr(config_f0, "chronos_config"):
        config_f0.chronos_config["context_length"] = args.context_length
        config_f0.chronos_config["prediction_length"] = args.prediction_length
    model_f0 = ChronosBoltModelForForecastingWithRetrieval(config=config_f0, augment='baseline')
    model_f0.debug_shapes = False
    model_f0._debug_shapes_printed = True
    f0_state_dict = torch.load(args.f0_checkpoint_path, map_location='cpu')
    missing_f0, unexpected_f0 = model_f0.load_state_dict(f0_state_dict, strict=False)
    print(f"[RIDDE_rob] f_0(TrueBase)从 {args.f0_checkpoint_path} 加载完成 "
          f"(missing={len(missing_f0)}, unexpected={len(unexpected_f0)})")
    model_f0.to(device)
    model_f0.eval()
    for p in model_f0.parameters():
        p.requires_grad = False

if args.augment_mode == 'afocus_roar_gatecal_hosthead':
    # host头给一个单独的(更小的)学习率，psi_base沿用--learning_rate。用name.startswith('host.')
    # 精确限定，不能只看HOST_HEAD_MODULE_NAMES这些子串是否出现在参数名里——否则host_ref(冻结、
    # 无梯度)里同名的参数也会被混进这个参数组，虽然因为requires_grad=False、.grad恒为None，
    # optimizer.step()本身会自动跳过它们、不影响训练结果，但这里保留这个筛选是为了代码逻辑自洽，
    # 不依赖"反正冻结了所以无所谓"这个巧合。
    from afocus_model_roar_gatecal_hosthead import HOST_HEAD_MODULE_NAMES
    _head_params = [p for n, p in model.named_parameters()
                     if n.startswith('host.') and any(h in n for h in HOST_HEAD_MODULE_NAMES)]
    _gate_params = [p for n, p in model.named_parameters() if n.startswith('psi_base.')]
    print(f'[afocus_roar_gatecal_hosthead] host头参数组: {len(_head_params)}个 (lr={args.host_head_lr}), '
          f'gate参数组: {len(_gate_params)}个 (lr={args.learning_rate})')
    optim_param_groups = [
        {'params': _head_params, 'lr': args.host_head_lr},
        {'params': _gate_params, 'lr': args.learning_rate},
    ]
    params = _head_params + _gate_params  # 保持params是flat tensor列表，供下面clip_grad_norm_(params,...)用
                                            # (torch.nn.utils.clip_grad_norm_不接受optimizer风格的param-group字典)
else:
    optim_param_groups = params = list(model.parameters())
    if nu_tilde is not None:
        params = params + [nu_tilde]
        optim_param_groups = params

if args.optimizer == 'adam':
    model_optim = torch.optim.Adam(optim_param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    model_optim = torch.optim.AdamW(optim_param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

# freeze params
if args.freeze_chronos_bolt:
    layers_to_unfreeze = ['gate_layer', 'encode_mlp', 'mha', 'ffn']
    if args.model in ('Moirai2Retrieve', 'TimesFM25Retrieve') and args.augment_mode == 'moe':
        # Moirai2MoEModelForForecastingWithRetrieval / TimesFM25MoEModelForForecastingWithRetrieval
        # add their own quantile head (quantile_pred_head) instead of reusing a native backbone
        # head, unlike the Chronos-Bolt 'moe' path -- the base layers_to_unfreeze list above
        # doesn't cover it.
        layers_to_unfreeze.append('quantile_pred_head')
    if args.augment_mode == 'moe3':
        if args.model == 'ChronosBoltRetrieve':
            layers_to_unfreeze.append('output_patch_embedding')
        elif args.model == 'MOMENTRetrieve':
            # import pdb; pdb.set_trace()
            layers_to_unfreeze.append('head')
    elif args.augment_mode == 'gate':
        layers_to_unfreeze.append('gate_linear1')
        layers_to_unfreeze.append('gate_linear2')
    elif args.augment_mode == 'moe_disentangle':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'mha',
            'ffn',
            'gate_layer',
            'disentangle_gate',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_transition',
            'dyn_transition',
            'inv_head',
            'dyn_head',
            'final_head',
        ])
    elif args.augment_mode in ['idf_branch', 'idf_x']:
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'inv_residual_head',
            'dyn_pred_head',
            'final_pred_head',
        ])
    elif args.augment_mode in ('idf_clean_dis', 'idf_clean_dis_v3'):
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_clean_dis_v4':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_ridde_v2':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_clean_dis_deepmlp':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_clean_dis_ts3align':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
            'inv_aux_backproj',
            'dyn_aux_backproj',
        ])
    elif args.augment_mode == 'idf_h_linear_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'h_pred_head',
        ])
    elif args.augment_mode == 'idf_h_native_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
        ])
    elif args.augment_mode == 'idf_y_linear_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'h_pred_head',
            'y_linear_head',
        ])
    elif args.augment_mode == 'idf_branch_gru':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'inv_residual_head',
            'dyn_gru',
            'dyn_out_head',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_branch_gru_q':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'inv_residual_head',
            'dyn_gru',
            'dyn_out_head',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_dual_direct_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_hidden_proj',
            'dyn_ret_hidden_proj',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_residual':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head',
        ])
    elif args.augment_mode == 'idf_dual_projector':
        layers_to_unfreeze.extend([
            'fuse_gate',
            'g_inv',
            'g_dyn',
            'f_inv',
            'f_dyn',
        ])
    elif args.augment_mode == 'idf_dual_projector_mlp':
        layers_to_unfreeze.extend([
            'retrieved_x_encoder_mlp',
            'fuse_gate',
            'g_inv',
            'g_dyn',
            'f_inv',
            'f_dyn',
        ])
    elif args.augment_mode == 'idf_trr_dualpath':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'P_inv',
            'P_dyn',
            'f_inv',
            'f_dyn',
        ])
    elif args.augment_mode == 'idf_trr_dualpath_learnfuse':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'P_inv',
            'P_dyn',
            'f_inv',
            'f_dyn',
            'final_pred_head',
        ])
    elif args.augment_mode == 'baseline':
        # No-Retrieval Base(方案C):augment_mode='baseline' 的前向传播完全不碰检索,
        # 只是把 output_patch_embedding(sequence_output) 作为原生 Chronos-Bolt 预测头
        # 跑一遍。如果只用默认的 layers_to_unfreeze(gate_layer/encode_mlp/mha/ffn ——
        # 这条代码路径里一个都不存在),这个分支可训练参数为零,等于一个从没在这个
        # 数据集上微调过的纯 zero-shot 预测器,而 idf_clean_dis 的融合头是训练过的 ——
        # 这不公平。只放开 output_patch_embedding,让它享受和 RAG 融合头一样的训练机会
        # (同样的数据、步数、学习率、优化器,冻结主干),只是不给检索输入。
        layers_to_unfreeze.append('output_patch_embedding')

    if args.lambda_rob > 0:
        # RIDDE_检索扰动相对风险目标(rob_doc)："参考检索权重由鲁棒训练开始前的RIDDE
        # 检查点生成,冻结该检查点中用于检索与注意力计算的模块"——具体做法就是让
        # encode_mlp/ret_score_head在这个阶段不进入可训练列表,不需要额外维护一份
        # 单独的frozen model实例。
        _before_filter = set(layers_to_unfreeze)
        layers_to_unfreeze = [l for l in layers_to_unfreeze if l not in ('encode_mlp', 'ret_score_head')]
        print(f'[RIDDE_rob] lambda_rob>0: 从可训练列表移除 {_before_filter - set(layers_to_unfreeze)}(参考权重omega固定,不再训练)')

    if args.augment_mode in ('afocus', 'afocus_combo4', 'afocus_roar', 'afocus_roar_v2'):
        layers_to_unfreeze = ['psi_']  # method A (all variants): only psi is trainable, host fully frozen
    if args.augment_mode == 'afocus_roar_gatecal':
        layers_to_unfreeze = ['psi_base']  # gate-only精修：host+psi_cand冻结，只放开gate(psi_base)。
        # 注意 gate_ref 快照submodule特意不叫psi_base_ref之类的名字——'psi_base' in name 是子串匹配，
        # 叫psi_base_ref会被这里意外一起放开，所以在afocus_model_roar_gatecal.py里叫gate_ref。
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze the specified layers
    for name, param in model.named_parameters():
        if name.startswith('backbone.'):
            # Moirai2's own backbone submodule names its internal transformer FFN
            # sublayer 'ffn' (uni2ts TransformerEncoderLayer.ffn), which collides
            # with the substring match below and would otherwise unfreeze it --
            # the Moirai2 backbone (everything under the 'backbone.' prefix, see
            # models/Moirai2.py) must always stay frozen regardless of
            # layers_to_unfreeze.
            param.requires_grad = False
        else:
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

if args.augment_mode == 'afocus_roar_gatecal_hosthead':
    # 独立于--freeze_chronos_bolt之外的无条件冻结/解冻逻辑(不能依赖上面那个if args.freeze_
    # chronos_bolt:代码块——之前afocus/combo4/roar/roar_v2/gatecal这几个变体"只有psi_*可训练"
    # 能生效纯属巧合：AFocusModel.__init__本来就把host整体冻结了，psi_cand/psi_base构造时
    # nn.Linear默认requires_grad=True没被碰过，凑巧等于想要的最终状态；这次要主动把host被冻结
    # 的头部分重新解冻，这个操作只存在于--freeze_chronos_bolt那个大if块内部——冒烟测试第一次跑
    # 的时候没传这个flag，导致host头10个参数实际上全程没有解冻，只有psi_base真正在训(见调试记
    # 录：Trainable parameters只打出4个而不是14个)。这里改成完全不依赖那个flag、无条件执行，
    # 避免同一个坑再次发生。
    from afocus_model_roar_gatecal_hosthead import HOST_HEAD_MODULE_NAMES
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        p.requires_grad = ('psi_base' in name) or \
            (name.startswith('host.') and any(h in name for h in HOST_HEAD_MODULE_NAMES))
    # 上面这行对host_ref同名参数也会命中(同一个子串坑，'encode_mlp' in 'host_ref.encode_mlp...'
    # 同样是True)，显式重新冻结回去——不是防御性冗余，跳过会导致候选构造/gate输入特征/r_ref/
    # y_hat_ref这些"应该完全不变"的量偷偷跟着训练漂移。
    for p in model.host_ref.parameters():
        p.requires_grad = False
    print('[afocus_roar_gatecal_hosthead] host头(encode_mlp/ret_score_head/fuse_gate/h_pred_head)'
          '+psi_base已解冻，host主干/psi_cand/gate_ref/host_ref保持冻结(不依赖--freeze_chronos_bolt)')

trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
print('Trainable parameters:')
for name in trainable_param_names:
    print(name)

model.train()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

# retrieval already done, do not need to load the embedding model
embedding_model = None

# load retriever
retriever = Retriever_for_pretrain(
    retrieval_database_path=args.retrieval_database_path,
    dimension=768,
    embedding_model=embedding_model,
)
retriever.build_index()

## load data
if args.lambda_trr > 0:
    # RIDDE_新版目标函数与最终实验方案 Stage 4 (5.2/5.3节)：环境平衡采样。
    # !!!环境定义说明(2026-09-09 更新：重叠窗口泄露诊断结果，细节见 dataset.py 里
    # EnvironmentBalancedPretrainDataset 类的 docstring)：现有 pretrain_pairs_ctx512/
    # 下的30个parquet分片每一行的 start 字段都是统一的1970-01-01占位值，没有真实
    # 时间戳也没有item_id，无法构造方案5.2节字面要求的"连续时间块"环境。但抽样诊断
    # (chunk内部滑窗重合比例99.3%~100%、6个chunk间样本级重叠数=0)证实这30个分片是
    # 按原始序列/来源对象切分的，不存在重叠窗口跨环境泄露——"环境"是干净的序列/来源
    # 对象级固定分组，不是方案文档7.5节的"random blocks"随机负对照，但也还不是
    # "chronological blocks"完整方法(chunk之间没有已知时间顺序)。这次结果可以作为
    # "环境定义为序列/来源对象级(非时间连续)"的正式CVaR结果去汇报。
    all_chunk_files = sorted(Path(args.data_path).glob('*.parquet'))
    env_files = [f for f in all_chunk_files if f.name not in STAGE4_TINY_CHUNK_BASENAMES]
    print(f"[Stage4] 环境(序列/来源对象级固定分组=parquet分片)数量: {len(env_files)} "
          f"(已排除小分片: {sorted(STAGE4_TINY_CHUNK_BASENAMES)})")
    effective_batch_size = args.env_group_size * args.env_batch_size
    if effective_batch_size != args.batch_size:
        print(f"[Stage4] 注意：env_group_size({args.env_group_size}) * "
              f"env_batch_size({args.env_batch_size}) = {effective_batch_size}，"
              f"跟 --batch_size={args.batch_size} 不一致。实际训练用的 batch size "
              f"以 env_group_size*env_batch_size 为准(每个batch必须严格是G组m个连续"
              f"同环境样本，DataLoader 的 batch_size 强制改成这个值)，--batch_size "
              f"这个参数在 Stage 4 下只影响 model_id 里的命名，不影响实际训练。")
    dataset = EnvironmentBalancedPretrainDataset(
        env_files=env_files,
        drop_prob=args.drop_prob,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        top_k=args.top_k,
        env_group_size=args.env_group_size,
        env_batch_size=args.env_batch_size,
        env_shuffle_buffer_length=args.env_shuffle_buffer_length,
    )
    train_loader = DataLoader(dataset, batch_size=effective_batch_size, num_workers=0)
else:
    dataset = CustomPretrainDataset(
        args.data_path,
        retriever=retriever,
        mode='training',
        drop_prob=args.drop_prob,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        retrieve_lookback_length=args.retrieve_lookback_length,
        top_k=args.top_k,
    ).shuffle(shuffle_buffer_length=args.shuffle_buffer_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

if args.augment_mode in ('afocus', 'afocus_combo4', 'afocus_roar', 'afocus_roar_v2'):
    # t_h, mu_h, mu_g, c = beta*mu_hg 全部在训练正式开始前，从一个独立的迭代器过
    # n_batches 个 batch 固定下来（AFocusModel.calibrate 内部保存/恢复了 RNG 状态，
    # 不会扰动主训练 loader 的采样顺序）。四种 weight_mode 都走同一套标定流程 ——
    # uniform/joint 不需要 mu_h/mu_g/c，但 hard/gain 需要，而且 t_h 必须跟 calibrate()
    # 里算 mu_h/mu_g 用的是同一个值，不能分开算，否则权重公式和标定值会悄悄脱节。
    _calib_loader = DataLoader(CustomPretrainDataset(args.data_path, retriever=retriever, mode='training',
        drop_prob=args.drop_prob, context_length=args.context_length, prediction_length=args.prediction_length,
        retrieve_lookback_length=args.retrieve_lookback_length, top_k=args.top_k
        ).shuffle(shuffle_buffer_length=args.shuffle_buffer_length), batch_size=args.batch_size, num_workers=0)
    model.to(device)
    _stats = model.calibrate(_calib_loader, retriever, device, n_batches=args.th_calib_batches)
    print(f"[afocus] weight_mode={args.afocus_weight_mode} calibrated over {_stats['n']} samples: "
          f"t_h={_stats['t_h']:.6f} mu_h={_stats['mu_h']:.6f} mu_g={_stats['mu_g']:.6f} "
          f"mu_hg=c={_stats['c']:.6f}")
    del _calib_loader


## train
iter_count = 0
train_loss = []
try:
    total_steps = min(len(train_loader), args.train_steps)
except TypeError:
    total_steps = args.train_steps
pbar = tqdm(enumerate(train_loader), total=total_steps)
for i, batch in pbar:
    if i >= args.train_steps:
        print('training finished')
        break
    model.train()
    iter_count += 1
    model_optim.zero_grad()
    retrieved_seqs = torch.tensor(retriever.whole_seq[batch['indices']])
    if args.kill_retrieval:
        # Break the query<->neighbor correspondence while keeping retrieved_seqs
        # on-distribution (still real windows from the same KB, just not this
        # query's actual nearest neighbors). Zeroing instead would feed the
        # fusion head an input it never saw during real idf_clean_dis training,
        # which is a confound of its own.
        perm = torch.randperm(retrieved_seqs.shape[0])
        retrieved_seqs = retrieved_seqs[perm]

    if not args.use_multi_gpu:
        batch['x'] = batch['x'].float().to(device)
        batch['y'] = batch['y'].float().to(device)
        batch['distances'] = batch['distances'].float().to(device)
        retrieved_seqs = retrieved_seqs.float().to(device)
    if args.model == 'ChronosBoltRetrieve':
        outputs = model(context = batch['x'].float(),
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(),
                        distances = batch['distances'].float())                  # ChronosBoltOutput
    elif args.model == 'Moirai2Retrieve':
        outputs = model(context = batch['x'].float(),
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(),
                        distances = batch['distances'].float())                  # Moirai2RiddeOutput / Moirai2MoEOutput
    elif args.model == 'TimesFM25Retrieve':
        outputs = model(context = batch['x'].float(),
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(),
                        distances = batch['distances'].float())                  # TimesFM25RiddeOutput / TimesFM25MoEOutput
    elif args.model == 'MOMENTRetrieve':
        outputs = model(x_enc=batch['x'].float().unsqueeze(1), retrieved_seq=retrieved_seqs.float())
        outputs = outputs.forecast.squeeze(1)                                                     
        loss = criterion(outputs, batch['y'].float())
    else:
        print('model error')
    if args.model == 'MOMENTRetrieve':
        pass
    else:
        loss = outputs.loss
    loss = loss.mean()

    # FINAL_MSE_OBJECTIVE_PATCH_V1: compare unweighted component gradients on the same batch.
    if args.calibrate_final_mse_grad_norm and i == 0:
        if args.model != 'ChronosBoltRetrieve' or outputs.loss_final_mse is None:
            raise RuntimeError("final-MSE calibration requires ChronosBoltRetrieve with a target")
        if not (args.final_mse_grad_ratio > 0):
            raise ValueError("--final_mse_grad_ratio must be positive")

        trainable_params = [p for p in params if p.requires_grad]

        def _current_grad_l2():
            grad_parts = [
                p.grad.detach().float().norm(2)
                for p in trainable_params
                if p.grad is not None
            ]
            if not grad_parts:
                return loss.new_zeros(())
            return torch.stack(grad_parts).norm(2)

        model_optim.zero_grad()
        outputs.loss_forecast.backward(retain_graph=True)
        grad_norm_pred = _current_grad_l2()
        model_optim.zero_grad()
        outputs.loss_final_mse.backward()
        grad_norm_final_mse = _current_grad_l2()
        model_optim.zero_grad()

        eps_cal = 1e-12
        pred_value = float(grad_norm_pred.item())
        mse_value = float(grad_norm_final_mse.item())
        if not (pred_value > eps_cal and mse_value > eps_cal):
            raise RuntimeError(
                f"cannot calibrate: pinball_grad={pred_value:.6g}, "
                f"final_mse_grad={mse_value:.6g}"
            )
        suggested = args.final_mse_grad_ratio * pred_value / mse_value
        print(f"[final_mse_calibration] loss_forecast={outputs.loss_forecast.item():.8f} "
              f"loss_final_mse={outputs.loss_final_mse.item():.8f}")
        print(f"[final_mse_calibration] grad_pinball={pred_value:.8g} "
              f"grad_final_mse={mse_value:.8g} target_ratio={args.final_mse_grad_ratio:.6g}")
        print(f"FINAL_MSE_SUGGESTED_LAMBDA={suggested:.10g}")
        raise SystemExit(0)

    if args.calibrate_grad_norm and i == 0:
        trainable_params = [p for p in params if p.requires_grad]
        model_optim.zero_grad()
        outputs.loss_forecast.backward(retain_graph=True)
        grad_norm_pred = torch.sqrt(sum(
            (p.grad.detach() ** 2).sum() for p in trainable_params if p.grad is not None
        ))
        model_optim.zero_grad()
        outputs.loss_delta.backward()
        grad_norm_delta = torch.sqrt(sum(
            (p.grad.detach() ** 2).sum() for p in trainable_params if p.grad is not None
        ))
        model_optim.zero_grad()
        ratio = (grad_norm_delta / grad_norm_pred).item()
        print(f"[calibrate_grad_norm] loss_forecast={outputs.loss_forecast.item():.6f} "
              f"loss_delta(raw,未乘lambda_delta)={outputs.loss_delta.item():.6f}")
        print(f"[calibrate_grad_norm] ||grad(loss_forecast)||={grad_norm_pred.item():.6f} "
              f"||grad(loss_delta)||={grad_norm_delta.item():.6f} ratio={ratio:.6f}")
        for target_ratio in (0.05, 0.10):
            suggested = target_ratio / ratio
            print(f"[calibrate_grad_norm] 目标梯度范数占比={target_ratio:.2f} "
                  f"-> 建议lambda_delta≈{suggested:.6f}")
        raise SystemExit(0)

    # RIDDE_新版目标函数与最终实验方案 Stage 4 (4.2-4.6节)：L_TRR/CVaR。lambda_trr=0
    # (Stage 3 及更早的所有 augment_mode)完全跳过这一段，行为跟改动前一模一样。
    # 这里的 loss 到这一步为止(对 idf_trr_dualpath 来说)已经是模型内部算好的
    # L_pred + lambda_sep*L_sep(方案文档4.6节的前两项+第三项)，下面只需要再加上
    # lambda_trr*L_TRR 这一项，不需要改模型内部任何东西。
    loss_trr = loss.new_zeros(())
    nu_value = loss.new_zeros(())
    r_e_mean = loss.new_zeros(())
    r_e_max = loss.new_zeros(())
    if args.lambda_trr > 0:
        assert 'env_id' in batch, \
            "lambda_trr>0 时训练数据必须来自 EnvironmentBalancedPretrainDataset(带env_id字段)"
        assert outputs.loss_forecast_per_sample is not None, \
            "主模型没有返回 loss_forecast_per_sample，检查 ChronosBolt.py 的改动是否生效"
        with torch.no_grad():
            outputs_B = model_B(
                context=batch['x'].float(),
                target=batch['y'].float(),
                retrieved_seq=retrieved_seqs.float(),
                distances=batch['distances'].float(),
            )
        per_sample_ret = outputs.loss_forecast_per_sample          # (Batch,) 保留计算图，用于反传到Θ
        per_sample_ref = outputs_B.loss_forecast_per_sample.detach()  # B 冻结，双重保险再 detach 一次
        env_id = batch['env_id'].to(device)
        eps_trr = 1e-4
        r_e_list = []
        for e in torch.unique(env_id):
            mask = (env_id == e)
            R_e_ret = per_sample_ret[mask].mean()
            R_e_0 = per_sample_ref[mask].mean()
            delta_e = torch.log((R_e_ret + eps_trr) / (R_e_0 + eps_trr))
            r_e_list.append(torch.clamp(delta_e, min=0.0))
        r_e_stack = torch.stack(r_e_list)  # (实际到场的环境数,) 正常应等于 env_group_size
        nu_value = F.softplus(nu_tilde)
        loss_trr = nu_value + (
            1.0 / ((1.0 - args.cvar_alpha) * r_e_stack.shape[0])
        ) * torch.clamp(r_e_stack - nu_value, min=0.0).sum()
        with torch.no_grad():
            r_e_mean = r_e_stack.mean()
            r_e_max = r_e_stack.max()
        loss = loss + args.lambda_trr * loss_trr

    # RIDDE_检索扰动相对风险目标 (rob_doc): L = L_pred + lambda_rob*L_rob。lambda_rob=0
    # 完全跳过，行为和改动前一模一样。
    loss_rob = loss.new_zeros(())
    rob_active_frac = loss.new_zeros(())
    rob_kl_max = loss.new_zeros(())
    rob_f0_gap_diag = loss.new_zeros(())
    if args.lambda_rob > 0:
        from utils.ridde_robust_loss import RobustConfig, robust_relative_objective, squared_error
        assert outputs.aux_e_q_rob is not None and outputs.aux_retrieved_y_enc_rob is not None \
            and outputs.aux_omega_rob is not None and outputs.aux_target_rob is not None, \
            "lambda_rob>0 要求 augment_mode=idf_trr_dualpath_learnfuse 且模型已吐出 aux_*_rob 字段,检查STEP1的patch是否生效"
        _m = model.module if hasattr(model, 'module') else model
        central_idx = torch.abs(_m.quantiles - 0.5).argmin()
        e_q_rob = outputs.aux_e_q_rob.detach()
        retrieved_y_enc_rob = outputs.aux_retrieved_y_enc_rob.detach()
        omega_rob = outputs.aux_omega_rob.detach()
        target_point = outputs.aux_target_rob.detach().squeeze(1)

        with torch.no_grad():
            clean_pred = _m.dualpath_predict_with_pi(e_q_rob, retrieved_y_enc_rob, omega_rob)[:, central_idx]
            baseline_loss = squared_error(clean_pred, target_point, reduction='sum')
        assert torch.isfinite(baseline_loss).all() and (baseline_loss >= 0).all(), \
            "baseline_loss(ω_i下的正常clean预测)出现非法值(非有限或负数)，检查dualpath_predict_with_pi/omega_rob"

        # 诊断用，不参与loss：仍跑一次f_0(Query-only)前向，只用来记录"相对完全不用检索"的差距，
        # 2026-09-12改动：baseline从f_0换成模型自身clean loss，详见项目文档
        # ridde-robust-relative-risk-l_rob-integration.md 的"Hinge几乎不激活问题排查"一节
        with torch.no_grad():
            outputs_f0 = model_f0(
                context=batch['x'].float(),
                target=batch['y'].float(),
                retrieved_seq=retrieved_seqs.float(),
                distances=batch['distances'].float(),
            )
            f0_point = outputs_f0.quantile_preds[:, central_idx].detach()
            f0_loss_diag = squared_error(f0_point, target_point, reduction='sum')
            rob_f0_gap_diag = (f0_loss_diag - baseline_loss).mean()

        def _predict_pi(pi):
            fused = _m.dualpath_predict_with_pi(e_q_rob, retrieved_y_enc_rob, pi)
            return fused[:, central_idx]

        # 下面这几个轻量头临时切eval()只是为了让predict()在omega/pi_star之间反复
        # 调用时结果确定(没有dropout随机性)，这是把模块返回的loss拆成clean_i+excess
        # 两部分再精确相减、避免L_pred重复计入的前提；eval()不冻结参数，梯度照样
        # 反传，train()/eval()状态在这段计算结束后立刻还原，不影响这个step其余部分。
        # 整个模型做一次eval()/train()切换,而不是手动维护P_inv/P_dyn/f_inv/f_dyn/
        # final_pred_head这份列表——dualpath_predict_with_pi只会碰到_m的这几个头,
        # 不会重新跑backbone/encode_mlp/ret_score_head(它们本来就冻结),所以切eval()
        # 对这次计算范围之外的部分零影响;以后这几个头里如果加了dropout/batchnorm,
        # 也会被自动覆盖到,不用回来同步这份列表。eval()不冻结参数,梯度照样反传。
        _rob_was_training = _m.training
        _m.eval()
        try:
            cfg = RobustConfig(rho=1.0, radius=args.kl_radius, inner_steps=args.rob_inner_steps,
                               step_size=args.rob_step_size, random_restarts=args.rob_random_restarts,
                               bisection_steps=args.rob_bisection_steps, reduction='sum')
            loss_combo, rob_info = robust_relative_objective(
                predict=_predict_pi, target=target_point, omega=omega_rob,
                baseline_loss=baseline_loss, cfg=cfg,
            )
            # loss_combo = clean_i.mean() + 1.0*excess.mean()(cfg.rho=1固定，只是为了
            # 触发模块内部的对抗搜索，不是真正的rho_rob)。clean_i是"平方L2+中位数分位
            # 数"版本的L_pred替身，跟本文件已有的loss_forecast(pinball loss)是同一个
            # 角色、不同度量，不能重复计入——用同样的predict/target/reduction重新算一
            # 次clean_i_grad，代数上跟模块内部的clean_i完全相等，减掉后只剩纯L_rob项。
            clean_i_grad = squared_error(_predict_pi(omega_rob), target_point, reduction='sum')
        finally:
            _m.train(_rob_was_training)

        loss_rob = loss_combo - clean_i_grad.mean()
        with torch.no_grad():
            rob_active_frac = rob_info['active_fraction']
            rob_kl_max = rob_info['kl'].max()
        loss = loss + args.lambda_rob * loss_rob

    if args.model == 'ChronosBoltRetrieve':
        loss_forecast = outputs.loss_forecast.mean() if outputs.loss_forecast is not None else loss
        loss_final_mse = outputs.loss_final_mse.mean() if outputs.loss_final_mse is not None else loss.new_zeros(())
        loss_cons = outputs.loss_cons.mean() if outputs.loss_cons is not None else loss.new_zeros(())
        loss_smooth = outputs.loss_smooth.mean() if outputs.loss_smooth is not None else loss.new_zeros(())
        loss_inv = outputs.loss_inv.mean() if outputs.loss_inv is not None else loss.new_zeros(())
        loss_dis = outputs.loss_dis.mean() if outputs.loss_dis is not None else loss.new_zeros(())
        loss_dis_output = outputs.loss_dis_output.mean() if outputs.loss_dis_output is not None else loss.new_zeros(())
        loss_ret = outputs.loss_ret.mean() if outputs.loss_ret is not None else loss.new_zeros(())
        loss_dyn = outputs.loss_dyn.mean() if outputs.loss_dyn is not None else loss.new_zeros(())
        loss_sem = outputs.loss_sem.mean() if outputs.loss_sem is not None else loss.new_zeros(())
        loss_xcov = outputs.loss_xcov.mean() if outputs.loss_xcov is not None else loss.new_zeros(())
        loss_ord = outputs.loss_ord.mean() if outputs.loss_ord is not None else loss.new_zeros(())
        loss_cos = outputs.loss_cos.mean() if outputs.loss_cos is not None else loss.new_zeros(())
        loss_gbal = outputs.loss_gbal.mean() if outputs.loss_gbal is not None else loss.new_zeros(())
        diag_cos_sim = outputs.diag_cos_sim.mean() if outputs.diag_cos_sim is not None else loss.new_zeros(())
        diag_gamma_mean = outputs.diag_gamma_mean.mean() if outputs.diag_gamma_mean is not None else loss.new_zeros(())
        diag_gamma_sat_frac = outputs.diag_gamma_sat_frac.mean() if outputs.diag_gamma_sat_frac is not None else loss.new_zeros(())
        diag_roughness_inv = outputs.diag_roughness_inv.mean() if outputs.diag_roughness_inv is not None else loss.new_zeros(())
        diag_roughness_dyn = outputs.diag_roughness_dyn.mean() if outputs.diag_roughness_dyn is not None else loss.new_zeros(())
        diag_energy_share_inv = outputs.diag_energy_share_inv.mean() if outputs.diag_energy_share_inv is not None else loss.new_zeros(())
        aux_loss_enabled = getattr(outputs, 'aux_loss_enabled', False)
    else:
        loss_forecast = loss
        loss_final_mse = loss.new_zeros(())
        loss_cons = loss.new_zeros(())
        loss_smooth = loss.new_zeros(())
        loss_inv = loss.new_zeros(())
        loss_dis = loss.new_zeros(())
        loss_dis_output = loss.new_zeros(())
        loss_ret = loss.new_zeros(())
        loss_dyn = loss.new_zeros(())
        loss_sem = loss.new_zeros(())
        loss_xcov = loss.new_zeros(())
        loss_ord = loss.new_zeros(())
        loss_cos = loss.new_zeros(())
        loss_gbal = loss.new_zeros(())
        diag_cos_sim = loss.new_zeros(())
        diag_gamma_mean = loss.new_zeros(())
        diag_gamma_sat_frac = loss.new_zeros(())
        diag_roughness_inv = loss.new_zeros(())
        diag_roughness_dyn = loss.new_zeros(())
        diag_energy_share_inv = loss.new_zeros(())
        aux_loss_enabled = False
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        log_payload = {
            'loss': loss.item(),
            'loss_forecast': loss_forecast.item(),
            'loss_final_mse': loss_final_mse.item(),
            'loss_cons': loss_cons.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_inv': loss_inv.item(),
            'loss_dis': loss_dis.item(),
            'loss_dis_output': loss_dis_output.item(),
            'loss_ret': loss_ret.item(),
            'loss_dyn': loss_dyn.item(),
            'loss_sem': loss_sem.item(),
            'loss_xcov': loss_xcov.item(),
            'loss_ord': loss_ord.item(),
            'loss_cos': loss_cos.item(),
            'loss_gbal': loss_gbal.item(),
            'diag_cos_sim': diag_cos_sim.item(),
            'diag_gamma_mean': diag_gamma_mean.item(),
            'diag_gamma_sat_frac': diag_gamma_sat_frac.item(),
            'diag_roughness_inv': diag_roughness_inv.item(),
            'diag_roughness_dyn': diag_roughness_dyn.item(),
            'diag_energy_share_inv': diag_energy_share_inv.item(),
            'lr': model_optim.param_groups[0]['lr']
            }
        if args.lambda_trr > 0:
            log_payload.update({
                'loss_trr': loss_trr.item(),
                'nu': nu_value.item(),
                'r_e_mean': r_e_mean.item(),
                'r_e_max': r_e_max.item(),
            })
        if args.lambda_rob > 0:
            log_payload.update({
                'loss_rob': loss_rob.item(),
                'rob_active_frac': rob_active_frac.item(),
                'rob_kl_max': rob_kl_max.item(),
                'rob_f0_gap_diag': rob_f0_gap_diag.item(),
            })
        wandb.log(log_payload)

    postfix = {
        'total': round(loss.item(), 4),
        'f': round(loss_forecast.item(), 4),
    }
    if args.lambda_final_mse > 0:
        postfix.update({'final_mse': round(loss_final_mse.item(), 4)})
    if args.lambda_trr > 0:
        postfix.update({
            'trr': round(loss_trr.item(), 4),
            'nu': round(nu_value.item(), 4),
            'r_max': round(r_e_max.item(), 4),
        })
    if args.lambda_rob > 0:
        postfix.update({
            'rob': round(loss_rob.item(), 4),
            'rob_af': round(rob_active_frac.item(), 4),
            'rob_kl': round(rob_kl_max.item(), 4),
            'rob_gap': round(rob_f0_gap_diag.item(), 4),
        })
    if args.augment_mode == 'idf_ridde_v2':
        postfix.update({
            'sem': round(loss_sem.item(), 4),
            'xcov': round(loss_xcov.item(), 4),
            'ord': round(loss_ord.item(), 4),
            'l_cos': round(loss_cos.item(), 4),
            'l_gbal': round(loss_gbal.item(), 4),
            'cos': round(diag_cos_sim.item(), 4),
            'g_mean': round(diag_gamma_mean.item(), 4),
            'g_sat': round(diag_gamma_sat_frac.item(), 4),
        })
    if aux_loss_enabled:
        if args.augment_mode in ['idf_dual_projector', 'idf_dual_projector_mlp']:
            postfix.update({
                'cons': round(loss_cons.item(), 4),
                'smooth': round(loss_smooth.item(), 4),
                'dis': round(loss_dis.item(), 4),
                'dyn': round(loss_dyn.item(), 4),
            })
        else:
            postfix.update({
                'cons': round(loss_cons.item(), 4),
                'smooth': round(loss_smooth.item(), 4),
                'inv': round(loss_inv.item(), 4),
                'dis': round(loss_dis.item(), 4),
                'dis_out': round(loss_dis_output.item(), 4),
                'ret': round(loss_ret.item(), 4),
                'dyn': round(loss_dyn.item(), 4),
            })
    pbar.set_postfix(postfix)

    train_loss.append(loss.item())

    if (i + 1) % args.evaluation_steps == 0:
        print("\titers: {0} | loss: {1:.7f}".format(i + 1, sum(train_loss) / len(train_loss)))
        train_loss = []
        speed = (time.time() - time_now) / iter_count
        print('\tspeed: {:.4f}s/iter'.format(speed))
        iter_count = 0
        time_now = time.time()
        # save model and optimizer
        if (i + 1) < args.train_steps and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            save_path = os.path.join(args.checkpoints, args.model_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path,f'model_steps{i}.pth'))
            torch.save(model_optim.state_dict(), os.path.join(save_path, f'optim_steps{i}.pth'))
            if nu_tilde is not None:
                # nu_tilde 不在 model.state_dict() 里(是外部单独的Parameter)，方案文档
                # 第9节要求记录 nu，这里单独存一份，方便复现/续训时对齐 CVaR 的分位变量。
                torch.save({'nu_tilde': nu_tilde.detach().cpu()},
                           os.path.join(save_path, f'nu_tilde_steps{i}.pth'))

        # adjust learning rate
        scheduler.step()
        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

    loss.backward()
    clip_grad_norm_(params, args.grad_clip_value)
    model_optim.step()
save_path = os.path.join(args.checkpoints, f"{args.model_id}_final.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ IDF checkpoint saved to {save_path}")
if nu_tilde is not None:
    nu_save_path = os.path.join(args.checkpoints, f"{args.model_id}_nu_final.pth")
    torch.save({'nu_tilde': nu_tilde.detach().cpu(), 'nu': F.softplus(nu_tilde).detach().cpu()}, nu_save_path)
    print(f"✅ nu_tilde/nu saved to {nu_save_path}")
                
