# ROAR

ROAR is a retrieval-based correction head that sits on top of a frozen, retrieval-augmented time-series
forecaster (the *host*). It builds 2K candidate futures from the K retrieved neighbours (a level-aligned
and an anchor-aligned version of each), mixes them with a candidate softmax `p`, and applies the mix to the
host's forecast through an independent magnitude gate `u` (abstain when retrieval does not help).

Training has three stages:

| Stage | What is trained | Loss |
|---|---|---|
| 0 | Host fusion head (`idf_h_linear_head`) on a frozen backbone | 9-quantile pinball |
| 1 | ROAR heads `ψ_cand`, `ψ_base` (host frozen) | hard-sample-weighted pinball + `λ_gate`·BCE(u, 1[correction helps]) |
| 2 | `ψ_base` + host fusion head (`ψ_cand` frozen) | `MSE(ŷ, y) + κ·MSE(ŷ, sg(ŷ_ref))` |

Supported backbones: Chronos-Bolt, Moirai-2.0, TimesFM-2.5 (all frozen).

## Layout

```
roar/afocus_model.py                        base class (host wrapper, t_h calibration)
roar/afocus_model_roar.py                   Stage 1 model (ROAR)
roar/afocus_model_roar_gatecal*.py          Stage 2 models
roar/pretrain_A20.py                        Stage 1 / Stage 2 training
roar/zeroshot_A20.py                        zero-shot evaluation
roar/backbone_ablation_k10.sh               end-to-end pipeline (Stage 0 eval -> 1 -> 2 -> eval)
pretrain.py                                 Stage 0 host pretraining
models/                                     backbone + host fusion-head implementations
third_party/uni2ts                          minimal vendored Moirai-2 module (Apache-2.0)
```

## Setup

```bash
pip install torch transformers chronos-forecasting faiss-gpu pandas scikit-learn wandb huggingface_hub
# TimesFM-2.5 backbone only:
pip install -e "git+https://github.com/google-research/timesfm.git#egg=timesfm[torch]"
```

Data and weights are not included. Put them next to the repo (or set `DATA_ROOT`):

```
$DATA_ROOT/datasets/pretrain/pretrain_pairs_ctx512/        # TS-RAG pretraining pairs (precomputed neighbours)
$DATA_ROOT/retrieval_database/pretrain/retrieval_database_512.parquet
$DATA_ROOT/datasets/{ETT-small,weather,exchange_rate,electricity}/
checkpoints/base/                                           # Chronos-Bolt-base weights
```

The TS-RAG pretraining data / retrieval database are available at
https://huggingface.co/datasets/nkh/TS-RAG-Data.

## Best configuration (Chronos-Bolt)

`top_k = 10` everywhere (host training, Stage 1/2 training, evaluation).

| | Optimizer | LR | Weight decay | Batch | Steps | Dropout | Retrieved |
|---|---|---|---|---|---|---|---|
| Stage 0 (host) | AdamW | 3e-4 | 0.01 | 256 | 10,000 | 0.2 | 10 |
| Stage 1 | AdamW | 3e-4 (constant) | 0.01 | 256 | 2,500 | 0.2 | 10 |
| Stage 2 | AdamW | ψ_base 3e-4, host head 3e-5 (constant) | 0.01 | 256 | 2,000 | 0.2 | 10 |

ROAR-specific: β = 0.5, `t_h` = median host error over 50 calibration batches, `λ_gate` = 1, `κ` = 15,
gradient clip 1.0, context 512, horizon 64.

```bash
# Stage 0 host (writes checkpoints/<id>/<id>_final.pth)
python pretrain.py --model ChronosBoltRetrieve --augment_mode idf_h_linear_head --top_k 10 \
  --train_steps 10000 --learning_rate 3e-4 --weight_decay 0.01 --drop_prob 0.2 --batch_size 256 \
  --freeze_chronos_bolt --model_id <host_id> --checkpoints checkpoints/<host_id> ...

# Stage 1 -> Stage 2 -> 7-dataset evaluation of Stage 0/1/2
BACKBONE=chronos LG=1 KAPPA=15 S1_STEPS=2500 S2_STEPS=2000 NDS=7 EVAL_S1=1 \
  bash roar/backbone_ablation_k10.sh
```

For `BACKBONE=moirai2|timesfm25` the same script also runs Stage 0 host pretraining if the host checkpoint
is missing. `SMOKE=1` runs a few steps of every stage as a quick check.

Only the ROAR code paths are maintained here; other `augment_mode` branches in the training/evaluation
scripts are leftovers from earlier ablations and may reference modules that are not included.
