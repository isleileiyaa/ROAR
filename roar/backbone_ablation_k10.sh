#!/usr/bin/env bash
# 换backbone实验(top_k全程一致版)：BACKBONE=moirai2|timesfm25
#   Host预训练(pretrain.py, idf_h_linear_head, 10000步, lr=5e-6) -> Stage0评测(host自己)
#   -> Stage1(afocus_roar, λ_gate=1, 1000步，不单独评测)
#   -> Stage2(afocus_roar_gatecal_hosthead, κ=5, 2000步) -> Stage2评测
# Host训练、Stage1/2训练、全部评测的top_k都是10(训练集每条样本预存20个邻居，10<=20不会被截断)。
# 7数据集评测，保留全部checkpoint。已有的host(top_k=10, 10000步)直接复用。
# 资源礼让：只用完全空闲(无任何计算进程)且显存充足的GPU；每一步开始前重新检查，
# 发现别的进程占用了这块卡/显存或磁盘不足就主动退出(已完成的步骤会被下次重跑跳过)。
# SMOKE=1: 冒烟测试，每步只训几十步、只评ETTh1，tag加_smoke前缀，不影响正式结果。
set -uo pipefail
TS="${TS:-$(cd "$(dirname "$0")/.." && pwd)}"   # 仓库根目录
FEN="${DATA_ROOT:-$TS/..}"   # 放 datasets/ 和 retrieval_database/ 的目录
cd "$TS"
PY="${PY:-python}"
TOPK=10
HOST_TOPK=10
# 可用环境变量覆盖(默认值=之前换backbone实验的设定)：LG KAPPA S1_STEPS S2_STEPS NDS(6|7) EVAL_S1(0|1) RUN_TAG
LG="${LG:-1}"
KAPPA="${KAPPA:-5}"
EVAL_S1="${EVAL_S1:-0}"
NDS="${NDS:-7}"
RUN_TAG="${RUN_TAG:-}"
STAGE2_SUFFIX="${STAGE2_SUFFIX:-}"  # 同一个Stage1上试不同Stage2设定时区分tag，如 _s1500
BACKBONE="${BACKBONE:?BACKBONE=moirai2|timesfm25|chronos}"
case $BACKBONE in
  chronos) MODEL=ChronosBoltRetrieve;;
  moirai2) MODEL=Moirai2Retrieve;;
  timesfm25) MODEL=TimesFM25Retrieve;;
  *) echo "unknown BACKBONE=$BACKBONE"; exit 1;;
esac
MIN_FREE_MIB=20000
MIN_FREE_DISK_GB=15
SMOKE="${SMOKE:-0}"

if [ "$SMOKE" = "1" ]; then
  P=smoke_${BACKBONE}_k10
  HOST_STEPS=30; S1_STEPS=20; S2_STEPS=20; CALIB=2
  DATASETS=(ETTh1)
else
  P=${BACKBONE}_k10${RUN_TAG}
  HOST_STEPS=10000; S1_STEPS="${S1_STEPS:-1000}"; S2_STEPS="${S2_STEPS:-2000}"; CALIB=50
  DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity)
  [ "$NDS" = "6" ] && DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate)
fi
OUT_DIR="$TS/sidecar/kappa_lambda_sweep/${P}_backbone_ablation"
mkdir -p "$OUT_DIR"

HOST_TAG="${BACKBONE}_h_linear_head_full${HOST_STEPS}"  # host本来就是top_k=10训的，不带_k10后缀，直接复用
[ "$BACKBONE" = "chronos" ] && [ "$SMOKE" != "1" ] && HOST_TAG="idf_h_linear_head_paperhp_full10000"  # Chronos主模型复用paperhp host
STAGE1_TAG="${P}_stage1_lg${LG}_seed2021"
STAGE2_TAG="${P}_stage2${STAGE2_SUFFIX}_lg${LG}_kappa${KAPPA}_seed2021"
HOST_CKPT="checkpoints/${HOST_TAG}/${HOST_TAG}_final.pth"
S1_CKPT="checkpoints/${STAGE1_TAG}/${STAGE1_TAG}_final.pth"
S2_CKPT="checkpoints/${STAGE2_TAG}/${STAGE2_TAG}_final.pth"

bail() { echo "[$P] 主动退出: $*" | tee -a "$OUT_DIR/ABORTED"; exit 2; }
gpu_busy() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$uuid"
}
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
check_resources() {
  gpu_busy "$GPU" && bail "GPU$GPU 上出现了别的进程($1之前)"
  [ "$(gpu_free_mib "$GPU")" -lt "$MIN_FREE_MIB" ] && bail "GPU$GPU 空闲显存不足 ${MIN_FREE_MIB}MiB($1之前)"
  local free_gb
  free_gb=$(df --output=avail -BG "$TS" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -lt "$MIN_FREE_DISK_GB" ] && bail "磁盘只剩${free_gb}G($1之前)"
  return 0
}
check_log() {  # 训练日志里有Traceback/NaN就停
  if grep -qiE "Traceback|[^a-z]nan[^a-z]" "$1"; then echo "[$P] $1 里发现Traceback/NaN"; exit 1; fi
}

GPU=""
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  if ! gpu_busy "$i" && [ "$(gpu_free_mib "$i")" -ge "$MIN_FREE_MIB" ]; then GPU=$i; break; fi
done
[ -z "$GPU" ] && bail "没有完全空闲的GPU"
echo "[$P] 使用 GPU$GPU (SMOKE=$SMOKE)"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE=offline
export PYTHONPATH="$TS"
export PRETRAIN_SEED=2021
export HF_HUB_OFFLINE=1  # backbone权重只从本地HF缓存加载，不依赖代理/网络(01:34那次就是代理断了导致TimesFM加载失败)

declare -A DS_DATA=( [ETTh1]=ett_h_retrieve [ETTh2]=ett_h_retrieve [ETTm1]=ett_m_retrieve [ETTm2]=ett_m_retrieve [weather]=custom_retrieve [exchange_rate]=custom_retrieve [electricity]=custom_retrieve )
declare -A DS_FREQ=( [ETTh1]=hour [ETTh2]=hour [ETTm1]=minute [ETTm2]=minute [weather]=10minutes [exchange_rate]=hour [electricity]=hour )
declare -A DS_ROOT=( [ETTh1]="$FEN/datasets/ETT-small/" [ETTh2]="$FEN/datasets/ETT-small/" [ETTm1]="$FEN/datasets/ETT-small/" [ETTm2]="$FEN/datasets/ETT-small/" [weather]="$FEN/datasets/weather/" [exchange_rate]="$FEN/datasets/exchange_rate/" [electricity]="$FEN/datasets/electricity/" )

eval_one() {
  local ckpt="$1" tag="$2" augment_mode="$3"
  local summary="$OUT_DIR/${tag}_summary.txt"
  if [ -f "$OUT_DIR/${tag}_EVAL_DONE" ]; then echo "[$tag] 评测已完成，跳过"; return 0; fi
  : > "$summary"
  for ds in "${DATASETS[@]}"; do
    check_resources "评测${tag}/${ds}"
    local log="$OUT_DIR/${tag}_eval_${ds}.log"
    "$PY" roar/zeroshot_A20.py \
      --root_path "${DS_ROOT[$ds]}" --data_path "${ds}.csv" \
      --model_id "${ds}_zeroshot_512_pred_64_512_retrieve_64_${tag}" \
      --data "${DS_DATA[$ds]}" --top_k "$TOPK" \
      --checkpoint_model_path "$ckpt" --pretrained_model_path "$TS/checkpoints/base/" \
      --seq_len 512 --label_len 0 --pred_len 64 --lookback_length 512 \
      --batch_size 256 --num_workers 0 --freq 0 --percent 100 \
      --model "$MODEL" --gpu_loc 0 \
      --save_file_name "eval_${tag}_${ds}.txt" \
      --retrieval_database_dir "$FEN/retrieval_database/" --dimension 768 \
      --embedding_model_type chronos --metadata_frequency "${DS_FREQ[$ds]}" \
      --metadata_database_name "$ds" --augment_mode "$augment_mode" --afocus_host_aug idf_h_linear_head \
      > "$log" 2>&1
    local mse mae
    mse=$(grep "^MSE:" "$log" | awk '{print $2}')
    mae=$(grep "^MAE:" "$log" | awk '{print $2}')
    if [ -z "$mse" ]; then echo "[$tag] $ds 评测失败，查看 $log"; exit 1; fi
    echo "$ds  MSE=$mse  MAE=$mae" | tee -a "$summary"
  done
  python3 -c "
import re
mses,maes=[],[]
for line in open('$summary'):
    m=re.search(r'MSE=([\d.]+)\s+MAE=([\d.]+)', line)
    if m:
        mses.append(float(m.group(1))); maes.append(float(m.group(2)))
if mses:
    print(f'AVG({len(mses)}ds)  MSE={sum(mses)/len(mses):.4f}  MAE={sum(maes)/len(maes):.4f}')
" | tee -a "$summary"
  touch "$OUT_DIR/${tag}_EVAL_DONE"
}

ROAR_COMMON=(--top_k "$TOPK" --retrieve_lookback_length 512
  --retrieval_database_path "$FEN/retrieval_database/pretrain/retrieval_database_512.parquet"
  --afocus_host_aug idf_h_linear_head --afocus_beta 0.5 --th_calib_batches "$CALIB"
  --pretrained_model_path checkpoints/base/ --context_length 512 --prediction_length 64
  --data_path "$FEN/datasets/pretrain/pretrain_pairs_ctx512"
  --optimizer adamw --learning_rate 0.0003 --weight_decay 0.01
  --tmax 20 --drop_prob 0.2 --batch_size 256 --grad_clip_value 1.0
  --shuffle_buffer_length 10000 --gpu_loc 0 --afocus_host_ckpt "$HOST_CKPT")

# --- Host预训练 ---
if [ "$BACKBONE" = "chronos" ] && ! [ -f "$HOST_CKPT" ]; then echo "Chronos host $HOST_CKPT 不存在(本脚本不负责重训Chronos host)"; exit 1; fi
if [ -f "$HOST_CKPT" ]; then
  echo "[$HOST_TAG] host checkpoint已存在，跳过"
else
  check_resources "Host预训练"
  echo "=== Host预训练($MODEL, idf_h_linear_head, ${HOST_STEPS}步, lr=5e-6, top_k=$HOST_TOPK) ==="
  mkdir -p "checkpoints/${HOST_TAG}"
  "$PY" pretrain.py \
    --model_id "$HOST_TAG" --model "$MODEL" \
    --top_k "$HOST_TOPK" --retrieve_lookback_length 512 \
    --retrieval_database_path "$FEN/retrieval_database/pretrain/retrieval_database_512.parquet" \
    --augment_mode idf_h_linear_head --context_length 512 --prediction_length 64 \
    --data_path "$FEN/datasets/pretrain/pretrain_pairs_ctx512" \
    --train_steps "$HOST_STEPS" --evaluation_steps 1000 \
    --optimizer adamw --learning_rate 0.000005 --weight_decay 0.01 \
    --tmax 20 --drop_prob 0.2 --batch_size 256 --grad_clip_value 1.0 \
    --shuffle_buffer_length 10000 --freeze_chronos_bolt \
    --checkpoints "checkpoints/${HOST_TAG}" \
    > "$OUT_DIR/${HOST_TAG}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] || ! [ -f "$HOST_CKPT" ] && { echo "Host预训练失败(rc=$rc)，查看 $OUT_DIR/${HOST_TAG}_train.log"; exit 1; }
  check_log "$OUT_DIR/${HOST_TAG}_train.log"
  echo "Host预训练完成"
fi

echo "=== Stage0评测($BACKBONE host自己, idf_h_linear_head, top_k=$TOPK) ==="
eval_one "$HOST_CKPT" "${P}_stage0_host" "idf_h_linear_head"

# --- Stage1 ---
if [ -f "$S1_CKPT" ]; then
  echo "[$STAGE1_TAG] checkpoint已存在，跳过"
else
  check_resources "Stage1训练"
  echo "=== Stage1训练(afocus_roar, lambda_gate=$LG, ${S1_STEPS}步, top_k=$TOPK) ==="
  mkdir -p "checkpoints/${STAGE1_TAG}"
  "$PY" roar/pretrain_A20.py \
    --model_id "$STAGE1_TAG" --model "$MODEL" "${ROAR_COMMON[@]}" \
    --train_steps "$S1_STEPS" --checkpoints "checkpoints/${STAGE1_TAG}" \
    --augment_mode afocus_roar --lambda_gate "$LG" \
    > "$OUT_DIR/${STAGE1_TAG}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] || ! [ -f "$S1_CKPT" ] && { echo "Stage1训练失败(rc=$rc)，查看 $OUT_DIR/${STAGE1_TAG}_train.log"; exit 1; }
  check_log "$OUT_DIR/${STAGE1_TAG}_train.log"
  echo "Stage1训练完成"
fi

if [ "$EVAL_S1" = "1" ]; then
  echo "=== Stage1评测 ==="
  eval_one "$S1_CKPT" "${P}_stage1" "afocus_roar"
fi

# --- Stage2 ---
if [ -f "$S2_CKPT" ]; then
  echo "[$STAGE2_TAG] checkpoint已存在，跳过"
else
  check_resources "Stage2训练"
  echo "=== Stage2训练(afocus_roar_gatecal_hosthead, kappa=$KAPPA, ${S2_STEPS}步, top_k=$TOPK) ==="
  mkdir -p "checkpoints/${STAGE2_TAG}"
  "$PY" roar/pretrain_A20.py \
    --model_id "$STAGE2_TAG" --model "$MODEL" "${ROAR_COMMON[@]}" \
    --train_steps "$S2_STEPS" --checkpoints "checkpoints/${STAGE2_TAG}" \
    --augment_mode afocus_roar_gatecal_hosthead --host_head_lr 3e-5 --gatecal_kappa "$KAPPA" \
    --init_from_checkpoint "$S1_CKPT" \
    > "$OUT_DIR/${STAGE2_TAG}_train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] || ! [ -f "$S2_CKPT" ] && { echo "Stage2训练失败(rc=$rc)，查看 $OUT_DIR/${STAGE2_TAG}_train.log"; exit 1; }
  check_log "$OUT_DIR/${STAGE2_TAG}_train.log"
  echo "Stage2训练完成"
fi

echo "=== Stage2评测 ==="
eval_one "$S2_CKPT" "${P}_stage2${STAGE2_SUFFIX}" "afocus_roar_gatecal_hosthead"

echo "=== 全部完成 ==="
STAGES=(stage0_host stage2${STAGE2_SUFFIX}); [ "$EVAL_S1" = "1" ] && STAGES=(stage0_host stage1 stage2${STAGE2_SUFFIX})
for s in "${STAGES[@]}"; do echo "--- $s ---"; tail -1 "$OUT_DIR/${P}_${s}_summary.txt"; done
touch "$OUT_DIR/ABLATION_DONE"
