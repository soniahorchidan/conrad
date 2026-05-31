#!/bin/bash
#
# Run CRC benchmark across the 5 new Query2Box topologies on fb15k-237.
# Assumes calibration + test queries have already been generated via
# scripts/generate_calibration.sh for the chosen incompleteness level.

set -uo pipefail  # NOTE: no -e — we want one model's failure to not kill the rest.

# ─── Configuration ─────────────────────────────────────────────────────────────
DATASET="yago310"
INCOMPLETENESS=20
CONFIDENCE_LEVELS=(0.6 0.7 0.8 0.9)
MAX_EVAL_QUERIES=1000

MODELS=(
    "TwoHopPipeline"
    "TwoIntersectPipeline"
    "ThreeIntersectPipeline"
    "ProjectIntersectPipeline"
    "UnionProjectPipeline"
)

CALIB_BATCH_SIZE=64
INFERENCE_BATCH_SIZE=1

NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"

# ─── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRC_SCRIPT="${SCRIPT_DIR}/run_crc_benchmark_auto.py"
CACHE_DIR="${REPO_ROOT}/artifacts/snapshots/ultraquery/vector_crc_cache"

# ─── Logging ───────────────────────────────────────────────────────────────────
log() {
    local level="$1"; shift
    local yellow='\033[0;33m'; local reset='\033[0m'
    echo -e "${yellow}[$(date '+%Y-%m-%d %H:%M:%S')][${level}]: $*${reset}"
}

# ─── Pre-flight checks ─────────────────────────────────────────────────────────
DATASET_NORMALIZED="$(echo "${DATASET}" | tr -d '-')"
CALIB_BASE="${REPO_ROOT}/artifacts/queries/${DATASET_NORMALIZED}/calibration"

for model in "${MODELS[@]}"; do
    case "${model}" in
        TwoHopPipeline)            qt="2p_pipeline" ;;
        TwoIntersectPipeline)      qt="2i_pipeline" ;;
        ThreeIntersectPipeline)    qt="3i_pipeline" ;;
        ProjectIntersectPipeline)  qt="pi_pipeline" ;;
        UnionProjectPipeline)      qt="up_pipeline" ;;
    esac
    if [ ! -s "${CALIB_BASE}/${qt}/queries.pkl" ]; then
        log ERROR "Missing calibration data: ${CALIB_BASE}/${qt}/queries.pkl"
        log ERROR "Run: bash scripts/generate_calibration.sh --dataset ${DATASET} --delete-edges-perc ${INCOMPLETENESS}"
        exit 1
    fi
done

# ─── Main loop ─────────────────────────────────────────────────────────────────
log INFO "Starting new-topology benchmark sweep"
log INFO "Dataset:           ${DATASET}"
log INFO "Incompleteness:    ${INCOMPLETENESS}%"
log INFO "Confidence levels: ${CONFIDENCE_LEVELS[*]}"
log INFO "Models:            ${MODELS[*]}"
log INFO "Max eval queries:  ${MAX_EVAL_QUERIES}"

cd "${SCRIPT_DIR}"  # run_crc_benchmark_auto.py spawns `python validate_crc_composition.py` with a relative path

declare -a SUCCEEDED=() FAILED=()
for model in "${MODELS[@]}"; do
    log INFO "─────────────────────────────────────────"
    log INFO "Running benchmark for ${model}"
    log INFO "─────────────────────────────────────────"

    if python -u "${CRC_SCRIPT}" \
        --neo4j-host "${NEO4J_HOST}" \
        --neo4j-bolt-port "${NEO4J_BOLT_PORT}" \
        --confidence-levels "${CONFIDENCE_LEVELS[@]}" \
        --max-eval-queries "${MAX_EVAL_QUERIES}" \
        --model_to_infer "${model}" \
        --calib_batch_size "${CALIB_BATCH_SIZE}" \
        --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
        --dataset "${DATASET}" \
        --incompleteness "${INCOMPLETENESS}" \
        --use-ultraquery; then
        log INFO "${model} completed"
        SUCCEEDED+=("${model}")
    else
        log ERROR "${model} failed (continuing with remaining models)"
        FAILED+=("${model}")
    fi

    # Clear the per-model ULTRA score cache before the next topology
    if [ -d "${CACHE_DIR}" ]; then
        log INFO "Clearing cache: ${CACHE_DIR}"
        rm -rf "${CACHE_DIR}"/*
    fi
done

# ─── Summary ───────────────────────────────────────────────────────────────────
log INFO "═════════════════════════════════════════"
log INFO "Sweep complete"
log INFO "  Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
log INFO "  Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
log INFO "Results under: ${REPO_ROOT}/artifacts/benchmark/conrad_bench_${DATASET}_<model>_${INCOMPLETENESS}/"
log INFO "═════════════════════════════════════════"

# Non-zero exit if anything failed, so callers / CI can detect it.
[ ${#FAILED[@]} -eq 0 ]
