#!/bin/bash

# Re-run neural baseline with --no-score-cache for:
#   dataset: fb15k-237
#   model: ThreeHopPipeline
#   incompleteness: 5, 20, 40
#
# Produces threshold-specific latencies (one fresh inference run per θ),
# unlike the cached path which reuses a single execution_time across all θ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline_benchmarks.py"
GENERATE_CALIBRATION_SCRIPT="${REPO_ROOT}/scripts/generate_calibration.sh"

DATASET="fb15k-237"
MODEL="ThreeHopPipeline"
QUERY_TYPE="3p_pipeline"
INCOMPLETENESS_LEVELS=(5 20 40)
NEURAL_THRESHOLDS=(0.7 0.8 0.9 0.99)

NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
ULTRA_LOAD_PATH="${ULTRA_LOAD_PATH:-}"

dataset_normalized=$(echo "${DATASET}" | tr -d '-')
query_dir="${REPO_ROOT}/artifacts/queries/${dataset_normalized}/test/${QUERY_TYPE}"

if [ ! -d "${query_dir}" ]; then
    echo "ERROR: Query directory not found: ${query_dir}"
    exit 1
fi

log() {
    echo -e "\033[0;33m[$(date '+%Y-%m-%d %H:%M:%S')][$1]: ${*:2}\033[0m"
}

for inc in "${INCOMPLETENESS_LEVELS[@]}"; do
    if [ "${inc}" -eq 40 ]; then
        max_queries=500
    else
        max_queries=2000
    fi

    out_dir="${REPO_ROOT}/artifacts/benchmark/neural_bench_${DATASET}_${MODEL}_${inc}_nocache"

    log INFO "----------------------------------------"
    log INFO "Neural baseline (no-cache): ${DATASET} / ${MODEL} / inc=${inc}%"
    log INFO "Thresholds: ${NEURAL_THRESHOLDS[*]} | max_queries=${max_queries}"
    log INFO "Output: ${out_dir}"
    log INFO "----------------------------------------"

    # Restore Neo4j to the correct graph state for this incompleteness level.
    # generate_calibration.sh clears Neo4j, reimports the dataset, adds inverse
    # relations, and deletes inc% of edges (deterministic, seed=1). It skips
    # query regeneration when test/calibration files already exist.
    log INFO "Restoring Neo4j: ${DATASET} with ${inc}% edges deleted"
    cd "${REPO_ROOT}"
    if ! bash "${GENERATE_CALIBRATION_SCRIPT}" \
        --neo4j-host "${NEO4J_HOST}" \
        --neo4j-bolt-port "${NEO4J_BOLT_PORT}" \
        --dataset "${DATASET}" \
        --delete-edges-perc "${inc}"; then
        log ERROR "Failed to restore Neo4j state for inc=${inc}"
        exit 1
    fi
    cd "${SCRIPT_DIR}"

    cmd=(
        python -u "${BASELINE_SCRIPT}"
        --neo4j-host "${NEO4J_HOST}"
        --neo4j-bolt-port "${NEO4J_BOLT_PORT}"
        --query-dir "${query_dir}"
        --dataset "${DATASET}"
        --incompleteness "${inc}"
        --baseline-type "neural"
        --output-dir "${out_dir}"
        --max-queries "${max_queries}"
        --ultra-batch-size 64
        --ultra-thresholds "${NEURAL_THRESHOLDS[@]}"
        --no-score-cache
        --use-ultraquery
    )
    if [ -n "${ULTRA_LOAD_PATH}" ]; then
        cmd+=(--load-path "${ULTRA_LOAD_PATH}")
    fi

    if ! "${cmd[@]}"; then
        log ERROR "Neural baseline failed for inc=${inc}"
        exit 1
    fi

    log INFO "Done: inc=${inc}"
done

log INFO "All runs completed."
