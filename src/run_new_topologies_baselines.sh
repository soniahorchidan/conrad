#!/bin/bash
#
# Run baseline benchmarks (symbolic Neo4j, neural ULTRA, hybrid) for the 5 new
# Query2Box topologies — 2p, 2i, 3i, pi, up. Mirrors run_new_topologies.sh,
# which only runs the ConRAD/CRC pipeline. Test queries must already exist under
# artifacts/queries/<dataset_normalized>/test/<topology>_pipeline.

set -uo pipefail  # NOTE: no -e — keep going if one topology / baseline fails.

# ─── Configuration ─────────────────────────────────────────────────────────────
DATASET="yago310"
INCOMPLETENESS=20
MAX_QUERIES=1000

# Topologies and the matching pipeline class names that
# run_baseline_benchmarks.py infers from query-dir suffix.
TOPOLOGIES=(
    "2p_pipeline:TwoHopPipeline"
    "2i_pipeline:TwoIntersectPipeline"
    "3i_pipeline:ThreeIntersectPipeline"
    "pi_pipeline:ProjectIntersectPipeline"
    "up_pipeline:UnionProjectPipeline"
)

NEURAL_THRESHOLDS=(0.7 0.8 0.9 0.99)
HYBRID_THRESHOLDS=(0.4 0.5 0.6 0.7)

# Which baseline types to run. Toggle any of these off if a sweep has already
# produced its results — each baseline appends to its own output dir.
RUN_SYMBOLIC=1
RUN_NEURAL=1
RUN_HYBRID=1

ULTRA_BATCH_SIZE=64

NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"

# ─── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline_benchmarks.py"

DATASET_NORMALIZED="$(echo "${DATASET}" | tr -d '-')"
TEST_BASE="${REPO_ROOT}/artifacts/queries/${DATASET_NORMALIZED}/test"

# ─── Logging ───────────────────────────────────────────────────────────────────
log() {
    local level="$1"; shift
    local yellow='\033[0;33m'; local reset='\033[0m'
    echo -e "${yellow}[$(date '+%Y-%m-%d %H:%M:%S')][${level}]: $*${reset}"
}

# ─── Pre-flight checks ─────────────────────────────────────────────────────────
for entry in "${TOPOLOGIES[@]}"; do
    topo="${entry%%:*}"
    qdir="${TEST_BASE}/${topo}"
    if [ ! -s "${qdir}/queries" ] || [ ! -s "${qdir}/gt" ]; then
        log ERROR "Missing test queries / gt at: ${qdir}"
        log ERROR "Run: bash scripts/generate_calibration.sh --dataset ${DATASET} --delete-edges-perc ${INCOMPLETENESS}"
        exit 1
    fi
done

cd "${SCRIPT_DIR}"

# ─── Main loop ─────────────────────────────────────────────────────────────────
log INFO "Starting baselines sweep for new topologies"
log INFO "Dataset:           ${DATASET}"
log INFO "Incompleteness:    ${INCOMPLETENESS}%"
log INFO "Topologies:        ${TOPOLOGIES[*]%%:*}"
log INFO "Neural thresholds: ${NEURAL_THRESHOLDS[*]}"
log INFO "Hybrid thresholds: ${HYBRID_THRESHOLDS[*]}"
log INFO "Symbolic | Neural | Hybrid = ${RUN_SYMBOLIC} | ${RUN_NEURAL} | ${RUN_HYBRID}"
log INFO "Max queries:       ${MAX_QUERIES}"

# Initialize with =() so that, under `set -u`, expanding an empty array
# (e.g. ${FAILED[*]} when nothing failed) does not trigger "unbound variable".
SUCCEEDED=()
FAILED=()

for entry in "${TOPOLOGIES[@]}"; do
    topo="${entry%%:*}"
    model="${entry##*:}"
    query_dir="${TEST_BASE}/${topo}"

    log INFO "─────────────────────────────────────────"
    log INFO "Topology: ${topo}  (pipeline: ${model})"
    log INFO "Query dir: ${query_dir}"
    log INFO "─────────────────────────────────────────"

    symbolic_output_dir="${REPO_ROOT}/artifacts/benchmark/symbolic_bench_${DATASET}_${model}_${INCOMPLETENESS}"
    neural_output_dir="${REPO_ROOT}/artifacts/benchmark/neural_bench_${DATASET}_${model}_${INCOMPLETENESS}"
    hybrid_output_dir="${REPO_ROOT}/artifacts/benchmark/hybrid_bench_${DATASET}_${model}_${INCOMPLETENESS}"

    # ── Symbolic ───────────────────────────────────────────────────────────
    if [ "${RUN_SYMBOLIC}" -eq 1 ]; then
        log INFO "[${topo}] Symbolic (Neo4j)..."
        if python -u "${BASELINE_SCRIPT}" \
            --query-dir "${query_dir}" \
            --dataset "${DATASET}" \
            --incompleteness "${INCOMPLETENESS}" \
            --baseline-type "symbolic" \
            --output-dir "${symbolic_output_dir}" \
            --max-queries "${MAX_QUERIES}" \
            --neo4j-host "${NEO4J_HOST}" \
            --neo4j-bolt-port "${NEO4J_BOLT_PORT}"; then
            SUCCEEDED+=("${topo}:symbolic")
        else
            log ERROR "[${topo}] Symbolic failed"
            FAILED+=("${topo}:symbolic")
        fi
    fi

    # ── Neural (ULTRA) ─────────────────────────────────────────────────────
    if [ "${RUN_NEURAL}" -eq 1 ]; then
        log INFO "[${topo}] Neural (ULTRA, thresholds=${NEURAL_THRESHOLDS[*]})..."
        # --use-ultraquery matches run_new_topologies.sh's checkpoint choice.
        # --no-score-cache: each (query, θ) runs its own ULTRA inference with
        # inter-hop pruning at θ, so per-threshold latency reflects the true
        # cost at that threshold (rather than being shared across thresholds).
        if python -u "${BASELINE_SCRIPT}" \
            --query-dir "${query_dir}" \
            --dataset "${DATASET}" \
            --incompleteness "${INCOMPLETENESS}" \
            --baseline-type "neural" \
            --output-dir "${neural_output_dir}" \
            --max-queries "${MAX_QUERIES}" \
            --ultra-batch-size "${ULTRA_BATCH_SIZE}" \
            --ultra-thresholds "${NEURAL_THRESHOLDS[@]}" \
            --no-score-cache \
            --use-ultraquery \
            --neo4j-host "${NEO4J_HOST}" \
            --neo4j-bolt-port "${NEO4J_BOLT_PORT}"; then
            SUCCEEDED+=("${topo}:neural")
        else
            log ERROR "[${topo}] Neural failed"
            FAILED+=("${topo}:neural")
        fi
    fi

    # ── Hybrid (one run per threshold; results appended to same CSV) ───────
    if [ "${RUN_HYBRID}" -eq 1 ]; then
        for thr in "${HYBRID_THRESHOLDS[@]}"; do
            log INFO "[${topo}] Hybrid (threshold=${thr})..."
            if python -u "${BASELINE_SCRIPT}" \
                --query-dir "${query_dir}" \
                --dataset "${DATASET}" \
                --incompleteness "${INCOMPLETENESS}" \
                --baseline-type "hybrid" \
                --output-dir "${hybrid_output_dir}" \
                --max-queries "${MAX_QUERIES}" \
                --ultra-batch-size "${ULTRA_BATCH_SIZE}" \
                --hybrid-thresholds "${thr}" "${thr}" "${thr}" \
                --use-ultraquery \
                --neo4j-host "${NEO4J_HOST}" \
                --neo4j-bolt-port "${NEO4J_BOLT_PORT}"; then
                SUCCEEDED+=("${topo}:hybrid@${thr}")
            else
                log ERROR "[${topo}] Hybrid @${thr} failed"
                FAILED+=("${topo}:hybrid@${thr}")
            fi
        done
    fi
done

# ─── Summary ───────────────────────────────────────────────────────────────────
log INFO "═════════════════════════════════════════"
log INFO "Sweep complete"
log INFO "  Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
log INFO "  Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
log INFO "Output dirs under: ${REPO_ROOT}/artifacts/benchmark/<baseline>_bench_${DATASET}_<model>_${INCOMPLETENESS}/"
log INFO "═════════════════════════════════════════"

[ ${#FAILED[@]} -eq 0 ]
