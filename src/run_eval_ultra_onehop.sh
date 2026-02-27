#!/bin/bash
#
# Self-contained 1-hop ULTRA vs UltraQuery evaluation: prepares data, starts Neo4j,
# imports the dataset, runs eval_ultra_onehop.py, optionally stops Neo4j.
#
# Neo4j holds a single database, so with multiple datasets we import and evaluate
# one dataset at a time.
#
# Usage:
#   From repo root:
#     bash src/run_eval_ultra_onehop.sh --dataset fb15k-237
#     bash src/run_eval_ultra_onehop.sh --dataset fb15k-237 nell-955 --output artifacts/eval_onehop_results.csv
#     bash src/run_eval_ultra_onehop.sh --dataset nell-955 --max-test 5000 --no-stop-neo4j
#
# Options:
#   --dataset DATASET [DATASET ...]   Datasets to evaluate (default: fb15k-237 nell-955)
#   --max-test N                       Max 1-hop test triples per dataset (default: 10000)
#   --output PATH                      Save results CSV to PATH (per-dataset files if multiple datasets)
#   --no-stop-neo4j                    Leave Neo4j running when done
#   --skip-neo4j-setup                 Do not start/import Neo4j; assume it is already running with the right data
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_SCRIPT="${REPO_ROOT}/build.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_ultra_onehop.py"
DATA_PATH="${REPO_ROOT}/artifacts/data"

NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"
NEO4J_DB="${NEO4J_DB:-neo4j}"
CYPHER_SHELL_BASE=(cypher-shell -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" -d "${NEO4J_DB}")

DATASETS=()
MAX_TEST=10000
OUTPUT_PATH=""
STOP_NEO4J=true
SKIP_NEO4J_SETUP=false

log() {
    local level="$1"
    shift
    local msg="$*"
    local yellow='\033[0;33m'
    local reset='\033[0m'
    echo -e "${yellow}[$(date '+%Y-%m-%d %H:%M:%S')][${level}]: ${msg}${reset}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --max-test)
            MAX_TEST="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --no-stop-neo4j)
            STOP_NEO4J=false
            shift
            ;;
        --skip-neo4j-setup)
            SKIP_NEO4J_SETUP=true
            shift
            ;;
        *)
            log ERROR "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(fb15k-237 nell-955)
fi

for d in "${DATASETS[@]}"; do
    if [ "$d" != "fb15k-237" ] && [ "$d" != "nell-955" ]; then
        log ERROR "Unsupported dataset: $d (supported: fb15k-237, nell-955)"
        exit 1
    fi
done

wait_for_neo4j() {
    local retries=30
    local delay=2
    for ((i = 1; i <= retries; i++)); do
        if echo "RETURN 1;" | "${CYPHER_SHELL_BASE[@]}" >/dev/null 2>&1; then
            return 0
        fi
        log INFO "Waiting for Neo4j to be ready (${i}/${retries})..."
        sleep "${delay}"
    done
    log ERROR "Neo4j did not become ready in time."
    exit 1
}

start_neo4j() {
    log INFO "Starting Neo4j"
    if "${BUILD_SCRIPT}" neo4j start --without-docker 2>/dev/null; then
        wait_for_neo4j
    else
        if echo "RETURN 1;" | "${CYPHER_SHELL_BASE[@]}" >/dev/null 2>&1; then
            log INFO "Neo4j already running; continuing"
        else
            log ERROR "Failed to start Neo4j"
            exit 1
        fi
    fi
}

stop_neo4j() {
    log INFO "Stopping Neo4j"
    if ! "${BUILD_SCRIPT}" neo4j stop --without-docker 2>/dev/null; then
        log WARN "Neo4j stop returned non-zero (may already be stopped)"
    fi
}

get_num_relations() {
    local result
    result=$(echo "MATCH ()-[r:Relation]->() RETURN MAX(r.type) AS max_type;" | "${CYPHER_SHELL_BASE[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}' | tr -d '[:space:]')
    if [ -z "${result}" ] || [ "${result}" = "null" ] || ! [[ "${result}" =~ ^[0-9]+$ ]]; then
        result=$(echo "MATCH ()-[r:Relation]->() RETURN COUNT(DISTINCT r.type) AS num_rels;" | "${CYPHER_SHELL_BASE[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}' | tr -d '[:space:]')
    fi
    if [ -z "${result}" ] || [ "${result}" = "null" ] || ! [[ "${result}" =~ ^[0-9]+$ ]]; then
        echo "0"
        return
    fi
    local num_relations=$((result + 1))
    echo "${num_relations}"
}

prepare_dataset() {
    local dataset="$1"
    local node_file="${DATA_PATH}/${dataset}/neo4j_train_ind_ent.csv"
    local rel_file="${DATA_PATH}/${dataset}/neo4j_train_ind_rels.csv"
    if [ ! -f "${node_file}" ] || [ ! -f "${rel_file}" ]; then
        log INFO "Preparing dataset ${dataset} (running process_dataset.py)"
        (cd "${REPO_ROOT}" && python3 scripts/process_dataset.py "${dataset}")
        if [ ! -f "${node_file}" ] || [ ! -f "${rel_file}" ]; then
            log ERROR "Dataset preparation failed for ${dataset}"
            exit 1
        fi
        log INFO "Dataset ${dataset} prepared"
    else
        log INFO "Dataset files found for ${dataset}"
    fi
}

import_dataset_into_neo4j() {
    local dataset="$1"
    log INFO "Importing dataset ${dataset} into Neo4j"
    stop_neo4j
    "${BUILD_SCRIPT}" neo4j import "${dataset}" --without-docker
    start_neo4j
    if [ "${dataset}" != "nell-955" ]; then
        log INFO "Adding inverse relations for ${dataset}"
        NUM_RELATIONS=$(get_num_relations)
        if [ -n "${NUM_RELATIONS}" ] && [ "${NUM_RELATIONS}" -gt 0 ]; then
            echo "MATCH (a:Entity)-[r:Relation]->(b:Entity) MERGE (b)-[:Relation {type: r.type + ${NUM_RELATIONS}}]->(a);" | "${CYPHER_SHELL_BASE[@]}"
        fi
    else
        log INFO "Skipping inverse relations for nell-955 (already included)"
    fi
}

log INFO "1-hop ULTRA vs UltraQuery evaluation (self-contained)"
log INFO "Datasets: ${DATASETS[*]}"
log INFO "Max test triples per dataset: ${MAX_TEST}"

for dataset in "${DATASETS[@]}"; do
    log INFO "=========================================="
    log INFO "Dataset: ${dataset}"
    log INFO "=========================================="

    prepare_dataset "${dataset}"

    if [ "${SKIP_NEO4J_SETUP}" = false ]; then
        import_dataset_into_neo4j "${dataset}"
    else
        log INFO "Skipping Neo4j setup (--skip-neo4j-setup); assuming Neo4j is running with correct data"
        start_neo4j 2>/dev/null || true
    fi

    out_arg=()
    if [ -n "${OUTPUT_PATH}" ]; then
        if [ ${#DATASETS[@]} -eq 1 ]; then
            if [[ "${OUTPUT_PATH}" = /* ]]; then
                out_arg=(--output "${OUTPUT_PATH}")
            else
                out_arg=(--output "${REPO_ROOT}/${OUTPUT_PATH}")
            fi
        else
            out_arg=(--output "${REPO_ROOT}/artifacts/eval_onehop_${dataset}.csv")
        fi
    fi

    log INFO "Running 1-hop evaluation for ${dataset}"
    cd "${SCRIPT_DIR}"
    python3 -u "${EVAL_SCRIPT}" \
        --dataset "${dataset}" \
        --max-test "${MAX_TEST}" \
        "${out_arg[@]}"
    log INFO "Evaluation completed for ${dataset}"
done

if [ "${STOP_NEO4J}" = true ]; then
    stop_neo4j
    log INFO "Neo4j stopped"
else
    log INFO "Neo4j left running (--no-stop-neo4j)"
fi

log INFO "All 1-hop evaluations completed successfully"
