#!/bin/bash

set -euo pipefail

log() {
    local level="$1"
    shift
    local msg="$*"
    local yellow='\033[0;33m'
    local reset='\033[0m'
    echo -e "${yellow}[$(date '+%Y-%m-%d %H:%M:%S')][${level}]: ${msg}${reset}"
}

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_SCRIPT="${REPO_ROOT}/build.sh"

# Parse command line arguments
NEO4J_HOST="localhost"
NEO4J_BOLT_PORT="7687"
DATASET=""
DELETE_EDGES_PERC=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --neo4j-host)
            NEO4J_HOST="$2"
            shift 2
            ;;
        --neo4j-bolt-port)
            NEO4J_BOLT_PORT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --delete-edges-perc)
            DELETE_EDGES_PERC="$2"
            shift 2
            ;;
        *)
            log ERROR "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate mandatory arguments
if [ -z "${DELETE_EDGES_PERC}" ]; then
    log ERROR "Missing required argument: --delete-edges-perc"
    log ERROR "Usage: $0 --delete-edges-perc <percentage> --dataset <dataset>"
    log ERROR "Supported datasets: fb15k-237, nell-955, yago310"
    exit 1
fi

if [ -z "${DATASET}" ]; then
    log ERROR "Missing required argument: --dataset"
    log ERROR "Usage: $0 --delete-edges-perc <percentage> --dataset <dataset>"
    log ERROR "Supported datasets: fb15k-237, nell-955, yago310"
    exit 1
fi

# Validate dataset name
if [ "${DATASET}" != "fb15k-237" ] && [ "${DATASET}" != "nell-955" ] && [ "${DATASET}" != "yago310" ]; then
    log ERROR "Unsupported dataset: ${DATASET}"
    log ERROR "Supported datasets: fb15k-237, nell-955, yago310"
    exit 1
fi

log INFO "Using dataset: ${DATASET}"

# Check if dataset files exist, and prepare if needed
DATA_PATH="${REPO_ROOT}/artifacts/data"
DATASET_DATA_DIR="${DATA_PATH}/${DATASET}"
NODE_FILE="${DATASET_DATA_DIR}/neo4j_train_ind_ent.csv"
REL_FILE="${DATASET_DATA_DIR}/neo4j_train_ind_rels.csv"

if [ ! -f "${NODE_FILE}" ] || [ ! -f "${REL_FILE}" ]; then
    log INFO "Dataset files not found. Preparing dataset..."
    case "${DATASET}" in
        fb15k-237|nell-955|yago310)
            # Use process_dataset.py for all supported datasets
            python3 "${REPO_ROOT}/scripts/process_dataset.py" "${DATASET}"
            ;;
        *)
            log ERROR "Unknown dataset: ${DATASET}"
            log ERROR "Supported datasets: fb15k-237, nell-955, yago310"
            log ERROR "You can run: python3 scripts/process_dataset.py <dataset>"
            exit 1
            ;;
    esac
    
    # Verify files were created
    if [ ! -f "${NODE_FILE}" ] || [ ! -f "${REL_FILE}" ]; then
        log ERROR "Failed to prepare dataset files"
        exit 1
    fi
    log INFO "Dataset preparation complete"
fi

NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"
NEO4J_DB="${NEO4J_DB:-neo4j}"

CYPHER_SHELL_BASE=(cypher-shell -a neo4j://"${NEO4J_HOST}":"${NEO4J_BOLT_PORT}" -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" -d "${NEO4J_DB}")

# Normalize dataset name for folder names (remove hyphens)
# e.g., "fb15k-237" -> "fb15k237", "nell-955" -> "nell955"
DATASET_NORMALIZED=$(echo "${DATASET}" | tr -d '-')

# Set dataset-specific paths (use normalized format for folder names)
CRC_DATA_DIR="${REPO_ROOT}/artifacts/queries/${DATASET_NORMALIZED}"
CALIBRATION_BASE_DIR="${CRC_DATA_DIR}/calibration"
TEST_BASE_DIR="${CRC_DATA_DIR}/test"

# Ensure directories exist and are empty
log INFO "Preparing output directories"
# rm -rf "${CALIBRATION_BASE_DIR}" "${TEST_BASE_DIR}"

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
    if "${BUILD_SCRIPT}" neo4j start --without-docker; then
        wait_for_neo4j
    else
        log WARN "Neo4j start command returned non-zero status, checking if it's already running"
        if echo "RETURN 1;" | "${CYPHER_SHELL_BASE[@]}" >/dev/null 2>&1; then
            log INFO "Neo4j appears to be running; continuing"
        else
            log ERROR "Failed to start Neo4j"
            exit 1
        fi
    fi
}

stop_neo4j() {
    log INFO "Stopping Neo4j"
    if ! "${BUILD_SCRIPT}" neo4j stop --without-docker; then
        log WARN "Neo4j stop command returned non-zero status (possibly already stopped)"
    fi
}

get_num_relations() {
    log INFO "Querying Neo4j for number of relations" >&2
    # Query to get the maximum relation type, which gives us the number of relations
    # (assuming relation types are 0-indexed and continuous)
    local result
    # cypher-shell outputs: header line, separator line, then data
    # We need to extract the actual value from the output
    result=$(echo "MATCH ()-[r:Relation]->() RETURN MAX(r.type) AS max_type;" | "${CYPHER_SHELL_BASE[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}' | tr -d '[:space:]')
    
    if [ -z "${result}" ] || [ "${result}" = "null" ] || ! [[ "${result}" =~ ^[0-9]+$ ]]; then
        log ERROR "Failed to get number of relations from Neo4j. Result: '${result}'" >&2
        log ERROR "Trying alternative query method..." >&2
        # Alternative: count distinct relation types
        result=$(echo "MATCH ()-[r:Relation]->() RETURN COUNT(DISTINCT r.type) AS num_rels;" | "${CYPHER_SHELL_BASE[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}' | tr -d '[:space:]')
        if [ -z "${result}" ] || [ "${result}" = "null" ] || ! [[ "${result}" =~ ^[0-9]+$ ]]; then
            log ERROR "Alternative query also failed. Result: '${result}'" >&2
            exit 1
        fi
        log INFO "Using COUNT(DISTINCT) method, found ${result} relations" >&2
        echo "${result}"
        return
    fi
    
    # Add 1 because relation types are 0-indexed (if max is 236, we have 237 relations)
    local num_relations=$((result + 1))
    log INFO "Found ${num_relations} relations in dataset (max type: ${result})" >&2
    echo "${num_relations}"
}

start_neo4j

log INFO "Clearing existing graph data"
echo "MATCH (n) DETACH DELETE n;" | "${CYPHER_SHELL_BASE[@]}"

stop_neo4j

log INFO "Importing fresh dataset into Neo4j"
"${BUILD_SCRIPT}" neo4j import "${DATASET}" --without-docker

start_neo4j

log INFO "Adding inverse relations"
# NELL955 already has inverse relations, so skip adding them
if [ "${DATASET}" != "nell-955" ]; then
    NUM_RELATIONS=$(get_num_relations 2>&1 | grep -E '^[0-9]+$' | tail -n 1)
    log INFO "Using inverse relation offset: ${NUM_RELATIONS}"
    echo "MATCH (a:Entity)-[r:Relation]->(b:Entity) MERGE (b)-[:Relation {type: r.type + ${NUM_RELATIONS}}]->(a);" | "${CYPHER_SHELL_BASE[@]}"
else
    log INFO "Skipping inverse relation creation for ${DATASET} (already has inverse relations)"
fi

# log INFO "Skipping adding inverse relations"

# Check if queries already exist.
# Note: test queries are stored as FILES named `queries` and `gt` (not directories).
QUERIES_EXIST=true
if [ ! -s "${CALIBRATION_BASE_DIR}/3p_pipeline/queries.pkl" ] || \
   [ ! -s "${CALIBRATION_BASE_DIR}/3p_pipeline/answers.pkl" ] || \
   [ ! -s "${CALIBRATION_BASE_DIR}/2ip_pipeline/queries.pkl" ] || \
   [ ! -s "${CALIBRATION_BASE_DIR}/2ip_pipeline/answers.pkl" ] || \
   [ ! -s "${CALIBRATION_BASE_DIR}/2u_pipeline/queries.pkl" ] || \
   [ ! -s "${CALIBRATION_BASE_DIR}/2u_pipeline/answers.pkl" ] || \
   [ ! -s "${TEST_BASE_DIR}/3p_pipeline/queries" ] || \
   [ ! -s "${TEST_BASE_DIR}/3p_pipeline/gt" ] || \
   [ ! -s "${TEST_BASE_DIR}/2ip_pipeline/queries" ] || \
   [ ! -s "${TEST_BASE_DIR}/2ip_pipeline/gt" ] || \
   [ ! -s "${TEST_BASE_DIR}/2u_pipeline/queries" ] || \
   [ ! -s "${TEST_BASE_DIR}/2u_pipeline/gt" ]; then
    QUERIES_EXIST=false
fi

if [ "$QUERIES_EXIST" = true ]; then
    log INFO "Queries already exist in ${CRC_DATA_DIR}, skipping generation"
else
    log INFO "Generating and splitting calibration queries (3p, 2ip, and 2u)"
    mkdir -p "${CRC_DATA_DIR}"
    python3 "${REPO_ROOT}/src/sampler/calibration_sampler.py" \
        --neo4j-host "${NEO4J_HOST}" \
        --neo4j-bolt-port "${NEO4J_BOLT_PORT}" \
        --generate-3p \
        --generate-2ip \
        --generate-2u \
        --num-queries 5000 \
        --num-2ip-queries 5000 \
        --num-2u-queries 5000 \
        --size-ratio 1.0 \
        --max-hop-size 50 \
        --extract-intermediate \
        --calib-split 0.8 \
        --test-path "${TEST_BASE_DIR}" \
        --calib-path "${CALIBRATION_BASE_DIR}"
fi

log INFO "Deleting ${DELETE_EDGES_PERC}% of edges at random"
python3 "${REPO_ROOT}/scripts/delete_random_edges.py" --perc "${DELETE_EDGES_PERC}" --neo4j-host "${NEO4J_HOST}" --neo4j-bolt-port "${NEO4J_BOLT_PORT}"

log INFO "Calibration generation pipeline completed successfully"
log INFO "Generated calibration and test queries:"
log INFO "  - calibration: ${CALIBRATION_BASE_DIR}"
log INFO "  - test: ${TEST_BASE_DIR}"