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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="${REPO_ROOT}/build.sh"

# Parse command line arguments
CONFIG_FILE="${REPO_ROOT}/config/orb_config.json"
DATASET_OVERRIDE=""
DELETE_EDGES_PERC=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET_OVERRIDE="$2"
            shift 2
            ;;
        --delete-edges-perc)
            DELETE_EDGES_PERC="$2"
            shift 2
            ;;
        *)
            # Legacy support: first positional arg is config file
            if [ -z "${CONFIG_FILE_SET:-}" ]; then
                CONFIG_FILE="$1"
                CONFIG_FILE_SET=1
            fi
            shift
            ;;
    esac
done

# Validate mandatory arguments
if [ -z "${DELETE_EDGES_PERC}" ]; then
    log ERROR "Missing required argument: --delete-edges-perc"
    log ERROR "Usage: $0 --delete-edges-perc <percentage> [--dataset <dataset>] [--config <config_file>]"
    log ERROR "Supported datasets: fb15k-237, nell-955"
    exit 1
fi

NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"
NEO4J_DB="${NEO4J_DB:-neo4j}"

CYPHER_SHELL_BASE=(cypher-shell -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" -d "${NEO4J_DB}")

# Validate and set dataset name
if [ -n "${DATASET_OVERRIDE}" ]; then
    DATASET="${DATASET_OVERRIDE}"
    # Validate dataset name
    if [ "${DATASET}" != "fb15k-237" ] && [ "${DATASET}" != "nell-955" ]; then
        log ERROR "Unsupported dataset: ${DATASET}"
        log ERROR "Supported datasets: fb15k-237, nell-955"
        exit 1
    fi
    log INFO "Using dataset from command line: ${DATASET}"
    
    # Update config file with the dataset
    if [ -f "${CONFIG_FILE}" ]; then
        # Create a temporary config with updated dataset
        TMP_CONFIG=$(mktemp)
        jq ".core.dataset = \"${DATASET}\"" "${CONFIG_FILE}" > "${TMP_CONFIG}"
        # Also update kuzu database path if it exists
        if jq -e '.core.kuzu_database_path' "${CONFIG_FILE}" > /dev/null 2>&1; then
            KUZU_PATH="./artifacts/databases/${DATASET}-kuzu"
            jq ".core.kuzu_database_path = \"${KUZU_PATH}\"" "${TMP_CONFIG}" > "${TMP_CONFIG}.tmp" && mv "${TMP_CONFIG}.tmp" "${TMP_CONFIG}"
        fi
        CONFIG_FILE="${TMP_CONFIG}"
        log INFO "Updated config with dataset: ${DATASET}"
    fi
else
    # Read dataset name from config file
    if [ ! -f "${CONFIG_FILE}" ]; then
        log ERROR "Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    DATASET=$(jq -r '.core.dataset' "${CONFIG_FILE}")
    if [ -z "${DATASET}" ] || [ "${DATASET}" = "null" ]; then
        log ERROR "Dataset not found in config file: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Validate dataset name
    if [ "${DATASET}" != "fb15k-237" ] && [ "${DATASET}" != "nell-955" ]; then
        log ERROR "Unsupported dataset in config: ${DATASET}"
        log ERROR "Supported datasets: fb15k-237, nell-955"
        exit 1
    fi
    
    log INFO "Using dataset: ${DATASET} from config: ${CONFIG_FILE}"
fi

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
"${BUILD_SCRIPT}" neo4j import "${CONFIG_FILE}" --without-docker

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

log INFO "Generating and splitting calibration queries (3p, 2ip, and 2u)"
mkdir -p "${CRC_DATA_DIR}"
python3 "${REPO_ROOT}/src/sampler/calibration_sampler.py" \
    --generate-3p \
    --generate-2ip \
    --generate-2u \
    --num-queries 2000 \
    --num-2ip-queries 2000 \
    --num-2u-queries 2000 \
    --size-ratio 1.0 \
    --max-hop-size 50 \
    --extract-intermediate \
    --calib-split 0.5 \
    --test-path "${TEST_BASE_DIR}" \
    --calib-path "${CALIBRATION_BASE_DIR}"

log INFO "Deleting ${DELETE_EDGES_PERC}% of edges at random"
python3 "${REPO_ROOT}/benchmark/scripts/delete_random_edges.py" --perc "${DELETE_EDGES_PERC}"

log INFO "Calibration generation pipeline completed successfully"
log INFO "Generated calibration and test queries:"
log INFO "  - calibration: ${CALIBRATION_BASE_DIR}"
log INFO "  - test: ${TEST_BASE_DIR}"

# Clean up temporary config file if we created one
if [ -n "${TMP_CONFIG:-}" ] && [ -f "${TMP_CONFIG}" ]; then
    rm -f "${TMP_CONFIG}"
fi