#!/bin/bash

set -euo pipefail

# Configuration
INCOMPLETENESS_LEVELS=(20 5 40)
DATASETS=("fb15k-237" "nell-955" "yago310")
# Vector CRC
MODELS=("TwoUnionPipeline" "ThreeHopPipeline" "TwoIntersectProjectPipeline")
# Non Vector CRC
# MODELS=("NonVector3HopNeural")
# BASELINE_TYPES=("neural" "symbolic" "hybrid") # unused

# Threshold configurations
NEURAL_THRESHOLDS=(0.7 0.8 0.9 0.99)
HYBRID_THRESHOLDS=(0.45 0.5 0.6 0.7)

# Benchmark evaluation configuration
CONFIDENCE_LEVELS=(0.5 0.6 0.7 0.8 0.9)
MAX_EVAL_QUERIES=2000

# Neo4j Instance
# NEO4J_HOST=localhost
# NEO4J_BOLT_PORT=7688

# Model-specific calibration batch sizes (increased for better GPU utilization)
declare -A CALIB_BATCH_SIZES=(
    ["ThreeHopPipeline"]=64
    ["TwoUnionPipeline"]=64
    ["TwoIntersectProjectPipeline"]=64
    ["NonVector3HopNeural"]=64
)

# Model to query type mapping
declare -A MODEL_TO_QUERY_TYPE=(
    ["ThreeHopPipeline"]="3p_pipeline"
    ["TwoUnionPipeline"]="2u_pipeline"
    ["TwoIntersectProjectPipeline"]="2ip_pipeline"
    ["NonVector3HopNeural"]="3p_pipeline"
)

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATE_CALIBRATION_SCRIPT="${REPO_ROOT}/scripts/generate_calibration.sh"
CACHE_DIR="${REPO_ROOT}/artifacts/snapshots/ultraquery/vector_crc_cache"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline_benchmarks.py"

# Vector CRC
CRC_SCRIPT="${SCRIPT_DIR}/run_crc_benchmark_auto.py"
# Non Vector CRC
# CRC_SCRIPT="${SCRIPT_DIR}/run_crc_non_vector_auto.py"

# Optional: Set load path for neural/hybrid baselines (auto-detected if not set)
# You can set this to a specific model path, or leave empty for auto-detection
ULTRA_LOAD_PATH="${ULTRA_LOAD_PATH:-}"

log() {
    local level="$1"
    shift
    local msg="$*"
    local yellow='\033[0;33m'
    local reset='\033[0m'
    echo -e "${yellow}[$(date '+%Y-%m-%d %H:%M:%S')][${level}]: ${msg}${reset}"
}

# Main experiment loop
log INFO "Starting comprehensive benchmark sweep"
log INFO "Incompleteness levels: ${INCOMPLETENESS_LEVELS[*]}"
log INFO "Datasets: ${DATASETS[*]}"
log INFO "Models: ${MODELS[*]}"

for dataset in "${DATASETS[@]}"; do
    log INFO "=========================================="
    log INFO "Processing dataset: ${dataset}"
    log INFO "=========================================="

    for incompleteness in "${INCOMPLETENESS_LEVELS[@]}"; do
        log INFO "----------------------------------------"
        log INFO "Processing dataset: ${dataset} (incompleteness: ${incompleteness}%)"
        log INFO "----------------------------------------"
        
        # Set max eval queries (reduced for yago310 or incompleteness 40)
        if [ "${dataset}" = "yago310" ] || [ "${incompleteness}" -eq 40 ]; then
            max_eval_queries=500
        else
            max_eval_queries="${MAX_EVAL_QUERIES}"
        fi
        log INFO "Using max_eval_queries: ${max_eval_queries}"
        
        # Generate calibration data for this dataset and incompleteness level
        # Must run from REPO_ROOT so build.sh (invoked by generate_calibration.sh) finds ./artifacts/data/
        cd "${REPO_ROOT}"
        log INFO "Generating calibration data for ${dataset} with ${incompleteness}% incompleteness"
        if ! bash "${GENERATE_CALIBRATION_SCRIPT}" --neo4j-host "${NEO4J_HOST}" --neo4j-bolt-port "${NEO4J_BOLT_PORT}" --dataset "${dataset}" --delete-edges-perc "${incompleteness}"; then
            log ERROR "Failed to generate calibration data for ${dataset} with ${incompleteness}% incompleteness"
            exit 1
        fi
        log INFO "Calibration data generation completed"
        
        # Run benchmarks for each model
        for model in "${MODELS[@]}"; do
            calib_batch_size="${CALIB_BATCH_SIZES[$model]}"
            
            log INFO "Starting ${model} benchmark (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
            log INFO "Using calib_batch_size: ${calib_batch_size}"
            
            cd "${SCRIPT_DIR}"
            if ! python -u "${CRC_SCRIPT}" \
                --neo4j-host "${NEO4J_HOST}" \
                --neo4j-bolt-port "${NEO4J_BOLT_PORT}" \
                --confidence-levels "${CONFIDENCE_LEVELS[@]}" \
                --max-eval-queries "${max_eval_queries}" \
                --model_to_infer "${model}" \
                --calib_batch_size "${calib_batch_size}" \
                --inference-batch-size 1 \
                --dataset "${dataset}" \
                --incompleteness "${incompleteness}"  \
                --use-ultraquery; then
                log ERROR "Benchmark failed for ${model} (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
                exit 1
            fi
            
            log INFO "${model} benchmark completed (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
            
            # Delete cache after each run
            log INFO "Clearing cache: ${CACHE_DIR}"
            rm -rf "${CACHE_DIR}"/*
            log INFO "Cache cleared"
            
            # Run baseline benchmarks for this model
            query_type="${MODEL_TO_QUERY_TYPE[$model]}"
            dataset_normalized=$(echo "${dataset}" | tr -d '-')
            query_dir="${REPO_ROOT}/artifacts/queries/${dataset_normalized}/test/${query_type}"
            
            log INFO "Running baseline benchmarks for ${model}"
            log INFO "Query directory: ${query_dir}"
            
            # Check if query directory exists
            if [ ! -d "${query_dir}" ]; then
                log WARN "Query directory not found: ${query_dir}, skipping baselines"
            else
                # Define explicit output directories for each baseline type to ensure separation
                symbolic_output_dir="${REPO_ROOT}/artifacts/benchmark/symbolic_bench_${dataset}_${model}_${incompleteness}"
                neural_output_dir="${REPO_ROOT}/artifacts/benchmark/neural_bench_${dataset}_${model}_${incompleteness}"
                hybrid_output_dir="${REPO_ROOT}/artifacts/benchmark/hybrid_bench_${dataset}_${model}_${incompleteness}"
                
                # Run symbolic baseline (no thresholds needed)
                log INFO "Running symbolic baseline for ${model} (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
                log INFO "Output directory: ${symbolic_output_dir}"
                symbolic_cmd=(
                    python -u "${BASELINE_SCRIPT}"
                    --neo4j-host "${NEO4J_HOST}"
                    --neo4j-bolt-port "${NEO4J_BOLT_PORT}"
                    --query-dir "${query_dir}"
                    --dataset "${dataset}"
                    --incompleteness "${incompleteness}"
                    --baseline-type "symbolic"
                    --output-dir "${symbolic_output_dir}"
                    --max-queries "${max_eval_queries}"
                )
                if ! "${symbolic_cmd[@]}"; then
                    log ERROR "Symbolic baseline failed for ${model} (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
                else
                    log INFO "Symbolic baseline completed for ${model}"
                fi
                
                # Run neural baseline with multiple thresholds
                log INFO "Running neural baseline with thresholds: ${NEURAL_THRESHOLDS[*]}"
                log INFO "Output directory: ${neural_output_dir}"
                neural_cmd=(
                    python -u "${BASELINE_SCRIPT}"
                    --neo4j-host "${NEO4J_HOST}"
                    --neo4j-bolt-port "${NEO4J_BOLT_PORT}"
                    --query-dir "${query_dir}"
                    --dataset "${dataset}"
                    --incompleteness "${incompleteness}"
                    --baseline-type "neural"
                    --output-dir "${neural_output_dir}"
                    --max-queries "${max_eval_queries}"
                    --ultra-batch-size 64
                    --ultra-thresholds "${NEURAL_THRESHOLDS[@]}"
                    --min-threshold "${NEURAL_THRESHOLDS[0]}"
                    --use-ultraquery
                )
                if [ -n "${ULTRA_LOAD_PATH}" ]; then
                    neural_cmd+=(--load-path "${ULTRA_LOAD_PATH}")
                fi
                if ! "${neural_cmd[@]}"; then
                    log ERROR "Neural baseline failed for ${model} (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
                else
                    log INFO "Neural baseline completed for ${model}"
                fi
                
                # Run hybrid baseline with multiple thresholds (one run per threshold)
                log INFO "Output directory: ${hybrid_output_dir}"
                
                for hybrid_threshold in "${HYBRID_THRESHOLDS[@]}"; do
                    log INFO "Running hybrid baseline with threshold ${hybrid_threshold} for ${model} (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
                    
                    # Determine number of thresholds needed based on query type
                    if [ "${query_type}" = "2u_pipeline" ]; then
                        # 2u needs 2 thresholds (one per branch)
                        hybrid_thresholds_arg=("${hybrid_threshold}" "${hybrid_threshold}")
                    elif [ "${query_type}" = "2ip_pipeline" ]; then
                        # 2ip needs 3 thresholds (branch1, branch2, projection)
                        hybrid_thresholds_arg=("${hybrid_threshold}" "${hybrid_threshold}" "${hybrid_threshold}")
                    else
                        # 3p needs 3 thresholds (hop1, hop2, hop3)
                        hybrid_thresholds_arg=("${hybrid_threshold}" "${hybrid_threshold}" "${hybrid_threshold}")
                    fi
                    
                    hybrid_cmd=(
                        python -u "${BASELINE_SCRIPT}"
                        --neo4j-host "${NEO4J_HOST}"
                        --neo4j-bolt-port "${NEO4J_BOLT_PORT}"
                        --query-dir "${query_dir}"
                        --dataset "${dataset}"
                        --incompleteness "${incompleteness}"
                        --baseline-type "hybrid"
                        --output-dir "${hybrid_output_dir}"
                        --max-queries "${max_eval_queries}"
                        --ultra-batch-size 64
                        --hybrid-thresholds "${hybrid_thresholds_arg[@]}"
                        --use-ultraquery
                    )
                    if [ -n "${ULTRA_LOAD_PATH}" ]; then
                        hybrid_cmd+=(--load-path "${ULTRA_LOAD_PATH}")
                    fi
                    if ! "${hybrid_cmd[@]}"; then
                        log ERROR "Hybrid baseline (threshold=${hybrid_threshold}) failed for ${model} (dataset: ${dataset}, incompleteness: ${incompleteness}%)"
                        # Continue with other thresholds
                    else
                        log INFO "Hybrid baseline (threshold=${hybrid_threshold}) completed for ${model}"
                    fi
                done
                
                log INFO "All baseline benchmarks completed for ${model}"
            fi
        done
        
        log INFO "Completed all models for ${dataset} (incompleteness: ${incompleteness}%)"
    done
    
    log INFO "Completed all incompleteness levels for dataset: ${dataset}"
done

log INFO "=========================================="
log INFO "All benchmarks completed successfully at $(date)"
log INFO "=========================================="

