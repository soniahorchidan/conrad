#!/bin/bash

# Quick test script for baseline benchmarks
# This runs a single baseline with a small number of queries for testing

set -euo pipefail

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline_benchmarks.py"

# Configuration - adjust these for your test
DATASET="fb15k-237"
MODEL="ThreeHopPipeline"
INCOMPLETENESS=5
MAX_QUERIES=10  # Small number for quick testing
QUERY_TYPE="3p_pipeline"

# Query directory
dataset_normalized=$(echo "${DATASET}" | tr -d '-')
query_dir="${REPO_ROOT}/artifacts/queries/${dataset_normalized}/test/${QUERY_TYPE}"

echo "=========================================="
echo "Testing Baseline Benchmarks"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Incompleteness: ${INCOMPLETENESS}%"
echo "Max queries: ${MAX_QUERIES}"
echo "Query directory: ${query_dir}"
echo "=========================================="
echo ""

# Check if query directory exists
if [ ! -d "${query_dir}" ]; then
    echo "ERROR: Query directory not found: ${query_dir}"
    exit 1
fi

cd "${SCRIPT_DIR}"

# Define explicit output directories for each baseline type (consistent with run.sh)
symbolic_output_dir="${REPO_ROOT}/artifacts/benchmark/symbolic_bench_${DATASET}_${MODEL}_${INCOMPLETENESS}"
hybrid_output_dir="${REPO_ROOT}/artifacts/benchmark/hybrid_bench_${DATASET}_${MODEL}_${INCOMPLETENESS}"

# Test 1: Symbolic baseline (fastest, no model needed)
echo "Testing symbolic baseline..."
echo "Output directory: ${symbolic_output_dir}"
python -u "${BASELINE_SCRIPT}" \
    --query-dir "${query_dir}" \
    --dataset "${DATASET}" \
    --incompleteness "${INCOMPLETENESS}" \
    --baseline-type "symbolic" \
    --output-dir "${symbolic_output_dir}" \
    --max-queries "${MAX_QUERIES}"

echo ""
echo "Symbolic baseline completed!"
echo ""

# Test 2: Hybrid baseline with multiple thresholds (tests append logic)
echo "Testing hybrid baseline with multiple thresholds..."
echo "Output directory: ${hybrid_output_dir}"

# Test with first threshold
echo "  Running hybrid with threshold 0.5..."
python -u "${BASELINE_SCRIPT}" \
    --query-dir "${query_dir}" \
    --dataset "${DATASET}" \
    --incompleteness "${INCOMPLETENESS}" \
    --baseline-type "hybrid" \
    --output-dir "${hybrid_output_dir}" \
    --max-queries "${MAX_QUERIES}" \
    --ultra-batch-size 64 \
    --hybrid-thresholds 0.5 0.5 0.5

# Test with second threshold (should append to same CSV)
echo "  Running hybrid with threshold 0.6..."
python -u "${BASELINE_SCRIPT}" \
    --query-dir "${query_dir}" \
    --dataset "${DATASET}" \
    --incompleteness "${INCOMPLETENESS}" \
    --baseline-type "hybrid" \
    --output-dir "${hybrid_output_dir}" \
    --max-queries "${MAX_QUERIES}" \
    --ultra-batch-size 64 \
    --hybrid-thresholds 0.6 0.6 0.6

echo ""
echo "Hybrid baseline completed!"
echo ""

# Test 3: Neural baseline (requires model, slower)
# Uncomment the following to test neural baseline:
# echo "Testing neural baseline with threshold 0.7..."
# python -u "${BASELINE_SCRIPT}" \
#     --query-dir "${query_dir}" \
#     --dataset "${DATASET}" \
#     --incompleteness "${INCOMPLETENESS}" \
#     --baseline-type "neural" \
#     --max-queries "${MAX_QUERIES}" \
#     --ultra-batch-size 64 \
#     --ultra-thresholds 0.7 \
#     --min-threshold 0.7

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
