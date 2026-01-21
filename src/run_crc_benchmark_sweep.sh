#!/bin/bash

# FULLY AUTOMATED Script to run CRC benchmark validation across multiple confidence levels
# Automatically calibrates lambdas and runs benchmarks - NO MANUAL STEPS!

set -e  # Exit on error

# Default configuration
CONFIDENCE_LEVELS="0.5 0.6 0.7"
MAX_CALIBRATION_QUERIES=500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --confidence-levels)
            CONFIDENCE_LEVELS="$2"
            shift 2
            ;;
        --max-calibration-queries)
            MAX_CALIBRATION_QUERIES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Fully automated CRC benchmark sweep. Calibrates lambdas and runs benchmarks."
            echo ""
            echo "Options:"
            echo "  --confidence-levels \"0.5 0.6 0.7\"    Confidence levels to test (default: 0.5 0.6 0.7)"
            echo "  --max-calibration-queries N          Number of calibration queries (default: 500)"
            echo "  --help                               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --confidence-levels \"0.5 0.6 0.7 0.8\" --max-calibration-queries  1000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "==================================================================="
echo "FULLY AUTOMATED CRC BENCHMARK SWEEP"
echo "==================================================================="
echo "This script will:"
echo "  1. Run calibration to optimize lambdas"
echo "  2. Save calibrated lambdas to JSON file"
echo "  3. Run benchmark queries for each confidence level using the saved lambdas"
echo "  4. Generate results CSV"
echo ""
echo "Configuration:"
echo "  Confidence levels: ${CONFIDENCE_LEVELS}"
echo "  Max calibration queries: ${MAX_CALIBRATION_QUERIES}"
echo "==================================================================="
echo ""

# Build command
CMD="python run_crc_benchmark_auto.py"
CMD="$CMD --confidence-levels ${CONFIDENCE_LEVELS}"
CMD="$CMD --max-calibration-queries ${MAX_CALIBRATION_QUERIES}"

# Run the automated Python script
echo "Running: ${CMD}"
echo ""
$CMD

echo ""
echo "==================================================================="
echo "Done! Check the output directory for results."
echo "==================================================================="

