#!/bin/bash

# FULLY AUTOMATED Script to run CRC benchmark validation across multiple confidence levels
# Automatically calibrates lambdas and runs benchmarks - NO MANUAL STEPS!

set -e  # Exit on error

# Default configuration
CONFIDENCE_LEVELS="0.5 0.6 0.7"
MAX_CALIBRATION_QUERIES=500
RAPS_LAMBDA=0.001
RAPS_KREG=1
AUTO_PLOT=false

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
        --raps-lambda)
            RAPS_LAMBDA="$2"
            shift 2
            ;;
        --raps-kreg)
            RAPS_KREG="$2"
            shift 2
            ;;
        --auto-plot)
            AUTO_PLOT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Fully automated CRC benchmark sweep. Calibrates lambdas and runs benchmarks."
            echo ""
            echo "Options:"
            echo "  --confidence-levels \"0.5 0.6 0.7\"    Confidence levels to test (default: 0.5 0.6 0.7)"
            echo "  --max-calibration-queries N          Number of calibration queries (default: 500)"
            echo "  --raps-lambda FLOAT                  RAPS lambda parameter (default: 0.001)"
            echo "  --raps-kreg INT                      RAPS k_reg parameter (default: 1)"
            echo "  --auto-plot                          Automatically generate plots after completion"
            echo "  --help                               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --confidence-levels \"0.5 0.6 0.7 0.8\" --max-calibration-queries 1000 --auto-plot"
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
echo "  2. Automatically extract and update HARDCODED_LAMBDAS"
echo "  3. Run benchmark queries for each confidence level"
echo "  4. Generate results CSV"
echo ""
echo "Configuration:"
echo "  Confidence levels: ${CONFIDENCE_LEVELS}"
echo "  Max calibration queries: ${MAX_CALIBRATION_QUERIES}"
echo "  RAPS parameters: lambda=${RAPS_LAMBDA}, k_reg=${RAPS_KREG}"
echo "  Auto-plot: ${AUTO_PLOT}"
echo "==================================================================="
echo ""

# Build command
CMD="python run_crc_benchmark_auto.py"
CMD="$CMD --confidence-levels ${CONFIDENCE_LEVELS}"
CMD="$CMD --max-calibration-queries ${MAX_CALIBRATION_QUERIES}"
CMD="$CMD --raps-lambda ${RAPS_LAMBDA}"
CMD="$CMD --raps-kreg ${RAPS_KREG}"

if [ "$AUTO_PLOT" = true ]; then
    CMD="$CMD --auto-plot"
fi

# Run the automated Python script
echo "Running: ${CMD}"
echo ""
$CMD

echo ""
echo "==================================================================="
echo "Done! Check the output directory for results."
echo "==================================================================="

