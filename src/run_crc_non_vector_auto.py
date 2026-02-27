#!/usr/bin/env python3
"""
Fully automated CRC benchmark sweep across multiple confidence levels.
Automatically calibrates lambdas and runs benchmark queries.
"""

import os
import subprocess
import re
from typing import List, Tuple


def run_command(cmd: List[str], log_path: str) -> Tuple[int, str]:
    """Run a command and capture output to both console and log file."""
    print(f"Running: {' '.join(cmd)}")
    
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print to console
            log_file.write(line)  # Write to log
            output_lines.append(line)
        
        process.wait()
        return process.returncode, ''.join(output_lines)


def run_benchmark_sweep(confidence_levels: List[float], output_dir: str, 
                       max_calibration_queries: int = 1600,
                       max_eval_queries: int = 1000,
                       extra_args: List[str] = None,
                       dataset: str = None) -> str:
    """
    Run full benchmark sweep with automatic lambda calibration.
    
    Args:
        extra_args: Additional arguments to pass through to validate_crc_composition.py
    
    Returns:
        Path to results CSV file
    """
    if extra_args is None:
        extra_args = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 0: CLEAN CACHE DIR")
    print("="*80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    cache_dir = os.path.join(repo_root, "artifacts", "non_vector_crc_cache")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Cleaned up cache directory: {cache_dir}")

    print("\n" + "="*80)
    print("STEP 1: RUNNING BENCHMARK QUERIES")
    print("="*80)
    
    results_csv = os.path.join(output_dir, "results_summary.csv")
    
    # Create CSV header with both all-queries and non-abstained metrics
    with open(results_csv, 'w') as f:
        f.write("confidence,precision,recall,f1,abstention_rate,non_abs_precision,non_abs_recall,non_abs_f1,num_queries,num_non_abstained\n")
    
    for conf in confidence_levels:
        print(f"\n>>> Running benchmark for confidence = {conf}")
        print("-"*80)
        
        benchmark_log = os.path.join(output_dir, f"benchmark_conf_{conf}.log")
        benchmark_cmd = [
            "python", "validate_crc_composition.py",
            "--mode", "both",
            "--confidence", str(conf),
            "--max-eval-queries", str(max_eval_queries),
        ] + extra_args
        
        returncode, output = run_command(benchmark_cmd, benchmark_log)
        
        if returncode != 0:
            print(f"WARNING: Benchmark failed for confidence {conf}")
            continue
        
        # Extract results from log
        with open(benchmark_log, 'r') as f:
            log_content = f.read()
        
        # Look for "OVERALL AVERAGE (All queries): Precision=X, Recall=Y, F1=Z"
        overall_all_match = re.search(r'OVERALL AVERAGE \(All queries\):\s*Precision=([\d.]+),\s*Recall=([\d.]+),\s*F1=([\d.]+)', log_content)
        
        # Look for "Abstention rate: X (Y/Z)"
        abstention_match = re.search(r'Abstention rate:\s*([\d.]+)\s*\((\d+)/(\d+)\)', log_content)
        
        # Look for "OVERALL AVERAGE (Non-abstained only): Precision=X, Recall=Y, F1=Z"
        overall_non_abs_match = re.search(r'OVERALL AVERAGE \(Non-abstained only\):\s*Precision=([\d.]+),\s*Recall=([\d.]+),\s*F1=([\d.]+)', log_content)
        
        # Look for "Non-abstained queries: X/Y"
        non_abs_count_match = re.search(r'Non-abstained queries:\s*(\d+)/(\d+)', log_content)
        
        # Look for "Total queries processed: X"
        queries_match = re.search(r'Total queries processed:\s*(\d+)', log_content)
        
        if overall_all_match and abstention_match and queries_match:
            # All queries metrics
            precision = float(overall_all_match.group(1))
            recall = float(overall_all_match.group(2))
            f1 = float(overall_all_match.group(3))
            
            # Abstention rate
            abstention_rate = float(abstention_match.group(1))
            abstained_count = int(abstention_match.group(2))
            
            # Total queries
            num_queries = int(queries_match.group(1))
            
            # Non-abstained metrics (if available)
            if overall_non_abs_match and non_abs_count_match:
                non_abs_precision = float(overall_non_abs_match.group(1))
                non_abs_recall = float(overall_non_abs_match.group(2))
                non_abs_f1 = float(overall_non_abs_match.group(3))
                num_non_abstained = int(non_abs_count_match.group(1))
            else:
                # All queries abstained
                non_abs_precision = 0.0
                non_abs_recall = 0.0
                non_abs_f1 = 0.0
                num_non_abstained = 0
            
            # Append to CSV
            with open(results_csv, 'a') as f:
                f.write(f"{conf},{precision},{recall},{f1},{abstention_rate},{non_abs_precision},{non_abs_recall},{non_abs_f1},{num_queries},{num_non_abstained}\n")
            
            print(f"Results (All): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            print(f"  Abstention: {abstention_rate:.4f} ({abstained_count}/{num_queries})")
            if num_non_abstained > 0:
                print(f"  Non-abstained: Precision={non_abs_precision:.4f}, Recall={non_abs_recall:.4f}, F1={non_abs_f1:.4f} ({num_non_abstained} queries)")
        else:
            print(f"WARNING: Could not extract results from benchmark log")

    # Step 4: Display summary
    print("\n" + "="*80)
    print("BENCHMARK SWEEP COMPLETED")
    print("="*80)
    
    print(f"\nResults saved to: {results_csv}")
    
    return results_csv


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated CRC benchmark sweep",
        epilog="Additional arguments after '--' will be passed to validate_crc_composition.py"
    )
    parser.add_argument("--confidence-levels", nargs='+', type=float, 
                       default=[0.5, 0.6, 0.7],
                       help="Confidence levels to test (default: 0.5 0.6 0.7)")
    parser.add_argument("--max-calibration-queries", type=int, default=100,
                       help="Number of queries for calibration (default: 100)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: crc_benchmark_results_TIMESTAMP)")
    parser.add_argument("--max-eval-queries", type=int, default=1000,
                       help="Maximum number of test queries to evaluate (default: 1000)")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["fb15k-237", "nell-955", "yago310"],
                       help="Dataset name: fb15k-237, nell-955, or yago310 (required for dataset-specific calibration data)")
    parser.add_argument("--incompleteness", type=int, default=None,
                       help="Data incompleteness level (e.g., 20 for 20% missing)")
    
    # Parse known args and collect any extra args to pass through
    args, extra_args = parser.parse_known_args()
    
    # Validate dataset is provided
    if args.dataset is None:
        parser.error("--dataset argument is required. Supported datasets: fb15k-237, nell-955, yago310")
    
    # Extract model_to_infer from extra_args
    model_to_infer = None
    for i, arg in enumerate(extra_args):
        if arg == "--model_to_infer" and i + 1 < len(extra_args):
            model_to_infer = extra_args[i + 1]
            break
    
    # Validate incompleteness is provided
    if args.incompleteness is None:
        parser.error("--incompleteness argument is required. Please specify the data incompleteness level (e.g., 20 for 20% missing)")
    
    # Validate model_to_infer is provided
    if model_to_infer is None:
        parser.error("--model_to_infer argument is required in extra_args. Please specify the model to infer (e.g., ThreeHopPipeline)")
    
    # Get repo root (script is in src/, go up one level)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Set output directory
    if args.output_dir is None:
        # Format: conrad_bench_<dataset>_<model_to_infer>_<incompleteness>
        folder_name = f"conrad_bench_{args.dataset}_{model_to_infer}_{args.incompleteness}"
        # Write to artifacts/benchmark/
        benchmark_dir = os.path.join(repo_root, "artifacts", "benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        args.output_dir = os.path.join(benchmark_dir, folder_name)
    else:
        # If output_dir is provided, ensure it's an absolute path or relative to repo root
        if not os.path.isabs(args.output_dir):
            args.output_dir = os.path.join(repo_root, args.output_dir)
    
    print("="*80)
    print("AUTOMATED CRC BENCHMARK SWEEP")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model to infer: {model_to_infer}")
    print(f"Incompleteness level: {args.incompleteness}%")
    print(f"Confidence levels: {args.confidence_levels}")
    print(f"Max calibration queries: {args.max_calibration_queries}")
    print(f"Output directory: {args.output_dir}")
    if extra_args:
        print(f"Extra args passed through: {' '.join(extra_args)}")
    print("="*80)
    print()
    
    # Ensure dataset is in extra_args if not already present
    if args.dataset and "--dataset" not in ' '.join(extra_args):
        extra_args = extra_args + ["--dataset", args.dataset]
    
    # Run the sweep
    results_csv = run_benchmark_sweep(
        confidence_levels=args.confidence_levels,
        output_dir=args.output_dir,
        max_calibration_queries=args.max_calibration_queries,
        max_eval_queries=args.max_eval_queries,
        extra_args=extra_args,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()

