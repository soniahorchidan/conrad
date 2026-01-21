#!/usr/bin/env python3
"""
Fully automated CRC benchmark sweep across multiple confidence levels.
Automatically calibrates lambdas and runs benchmark queries.
"""

import os
import sys
import subprocess
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple
import ast


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


def extract_calibrated_lambdas(log_path: str) -> Dict[float, List[float]]:
    """Extract calibrated lambda values from calibration log."""
    print(f"\nExtracting calibrated lambdas from {log_path}...")
    
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    lambdas_dict = {}
    
    # Search for the line containing 'calibrated_alphas'
    for line in log_content.split('\n'):
        if "'calibrated_alphas':" in line or '"calibrated_alphas":' in line:
            print(f"Found calibrated_alphas line")
            
            # Extract the dictionary part after 'calibrated_alphas':
            # Format: 'calibrated_alphas': {0.2: array([...]), 0.3: array([...]), ...}
            
            # Find the start of the dictionary
            dict_start = line.find("'calibrated_alphas':") + len("'calibrated_alphas':")
            if dict_start == len("'calibrated_alphas':") - 1:
                dict_start = line.find('"calibrated_alphas":') + len('"calibrated_alphas":')
            
            # Extract everything from the opening brace to handle nested structures
            dict_part = line[dict_start:].strip()
            
            # Use regex to find all confidence: array([...]) pairs
            # Pattern matches: number: array([numbers]) with optional dtype parameter
            # Handles both array([...]) and array([...], dtype=float32) formats
            pattern = r'([\d.]+):\s*array\(\[([^\]]+)\](?:,\s*dtype=\w+)?\)'
            matches = re.findall(pattern, dict_part)
            
            for alpha_str, values_str in matches:
                try:
                    alpha = float(alpha_str)
                    # Split values and handle scientific notation
                    values = []
                    for v in values_str.split(','):
                        v = v.strip()
                        if v:  # Skip empty strings
                            values.append(float(v))
                    
                    # Accept any number of values (2 for TwoUnionPipeline, 3 for ThreeHopPipeline, etc.)
                    if len(values) >= 2:
                        lambdas_dict[alpha] = values
                        print(f"  Extracted alpha={alpha}: {values}")
                    else:
                        print(f"  WARNING: Too few values ({len(values)}) for alpha={alpha}")
                except ValueError as e:
                    print(f"  WARNING: Could not parse alpha={alpha_str}: {e}")
            
            # If we found any lambdas, we're done
            if lambdas_dict:
                break
    
    if lambdas_dict:
        print(f"\nSuccessfully extracted lambdas for {len(lambdas_dict)} confidence levels:")
        for alpha in sorted(lambdas_dict.keys()):
            print(f"  {alpha}: {lambdas_dict[alpha]}")
        return lambdas_dict
    else:
        # Print more debug info
        print("\nERROR: Could not extract calibrated lambdas from log.")
        print("Searching for 'Calibration completed' in log...")
        for i, line in enumerate(log_content.split('\n')):
            if 'Calibration completed' in line or 'calibrated_alphas' in line:
                print(f"Line {i}: {line[:300]}")
        raise ValueError("Could not extract calibrated lambdas from log. Please check the calibration log.")


def run_benchmark_sweep(confidence_levels: List[float], output_dir: str, 
                       max_calibration_queries: int = 1600,
                       max_eval_queries: int = 1000,
                       extra_args: List[str] = None,
                       skip_calibration: bool = False,
                       dataset: str = None) -> str:
    """
    Run full benchmark sweep with automatic lambda calibration.
    
    Args:
        extra_args: Additional arguments to pass through to validate_crc_composition.py
        skip_calibration: If True, skip calibration and use existing lambda file from output directory
    
    Returns:
        Path to results CSV file
    """
    if extra_args is None:
        extra_args = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if not skip_calibration:
        # Step 1: Run calibration
        print("="*80)
        print("STEP 1: CALIBRATION")
        print("="*80)
        
        calibration_log = os.path.join(output_dir, "calibration.log")
        calibration_cmd = [
            "python", "validate_crc_composition.py",
            "--mode", "calibration",
            "--confidence", "0.8",  # Calibrates for all alphas, but needs test for one confidence level
            "--max-calibration-queries", str(max_calibration_queries),
        ] + extra_args
        # Ensure dataset is included if provided
        if dataset and "--dataset" not in ' '.join(extra_args):
            calibration_cmd.extend(["--dataset", dataset])
        
        returncode, output = run_command(calibration_cmd, calibration_log)
        
        if returncode != 0:
            print(f"ERROR: Calibration failed with return code {returncode}")
            sys.exit(1)
        
        # Step 2: Extract lambdas
        print("\n" + "="*80)
        print("STEP 2: EXTRACTING CALIBRATED LAMBDAS")
        print("="*80)
        
        try:
            lambdas_dict = extract_calibrated_lambdas(calibration_log)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        
        # Save extracted lambdas to JSON
        lambdas_json_path = os.path.join(output_dir, "calibrated_lambdas.json")
        with open(lambdas_json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_dict = {str(k): v for k, v in lambdas_dict.items()}
            json.dump(json_dict, f, indent=2)
        print(f"Lambdas saved to: {lambdas_json_path}")
    else:
        print("="*80)
        print("SKIPPING CALIBRATION - Using existing lambda file")
        print("="*80)
        print()
        # Try to find existing lambda file in output directory
        lambdas_json_path = os.path.join(output_dir, "calibrated_lambdas.json")
        if not os.path.exists(lambdas_json_path):
            print(f"ERROR: Lambda file not found at {lambdas_json_path}")
            print("Please provide a valid lambda file or run calibration first.")
            sys.exit(1)
        print(f"Using existing lambda file: {lambdas_json_path}")
    
    # Step 3: Run benchmarks for each confidence level
    print("\n" + "="*80)
    print("STEP 3: RUNNING BENCHMARK QUERIES")
    print("="*80)
    
    # Ensure we have a lambda file path
    if not skip_calibration:
        # lambdas_json_path was set above
        pass
    else:
        # lambdas_json_path should have been set in the else block above
        if 'lambdas_json_path' not in locals():
            lambdas_json_path = os.path.join(output_dir, "calibrated_lambdas.json")
    
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
            "--mode", "benchmark",
            "--confidence", str(conf),
            "--lambdas-file", lambdas_json_path,
            "--max-eval-queries", str(max_eval_queries)
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
    parser.add_argument("--skip-calibration", action="store_true",
                       help="Skip calibration step and use existing lambda file from output directory")
    parser.add_argument("--max-eval-queries", type=int, default=1000,
                       help="Maximum number of test queries to evaluate (default: 1000)")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["fb15k-237", "nell-955"],
                       help="Dataset name: fb15k-237 or nell-955 (required for dataset-specific calibration data)")
    
    # Parse known args and collect any extra args to pass through
    args, extra_args = parser.parse_known_args()
    
    # Validate dataset is provided
    if args.dataset is None:
        parser.error("--dataset argument is required. Supported datasets: fb15k-237, nell-955")
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"crc_benchmark_results_{timestamp}"
    
    print("="*80)
    print("AUTOMATED CRC BENCHMARK SWEEP")
    print("="*80)
    print(f"Dataset: {args.dataset}")
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
        skip_calibration=args.skip_calibration,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()

