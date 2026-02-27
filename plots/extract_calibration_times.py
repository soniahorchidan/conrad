#!/usr/bin/env python3
"""
Extract calibration times from calibration.log files and create a table.

The table shows dataset × template with wall-clock seconds for the offline calibration phase.
Breaks down into: score computation time and lambda optimization time.
"""

import re
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def parse_timestamp(line):
    """Extract timestamp from log line."""
    # Format: [2026-02-16 14:20:13][...]
    match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
    return None

def extract_dataset_template_from_path(log_path):
    """Extract dataset, template, and sparsity level from log file path."""
    # Path format: artifacts/benchmark_*/conrad_bench_{dataset}_{Template}_{incompleteness}/calibration.log
    
    parts = Path(log_path).parts
    # Find the directory containing calibration.log
    for i, part in enumerate(parts):
        if part == 'calibration.log':
            dir_name = parts[i-1]
            break
    else:
        return None, None, None
    
    # Parse: conrad_bench_{dataset}_{Template}_{incompleteness}
    # Handle cases like "fb15k-237", "nell-955", "yago310"
    match = re.match(r'conrad_bench_(.+?)_(.+?)_(\d+)', dir_name)
    if match:
        dataset = match.group(1)
        template = match.group(2)
        sparsity = int(match.group(3))
        return dataset, template, sparsity
    
    return None, None, None

def calculate_calibration_time(log_path):
    """
    Calculate wall-clock times for calibration phases from log file.
    
    Returns:
        dict with keys: 'total', 'score_computation', 'optimization'
        or None if times cannot be determined
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None
    
    # Find key markers
    start_prepare_time = None
    start_optimize_time = None
    end_optimize_time = None
    
    for line in lines:
        timestamp = parse_timestamp(line)
        if timestamp is None:
            continue
        
        # Score computation phase: from "Starting prepare_calibrate" to "Preparing to calibrate"
        if "Starting prepare_calibrate" in line and start_prepare_time is None:
            start_prepare_time = timestamp
        
        # Optimization phase starts: "Preparing to calibrate model"
        if "Preparing to calibrate model" in line and start_optimize_time is None:
            start_optimize_time = timestamp
        
        # Optimization phase ends: "Calibration for model ... done!"
        if "Calibration for model" in line and "done!" in line and end_optimize_time is None:
            end_optimize_time = timestamp
    
    # Calculate times
    result = {}
    
    if start_prepare_time is not None:
        if start_optimize_time is not None:
            # Score computation time
            result['score_computation'] = (start_optimize_time - start_prepare_time).total_seconds()
        
        if end_optimize_time is not None:
            # Optimization time
            if start_optimize_time is not None:
                result['optimization'] = (end_optimize_time - start_optimize_time).total_seconds()
            
            # Total time
            result['total'] = (end_optimize_time - start_prepare_time).total_seconds()
    
    # Fallback: if we can't find markers, use first/last timestamp
    if not result:
        timestamps = [parse_timestamp(line) for line in lines]
        timestamps = [t for t in timestamps if t is not None]
        
        if len(timestamps) >= 2:
            result['total'] = (timestamps[-1] - timestamps[0]).total_seconds()
            result['score_computation'] = None
            result['optimization'] = None
    
    return result if result else None

def find_all_calibration_logs(base_dir="artifacts"):
    """Find all calibration.log files."""
    base_path = Path(base_dir)
    calibration_logs = list(base_path.rglob("calibration.log"))
    return calibration_logs

def main(sparsity_filter=None, dataset_filter=None):
    """
    Main function to extract calibration times and create table.
    
    Args:
        sparsity_filter: If provided (e.g., 20), only include results for that sparsity level.
                        If None, include all sparsity levels.
        dataset_filter: If provided (e.g., "fb15k-237"), only include results for that dataset.
                        If None, include all datasets.
    """
    base_dir = "artifacts"
    
    # Find all calibration logs
    print(f"Searching for calibration.log files in {base_dir}...")
    filters = []
    if sparsity_filter is not None:
        filters.append(f"{sparsity_filter}% sparsity")
    if dataset_filter is not None:
        filters.append(f"dataset: {dataset_filter}")
    if filters:
        print(f"Filtering for: {', '.join(filters)}")
    log_files = find_all_calibration_logs(base_dir)
    print(f"Found {len(log_files)} calibration.log files")
    
    # Collect data: (dataset, template) -> list of times
    data = defaultdict(list)
    
    for log_path in log_files:
        dataset, template, sparsity = extract_dataset_template_from_path(str(log_path))
        if dataset is None or template is None or sparsity is None:
            print(f"Warning: Could not parse dataset/template/sparsity from {log_path}")
            continue
        
        # Filter by sparsity if requested
        if sparsity_filter is not None and sparsity != sparsity_filter:
            continue
        
        # Filter by dataset if requested
        if dataset_filter is not None and dataset != dataset_filter:
            continue
        
        time_dict = calculate_calibration_time(log_path)
        if time_dict is None:
            print(f"Warning: Could not calculate time for {log_path}")
            continue
        
        data[(dataset, template)].append(time_dict)
    
    # Create table
    # Get unique datasets and templates
    datasets = sorted(set(d for d, t in data.keys()))
    templates = sorted(set(t for d, t in data.keys()))
    
    # Create table data - separate tables for score computation and optimization
    table_data_score = []
    table_data_optim = []
    table_data_total = []
    
    for dataset in datasets:
        row_score = {'Dataset': dataset}
        row_optim = {'Dataset': dataset}
        row_total = {'Dataset': dataset}
        
        for template in templates:
            key = (dataset, template)
            if key in data:
                # Average if multiple runs, otherwise use single value
                time_dicts = data[key]
                
                # Average score computation times
                score_times = [d.get('score_computation') for d in time_dicts if d.get('score_computation') is not None]
                if score_times:
                    row_score[template] = sum(score_times) / len(score_times)
                else:
                    row_score[template] = None
                
                # Average optimization times
                optim_times = [d.get('optimization') for d in time_dicts if d.get('optimization') is not None]
                if optim_times:
                    row_optim[template] = sum(optim_times) / len(optim_times)
                else:
                    row_optim[template] = None
                
                # Average total times
                total_times = [d.get('total') for d in time_dicts if d.get('total') is not None]
                if total_times:
                    row_total[template] = sum(total_times) / len(total_times)
                else:
                    row_total[template] = None
            else:
                row_score[template] = None
                row_optim[template] = None
                row_total[template] = None
        
        table_data_score.append(row_score)
        table_data_optim.append(row_optim)
        table_data_total.append(row_total)
    
    # Helper function to print a table
    def print_table(table_data, title):
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        # Print header
        header = "Dataset"
        for template in templates:
            header += f"\t{template}"
        print(header)
        print("-" * len(header.replace('\t', '  ')))
        
        # Print rows
        for row in table_data:
            line = row['Dataset']
            for template in templates:
                val = row[template]
                if val is not None:
                    line += f"\t{val:.1f}"
                else:
                    line += "\t-"
            print(line)
        print("="*80)
    
    # Print all three tables
    title_parts = []
    if dataset_filter:
        title_parts.append(f"Dataset: {dataset_filter}")
    if sparsity_filter is not None:
        title_parts.append(f"{sparsity_filter}% Sparsity")
    title_suffix = f" - {' | '.join(title_parts)}" if title_parts else ""
    
    print_table(table_data_score, f"CALIBRATION SCORE COMPUTATION TIME (wall-clock seconds){title_suffix}")
    print_table(table_data_optim, f"LAMBDA OPTIMIZATION TIME (wall-clock seconds){title_suffix}")
    print_table(table_data_total, f"TOTAL CALIBRATION TIME (wall-clock seconds){title_suffix}")
    
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Templates: {len(templates)}")
    print(f"  Total entries: {sum(len(times) for times in data.values())}")

if __name__ == "__main__":
    import sys
    # Check if filters are provided as command line arguments
    sparsity_filter = None
    dataset_filter = None
    
    if len(sys.argv) > 1:
        try:
            sparsity_filter = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid sparsity value '{sys.argv[1]}'")
    
    if len(sys.argv) > 2:
        dataset_filter = sys.argv[2]
    
    main(sparsity_filter=sparsity_filter, dataset_filter=dataset_filter)
