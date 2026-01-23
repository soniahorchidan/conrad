import os
import argparse
import logging
import sys
import re
import pickle
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

from inference import ModelFactory, parse_args_inference
from utils import merge_args, set_logger, parse_time, get_graph
from graph_handler import Neo4JBackendDBController
from conformal_prediction.utils import compute_fnr_metrics
from conformal_prediction.validate_conformal_risk_control import get_dataset_statistics


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    confidence: float = 0.8
    max_queries_per_file: int = 1000
    query_start_index: int = 0
    calibration_threshold: float = 0.7
    
    # Dataset name (e.g., "fb15k-237" or "nell-955")
    dataset: str = None
    
    # Base directories - will be constructed based on dataset
    calibration_data_base: str = None
    test_data_base: str = None
    
    # Query type - set based on model type
    query_type: str = "3p_pipeline"  # Options: "3p_pipeline", "2ip_pipeline", "2u_pipeline"
    
    calibration_data_path: str = None  # Will be: calibration_data_base/query_type
    test_queries_path: str = None      # Will be: test_data_base/query_type
    
    max_calibration_queries: int = 100  # Limit calibration queries for testing
    max_gt_size: int = 1000  # Skip queries with >1000 GT hop3 entities (must match calibration filter!)
    lambdas_file_path: str = None  # Path to JSON file containing pre-calibrated lambdas (if provided, skips calibration)
    
    # Model to query type mapping
    MODEL_TO_QUERY_TYPE = {
        "threehoppipeline": "3p_pipeline",
        "twounionpipeline": "2u_pipeline",
        "twointersectprojectpipeline": "2ip_pipeline",
    }
    
    def load_lambdas_from_file(self) -> Dict[float, np.ndarray]:
        """
        Load lambda values from JSON file.
        
        Returns:
            Dictionary mapping confidence levels (float) to numpy arrays of lambda values.
            NOTE: The number of values depends on the pipeline type:
              - ThreeHopPipeline: 3 values [hop1, hop2, hop3]
              - TwoUnionPipeline: 2 values [branch1, branch2]
              - TwoIntersectProjectPipeline: 3 values [branch1, branch2, projection]
        
        Raises:
            ValueError: If lambda file is not found or cannot be loaded.
        """
        if not self.lambdas_file_path:
            raise ValueError("lambdas_file_path must be set to load lambdas from file")
        
        if not os.path.exists(self.lambdas_file_path):
            raise ValueError(f"Lambda file not found: {self.lambdas_file_path}")
        
        try:
            with open(self.lambdas_file_path, 'r') as f:
                lambdas_dict = json.load(f)
            
            # Convert string keys back to floats and lists to numpy arrays
            result = {}
            for key, value in lambdas_dict.items():
                alpha = float(key)
                result[alpha] = np.array(value)
            
            logging.info(f"Loaded lambdas from {self.lambdas_file_path} for {len(result)} confidence levels")
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from lambda file {self.lambdas_file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load lambdas from {self.lambdas_file_path}: {e}")
    
    def __post_init__(self):
        """Construct derived paths after initialization."""
        # Get repo root (assuming script is in src/, go up one level)
        if self.dataset:
            # Normalize dataset name for folder (remove hyphens)
            dataset_normalized = self.dataset.replace("-", "")
            
            # Get repo root: go up from src/ to repo root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            
            # Construct dataset-specific paths
            if self.calibration_data_base is None:
                self.calibration_data_base = os.path.join(repo_root, "artifacts", "queries", dataset_normalized, "calibration")
            if self.test_data_base is None:
                self.test_data_base = os.path.join(repo_root, "artifacts", "queries", dataset_normalized, "test")
        else:
            logging.error("Dataset not specified, using default paths")
            raise ValueError("Dataset not specified")
        
        # Construct final paths with query type
        if self.calibration_data_path is None:
            self.calibration_data_path = os.path.join(self.calibration_data_base, self.query_type)
        if self.test_queries_path is None:
            self.test_queries_path = os.path.join(self.test_data_base, self.query_type)


@dataclass
class QueryResult:
    """Container for query evaluation results."""
    precision: float
    recall: float
    f1: float
    predicted_values: List[int]
    ground_truth: List[int]
    is_abstained: bool = False  # True if no prediction was made (empty prediction set)
    query_time_ms: float = 0.0  # Query execution time in milliseconds


class CRCBenchmarkValidator:
    """Validates Conformal Risk Control using calibration and benchmark queries."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.target_queries = ["queries"]
        self.all_results: Dict[str, Dict[str, List[QueryResult]]] = {}
        self.calibration_results: List[QueryResult] = []
    
    def setup_model_and_calibration(self, inf_args) -> Tuple[ModelFactory, Any]:
        """Setup model and perform calibration."""
        logging.info("Setting up model and starting calibration")
        logging.info(f"Model: {inf_args.model_to_infer}, Device: {inf_args.device}")
        
        model_factory_pipeline = ModelFactory(inf_args)
        model_factory_pipeline.prepare_model()
        
        db_controller = Neo4JBackendDBController(
            f"neo4j://{inf_args.neo4j_host}:{inf_args.neo4j_bolt_port}",
            inf_args.node_unique_id,
            inf_args.relation_unique_id,
        )
        
        if self.config.lambdas_file_path:
            logging.info("Using pre-calibrated lambda values from file (skipping calibration)")
            try:
                lambdas_dict = self.config.load_lambdas_from_file()
                logging.info(f"Loaded lambdas for {len(lambdas_dict)} confidence levels")
                
                # Set the metadata on the conformal prediction object
                metadata_calibrate = {
                    'calibrated_alphas': lambdas_dict,
                    'vector_scores': True,
                    'num_hops': 3,
                    'note': f'Using pre-calibrated lambda values from {self.config.lambdas_file_path}'
                }
                model_factory_pipeline.model.conformal_prediction.metadata = metadata_calibrate
                logging.info(f"Pre-calibrated lambdas loaded: {metadata_calibrate}")
            except Exception as e:
                logging.error(f"Failed to load lambdas from file: {e}")
                raise
        else:
            metadata_calibrate = model_factory_pipeline.model.conformal_prediction.prepare_calibrate(
                inf_args.load_path,
                db_controller,
                threshold=self.config.calibration_threshold,
                calibration_data_path=self.config.calibration_data_path
            )
            logging.info(f"Calibration completed: {metadata_calibrate}")

        return model_factory_pipeline, db_controller
    
    def parse_query(self, query: str) -> Tuple[Any, List[int], float]:
        """
        Parse query string to extract query components.
        
        Supports:
        - 3p queries: query((entity_id, (rel1, rel2, rel3)))
        - 2u queries: query(((anchor1, rel1), (anchor2, rel2), '2u'))
        - 2ip queries: query(((anchor1, rel1), (anchor2, rel2), rel3, '2ip'))
        
        Returns:
            Tuple of (entity_id_or_tuple, rel_types, confidence)
        """
        # Try 3p format: query((entity_id, (rel1, rel2, rel3)))
        simple_3p_match = re.search(r'query\(\((\d+),\s*\((\d+),\s*(\d+),\s*(\d+)\)\)\)', query)
        if simple_3p_match:
            entity_id = int(simple_3p_match.group(1))
            rel_types = [int(simple_3p_match.group(2)), int(simple_3p_match.group(3)), int(simple_3p_match.group(4))]
            return entity_id, rel_types, 0.7
        
        # Try 2ip format: query(((anchor1, rel1), (anchor2, rel2), rel3, '2ip'))
        simple_2ip_match = re.search(r'query\(\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\),\s*(\d+),\s*[\'"]2ip[\'"]\)\)', query)
        if simple_2ip_match:
            anchor1 = int(simple_2ip_match.group(1))
            rel1 = int(simple_2ip_match.group(2))
            anchor2 = int(simple_2ip_match.group(3))
            rel2 = int(simple_2ip_match.group(4))
            rel3 = int(simple_2ip_match.group(5))
            # Return as tuple for 2ip queries: (anchor1, anchor2), [rel1, rel2, rel3]
            return (anchor1, anchor2), [rel1, rel2, rel3], 0.7
        
        # Try 2u format: query(((anchor1, rel1), (anchor2, rel2), '2u'))
        simple_2u_match = re.search(r'query\(\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\),\s*[\'"]2u[\'"]\)\)', query)
        if simple_2u_match:
            anchor1 = int(simple_2u_match.group(1))
            rel1 = int(simple_2u_match.group(2))
            anchor2 = int(simple_2u_match.group(3))
            rel2 = int(simple_2u_match.group(4))
            # Return as tuple for 2u queries: (anchor1, anchor2), [rel1, rel2]
            return (anchor1, anchor2), [rel1, rel2], 0.7
        
        # Try verbose format (legacy)
        entity_match = re.search(r'Entity \{id: (\d+)\}', query)
        if not entity_match:
            raise ValueError(f"Could not extract entity ID from query: {query}")
        entity_id = int(entity_match.group(1))
        
        relation_matches = re.findall(r'Relation \{type: (\d+)\}', query)
        if len(relation_matches) != 3:
            raise ValueError(f"Expected 3 relations, found {len(relation_matches)} in query: {query}")
        
        rel_types = [int(r) for r in relation_matches]
        
        confidence_match = re.search(r'WITH AT LEAST ([\d.]+) CONFIDENCE', query)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.7
        
        return entity_id, rel_types, confidence
    
    def run_prediction(self, model, query_tensor: torch.Tensor, 
                       confidence: float, graph_data: Any) -> List[List[int]]:
        """Run model prediction. Returns list of predictions (one per query in batch)."""
        with torch.no_grad():
            results = model.predict(query_tensor, confidence, graph_data)
        
        if isinstance(results, list):
            # Handle batched results
            if len(results) > 0 and isinstance(results[0], list):
                # Results is already a list of lists
                return results
            elif len(results) > 0:
                # Single result, wrap in list
                return [results]
            else:
                return []
        else:
            raise ValueError(f"Unexpected results format: {type(results)}")
    
    def _convert_to_int_list(self, values) -> List[int]:
        """Convert prediction values (tensor, numpy, or list) to list of integers."""
        if isinstance(values, torch.Tensor):
            return values.cpu().tolist()
        elif isinstance(values, list):
            converted = []
            for item in values:
                if isinstance(item, torch.Tensor):
                    converted.append(item.cpu().item())
                elif hasattr(item, 'item'):  # numpy scalar
                    converted.append(int(item.item()))
                else:
                    converted.append(int(item))
            return converted
        return values
    
    def load_ground_truth(self) -> List[List[int]]:
        """Load ground truth values."""
        gt_file = os.path.join(self.config.test_queries_path, "gt")
        
        gt_values = []
        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse space-separated integers
                    gt_values.append([int(x) for x in line.split()])
                else:
                    # Empty line = no ground truth for this query
                    gt_values.append([])
        
        return gt_values

    
    def process_single_query(self, query: str, model, graph_data: Any, 
                           ground_truth: List[int], override_confidence: float = None) -> Optional[QueryResult]:
        """Process a single query and return results."""
        try:
            entity_or_tuple, rel_types, confidence = self.parse_query(query)
            if override_confidence is not None:
                confidence = override_confidence
            
            query_tensor = self._build_query_tensor(entity_or_tuple, rel_types)
            
            # Validate tensor shape based on query type
            expected_shape = (1, 5) if len(rel_types) == 3 and isinstance(entity_or_tuple, tuple) else (1, 4)
            assert query_tensor.shape == expected_shape, f"Expected tensor shape {expected_shape}, got {query_tensor.shape}"
            
            start_time = time.time()
            predicted_batch = self.run_prediction(model, query_tensor, confidence, graph_data)
            query_time_ms = (time.time() - start_time) * 1000
            
            predicted_values = self._convert_to_int_list(predicted_batch[0]) if predicted_batch else []
            ground_truth = [int(x) for x in ground_truth] if ground_truth else []
            
            is_abstained = len(predicted_values) == 0
            fnr, precision, f1 = compute_fnr_metrics([predicted_values], [ground_truth])
            recall = 1 - fnr
            
            return QueryResult(precision, recall, f1, predicted_values, ground_truth, is_abstained, query_time_ms)
            
        except (ValueError, TimeoutError, Exception) as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return None
    
    def process_queries(self, model, graph_data: Any, inference_batch_size: int = 1):
        """Process all queries for the configured query type. Uses batching if batch_size > 1."""
        if inference_batch_size > 1:
            return self.process_queries_batched(model, graph_data, inference_batch_size)
        
        # Original single-query processing
        start_time = time.time()
        test_queries_path = self.config.test_queries_path
        query_type = self.config.query_type
        self.all_results[query_type] = {}
        
        for query_file in self.target_queries:
            print(f"Processing query file: {query_file}")
            self.all_results[query_type][query_file] = []
            
            # Load queries from file
            query_path = os.path.join(test_queries_path, query_file)
            with open(query_path, "r") as f:
                queries = f.readlines()
            
            # Load ground truth
            gt_values = self.load_ground_truth()

            # Process queries
            start_idx = self.config.query_start_index
            end_idx = start_idx + self.config.max_queries_per_file
            
            for i, query in enumerate(queries[start_idx:end_idx]):
                query = query.strip()
                if not query:
                    continue
                
                print(f"Running query {i+1}/{self.config.max_queries_per_file} from {query_file}")
                print(f"Query: {query}")
                
                gt = gt_values[i] if i < len(gt_values) else []

                result = self.process_single_query(query, model, graph_data, gt, override_confidence=self.config.confidence)
                
                if result:
                    self.all_results[query_type][query_file].append(result)
                    print(f"Query {i+1} - Precision: {result.precision:.4f}, Recall: {result.recall:.4f}, F1: {result.f1:.4f} | pred: {len(result.predicted_values)}, GT: {len(result.ground_truth)} | Time: {result.query_time_ms:.2f}ms")
        
        total_time_seconds = time.time() - start_time
        total_time_ms = total_time_seconds * 1000
        total_queries_processed = sum(len(results) for results in self.all_results[query_type].values())
        logging.info(f"Total time to process all benchmark queries: {total_time_seconds:.2f}s ({total_time_ms:.2f}ms) for {total_queries_processed} queries")
        print(f"\nTotal time to process all benchmark queries: {total_time_seconds:.2f}s ({total_time_ms:.2f}ms) for {total_queries_processed} queries")
    
    def _build_query_tensor(self, entity_or_tuple: Any, rel_types: List[int]) -> torch.Tensor:
        """Build query tensor from parsed query components."""
        if isinstance(entity_or_tuple, tuple):
            anchor1, anchor2 = entity_or_tuple
            if len(rel_types) == 2:
                return torch.tensor([[anchor1, rel_types[0], anchor2, rel_types[1]]], dtype=torch.long)
            elif len(rel_types) == 3:
                return torch.tensor([[anchor1, rel_types[0], anchor2, rel_types[1], rel_types[2]]], dtype=torch.long)
            else:
                raise ValueError(f"Unexpected number of relations for tuple query: {len(rel_types)}")
        else:
            return torch.tensor([[entity_or_tuple] + rel_types], dtype=torch.long)
    
    def _build_batch_tensor(self, queries: List[str], override_confidence: float = None) -> Tuple[torch.Tensor, float]:
        """Parse queries and build a batched query tensor."""
        query_tensors = []
        for query in queries:
            entity_or_tuple, rel_types, confidence = self.parse_query(query)
            if override_confidence is not None:
                confidence = override_confidence
            query_tensor = self._build_query_tensor(entity_or_tuple, rel_types)
            query_tensors.append(query_tensor)
        
        batch_query = torch.cat(query_tensors, dim=0)
        batch_confidence = override_confidence or self.config.confidence
        return batch_query, batch_confidence
    
    def _convert_predictions_to_results(self, predicted_batch: List[List[int]], 
                                       ground_truths: List[List[int]], 
                                       batch_time_ms: float) -> List[QueryResult]:
        """Convert batch predictions to QueryResult objects."""
        results = []
        for pred, gt in zip(predicted_batch, ground_truths):
            pred = self._convert_to_int_list(pred)
            gt = [int(x) for x in gt] if gt else []
            is_abstained = len(pred) == 0
            
            fnr, precision, f1 = compute_fnr_metrics([pred], [gt])
            recall = 1 - fnr
            
            query_time_ms = batch_time_ms / len(predicted_batch) if len(predicted_batch) > 0 else 0
            results.append(QueryResult(precision, recall, f1, pred, gt, is_abstained, query_time_ms))
        return results
    
    def _load_queries_from_file(self, query_file: str) -> List[str]:
        """Load and filter queries from a file."""
        query_path = os.path.join(self.config.test_queries_path, query_file)
        with open(query_path, "r") as f:
            queries = f.readlines()
        
        start_idx = self.config.query_start_index
        end_idx = start_idx + self.config.max_queries_per_file
        return [q.strip() for q in queries[start_idx:end_idx] if q.strip()]
    
    def _log_gpu_info(self, batch_size: int):
        """Log GPU availability information."""
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            logging.info(f"Detected {num_gpus} GPUs. Batched inference will leverage multi-GPU.")
        else:
            logging.info(f"Using batched inference with batch size {batch_size} on single GPU/CPU")
    
    def _log_batch_progress(self, batch_results: List[Optional[QueryResult]], 
                           batch_start: int, total_queries: int):
        """Log progress for a batch of queries."""
        for i, result in enumerate(batch_results):
            query_idx = batch_start + i + 1
            if result:
                print(f"Query {query_idx}/{total_queries} - "
                      f"Precision: {result.precision:.4f}, Recall: {result.recall:.4f}, "
                      f"F1: {result.f1:.4f} | pred: {len(result.predicted_values)}, "
                      f"GT: {len(result.ground_truth)} | Time: {result.query_time_ms:.2f}ms")
    
    def _log_summary(self, start_time: float, query_type: str):
        """Log summary statistics after processing all queries."""
        total_time_seconds = time.time() - start_time
        total_queries_processed = sum(len(results) for results in self.all_results[query_type].values())
        avg_time_per_query = total_time_seconds / total_queries_processed if total_queries_processed > 0 else 0
        logging.info(f"Total time to process all benchmark queries: {total_time_seconds:.2f}s for {total_queries_processed} queries "
                    f"({avg_time_per_query:.3f}s per query)")
        print(f"\nTotal time to process all benchmark queries: {total_time_seconds:.2f}s for {total_queries_processed} queries "
              f"({avg_time_per_query:.3f}s per query)")
    
    def process_queries_batched(self, model, graph_data: Any, batch_size: int = 8):
        """Process queries in batches for parallel inference."""
        start_time = time.time()
        query_type = self.config.query_type
        self.all_results[query_type] = {}
        
        self._log_gpu_info(batch_size)
        
        for query_file in self.target_queries:
            print(f"Processing query file: {query_file}")
            self.all_results[query_type][query_file] = []
            
            query_subset = self._load_queries_from_file(query_file)
            gt_values = self.load_ground_truth()
            
            for batch_start in range(0, len(query_subset), batch_size):
                batch_end = min(batch_start + batch_size, len(query_subset))
                batch_queries = query_subset[batch_start:batch_end]
                batch_gt = [gt_values[batch_start + i] if (batch_start + i) < len(gt_values) else [] 
                            for i in range(len(batch_queries))]
                
                batch_results = self.process_batch_queries(
                    batch_queries, model, graph_data, batch_gt, 
                    override_confidence=self.config.confidence
                )
                
                self.all_results[query_type][query_file].extend(batch_results)
                self._log_batch_progress(batch_results, batch_start, len(query_subset))
        
        self._log_summary(start_time, query_type)
    
    def process_batch_queries(self, queries: List[str], model, graph_data: Any, 
                              ground_truths: List[List[int]], override_confidence: float = None) -> List[Optional[QueryResult]]:
        """Process a batch of queries together."""
        if not queries:
            return []
        
        try:
            batch_query, batch_confidence = self._build_batch_tensor(queries, override_confidence)
            
            batch_start_time = time.time()
            predicted_batch = self.run_prediction(model, batch_query, batch_confidence, graph_data)
            batch_time_ms = (time.time() - batch_start_time) * 1000
            
            return self._convert_predictions_to_results(predicted_batch, ground_truths, batch_time_ms)
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
            return [None] * len(queries)
    
    def print_summary_statistics(self):
        """Print summary statistics for all processed queries."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Use the query_type directly
        query_type = self.config.query_type
        print(f"\nQUERY TYPE: {query_type}")
        
        for query_file in self.target_queries:
            if query_type in self.all_results and query_file in self.all_results[query_type]:
                results = self.all_results[query_type][query_file]
                if results:
                    # All queries metrics (including abstained)
                    avg_precision = sum(r.precision for r in results) / len(results)
                    avg_recall = sum(r.recall for r in results) / len(results)
                    avg_f1 = sum(r.f1 for r in results) / len(results)
                    avg_time_ms = sum(r.query_time_ms for r in results) / len(results)
                    
                    # Abstention statistics
                    total_count = len(results)
                    abstained_count = sum(1 for r in results if r.is_abstained)
                    abstention_rate = abstained_count / total_count if total_count > 0 else 0.0
                    
                    # Non-abstained metrics
                    non_abstained = [r for r in results if not r.is_abstained]
                    if non_abstained:
                        non_abs_precision = sum(r.precision for r in non_abstained) / len(non_abstained)
                        non_abs_recall = sum(r.recall for r in non_abstained) / len(non_abstained)
                        non_abs_f1 = sum(r.f1 for r in non_abstained) / len(non_abstained)
                        non_abs_avg_time_ms = sum(r.query_time_ms for r in non_abstained) / len(non_abstained)
                        print(f"{query_file}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f} | Abstention={abstention_rate:.4f} | Avg Time: {avg_time_ms:.2f}ms")
                        print(f"  (Non-abstained only: Precision={non_abs_precision:.4f}, Recall={non_abs_recall:.4f}, F1={non_abs_f1:.4f} | Avg Time: {non_abs_avg_time_ms:.2f}ms)")
                    else:
                        print(f"{query_file}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f} | Abstention={abstention_rate:.4f} | Avg Time: {avg_time_ms:.2f}ms")
                else:
                    print(f"{query_file}: No valid results")
            else:
                print(f"{query_file}: No results found")
        
        # Calculate overall averages
        all_results_list = []
        for dataset, query_results in self.all_results.items():
            for query_file, results in query_results.items():
                if results:
                    all_results_list.extend(results)
        
        total_queries = len(all_results_list)
        
        if total_queries > 0:
            # All queries metrics
            overall_precision = sum(r.precision for r in all_results_list) / total_queries
            overall_recall = sum(r.recall for r in all_results_list) / total_queries
            overall_f1 = sum(r.f1 for r in all_results_list) / total_queries
            overall_avg_time_ms = sum(r.query_time_ms for r in all_results_list) / total_queries
            
            # Abstention statistics
            abstained_count = sum(1 for r in all_results_list if r.is_abstained)
            abstention_rate = abstained_count / total_queries
            
            print(f"\nOVERALL AVERAGE (All queries): Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={overall_f1:.4f} | Avg Time: {overall_avg_time_ms:.2f}ms")
            print(f"Abstention rate: {abstention_rate:.4f} ({abstained_count}/{total_queries})")
            
            # Non-abstained metrics
            non_abstained_results = [r for r in all_results_list if not r.is_abstained]
            if non_abstained_results:
                non_abs_precision = sum(r.precision for r in non_abstained_results) / len(non_abstained_results)
                non_abs_recall = sum(r.recall for r in non_abstained_results) / len(non_abstained_results)
                non_abs_f1 = sum(r.f1 for r in non_abstained_results) / len(non_abstained_results)
                non_abs_overall_avg_time_ms = sum(r.query_time_ms for r in non_abstained_results) / len(non_abstained_results)
                print(f"OVERALL AVERAGE (Non-abstained only): Precision={non_abs_precision:.4f}, Recall={non_abs_recall:.4f}, F1={non_abs_f1:.4f} | Avg Time: {non_abs_overall_avg_time_ms:.2f}ms")
                print(f"Non-abstained queries: {len(non_abstained_results)}/{total_queries}")
            else:
                print("All queries abstained!")
            
            print(f"Total queries processed: {total_queries}")
        
        print("="*80)
    
    def load_calibration_queries(self) -> Tuple[List[Tuple], Dict[Tuple, List[int]]]:
        """Load calibration queries and answers from disk."""
        queries_path = os.path.join(self.config.calibration_data_path, "queries.pkl")
        answers_path = os.path.join(self.config.calibration_data_path, "answers.pkl")
        
        # Load queries
        with open(queries_path, "rb") as f:
            queries = pickle.load(f)
        
        # Load final answers (hop 3 for 3p queries)
        with open(answers_path, "rb") as f:
            answers = pickle.load(f)
        
        return queries, answers
    
    def process_calibration_queries(self, model, graph_data: Any):
        """Process calibration queries similar to benchmark queries."""
        start_time = time.time()
        print("\n" + "="*80)
        print("PROCESSING CALIBRATION QUERIES")
        print("="*80)
        
        queries, answers = self.load_calibration_queries()
        print(f"Loaded {len(queries)} calibration queries")
        
        # Limit number of queries for testing
        max_queries = min(len(queries), self.config.max_calibration_queries)
        queries = queries[:max_queries]
        print(f"Processing first {max_queries} calibration queries")
        print(f"NOTE: Small sample size ({max_queries} queries) may show high variance from calibration FNR")
        
        self.calibration_results = []
        skipped_count = 0
        
        for i, query in enumerate(tqdm(queries, desc="Processing calibration queries")):
            # Get ground truth answers for this query FIRST (for filtering)
            ground_truth = answers.get(query, set())
            
            # Convert set to list of integers
            if isinstance(ground_truth, set):
                ground_truth = [int(x) for x in ground_truth]
            elif isinstance(ground_truth, list):
                ground_truth = [int(x) for x in ground_truth]
            else:
                ground_truth = []
            
            # Filter: Skip queries with too many GT entities (memory optimization)
            # This MUST match the calibration filter to maintain exchangeability!
            if len(ground_truth) > self.config.max_gt_size:
                skipped_count += 1
                print(f"Skipping calibration query {i+1} with {len(ground_truth)} GT entities (max={self.config.max_gt_size})")
                continue
            
            # Convert query format based on query type
            if self.config.query_type == "2u_pipeline":
                # 2u query: ((anchor1, rel1), (anchor2, rel2), "2u")
                (anchor1, rel1), (anchor2, rel2), _ = query
                query_tensor = torch.tensor([[anchor1, rel1, anchor2, rel2]], dtype=torch.long)
            elif self.config.query_type == "2ip_pipeline":
                # 2ip query: ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
                (anchor1, rel1), (anchor2, rel2), rel3, _ = query
                query_tensor = torch.tensor([[anchor1, rel1, anchor2, rel2, rel3]], dtype=torch.long)
            else:
                # 3p query: (entity_id, (rel1, rel2, rel3))
                entity_id, rel_types = query
                rel_types = list(rel_types)
                query_tensor = torch.tensor([[entity_id] + rel_types], dtype=torch.long)
            
            # Validate tensor shape
            expected_shape = (1, 5) if self.config.query_type == "2ip_pipeline" else (1, 4)
            assert query_tensor.shape == expected_shape, f"Expected tensor shape {expected_shape}, got {query_tensor.shape}"
            
            # Run prediction with configured confidence
            start_time = time.time()
            predicted_batch = self.run_prediction(
                model, query_tensor, self.config.confidence, graph_data
            )
            query_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            # Extract single result from batch (run_prediction returns list of lists)
            predicted_values = self._convert_to_int_list(predicted_batch[0]) if predicted_batch else []
            
            # GT already extracted above for filtering
            fnr, precision, f1 = compute_fnr_metrics([predicted_values], [ground_truth])
            recall = 1 - fnr

            result = QueryResult(precision, recall, f1, predicted_values, ground_truth, False, query_time_ms)
            self.calibration_results.append(result)
            print(f"Calibration Query {i+1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} | Time: {query_time_ms:.2f}ms")
        
        # Print filtering summary
        total_time_seconds = time.time() - start_time
        total_time_ms = total_time_seconds * 1000
        print(f"\nFiltering Summary: Skipped {skipped_count} queries with >{self.config.max_gt_size} GT entities")
        print(f"Processed {len(self.calibration_results)} queries successfully")
        logging.info(f"Total time to process all calibration queries: {total_time_seconds:.2f}s ({total_time_ms:.2f}ms) for {len(self.calibration_results)} queries")
        print(f"Total time to process all calibration queries: {total_time_seconds:.2f}s ({total_time_ms:.2f}ms) for {len(self.calibration_results)} queries")
        
        # Print calibration summary
        if self.calibration_results:
            self.print_calibration_summary()
        else:
            print("No calibration queries processed successfully")
    
    def print_calibration_summary(self):
        """Print summary statistics for calibration queries."""
        print("\n" + "="*80)
        print("CALIBRATION QUERIES SUMMARY")
        print("="*80)
        
        if not self.calibration_results:
            print("No calibration results available")
            return
        
        # Calculate averages
        avg_precision = sum(r.precision for r in self.calibration_results) / len(self.calibration_results)
        avg_recall = sum(r.recall for r in self.calibration_results) / len(self.calibration_results)
        avg_f1 = sum(r.f1 for r in self.calibration_results) / len(self.calibration_results)
        
        print(f"Target recall: {self.config.confidence}")
        print(f"Total calibration queries processed: {len(self.calibration_results)}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
      
        print("="*80)


def main():
    """Main function to run CRC benchmark validation."""
    parser_validate_crc = argparse.ArgumentParser()
    parser_validate_crc.add_argument(
        "--confidence", type=float, default=0.8, help="The confidence level"
    )
    parser_validate_crc.add_argument(
        "--mode", 
        choices=["benchmark", "calibration", "both"], 
        default="both",
        help="Mode to run: 'benchmark' (only benchmark queries), 'calibration' (only calibration queries), or 'both' (default)"
    )
    parser_validate_crc.add_argument(
        "--max-calibration-queries", type=int, default=1000, help="Maximum number of calibration queries to process (only used in calibration/both modes)"
    )
    parser_validate_crc.add_argument(
        "--max-eval-queries", type=int, default=10000, help="Maximum number of test/evaluation queries to process (only used in benchmark/both modes)"
    )
    parser_validate_crc.add_argument(
        "--lambdas-file", type=str, default=None,
        help="Path to JSON file containing pre-calibrated lambda values (if provided, skips calibration and uses these values)"
    )
    parser_validate_crc.add_argument(
        "--dataset", type=str, default=None, 
        choices=["fb15k-237", "nell-955"],
        help="Dataset name: fb15k-237 or nell-955 (required for dataset-specific calibration data)"
    )
    parser_validate_crc.add_argument(
        "--use-ultraquery", action="store_true", default=False,
        help="Use the official UltraQuery checkpoint (ultraquery.pth) instead of dataset-specific models"
    )
    parser_validate_crc.add_argument(
        "--load-path", type=str, default=None,
        help="Override load_path to use a specific checkpoint file or directory (e.g., path to ultraquery.pth)"
    )
    args_validate_crc, remaining_args = parser_validate_crc.parse_known_args()
    
    # Validate dataset is provided
    if args_validate_crc.dataset is None:
        logging.error("--dataset argument is required. Supported datasets: fb15k-237, nell-955")
        sys.exit(1)

    inf_args = merge_args(
        parse_args_inference,
        "args_file",
        ["core", "inference_engine"],
        command_line_args=remaining_args,
    )
    
    # Initialize logging early so all messages are captured
    time_str = parse_time()
    os.makedirs(inf_args.log_path, exist_ok=True)
    set_logger(
        inf_args.log_path,
        f"validate_crc_{inf_args.model_to_infer}_{time_str}.log",
        True,
    )
    
    # Update load_path based on dataset to use dataset-specific model
    # Priority: 1) --load-path override, 2) --use-ultraquery flag, 3) dataset-specific model
    if args_validate_crc.load_path:
        # User explicitly specified load_path, use it
        inf_args.load_path = args_validate_crc.load_path
        logging.info(f"Using user-specified load_path: {inf_args.load_path}")
    elif args_validate_crc.use_ultraquery:
        # Use the official UltraQuery checkpoint
        # Set load_path to the parent directory (not the .pth file itself)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ultraquery_dir = os.path.join(root_dir, "artifacts", "snapshots", "ultraquery")
        
        if not os.path.exists(ultraquery_dir):
            logging.error(f"UltraQuery directory not found at: {ultraquery_dir}")
            logging.error("Please ensure artifacts/snapshots/ultraquery/ exists or use --load-path to specify the checkpoint location")
            sys.exit(1)
        
        inf_args.load_path = ultraquery_dir
        logging.info(f"Using UltraQuery checkpoint directory: {inf_args.load_path}")
        
        # Ensure msp_threshold is set to 0.0 for UltraQuery. Requirement from the original UltraQuery paper.
        if not hasattr(inf_args, 'msp_threshold') or inf_args.msp_threshold != 0.0:
            logging.info("Setting msp_threshold=0.0 for UltraQuery checkpoint")
            inf_args.msp_threshold = 0.0
    elif args_validate_crc.dataset:
        # Map dataset names to model snapshot directories
        dataset_to_model_path = {
            "fb15k-237": "ultra_fb15k237",
            "nell-955": "ultra_nell955"
        }
        
        if args_validate_crc.dataset in dataset_to_model_path:
            # Get the root directory (src -> .. -> repo root)
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            # Build path to dataset-specific model
            model_dir = dataset_to_model_path[args_validate_crc.dataset]
            inf_args.load_path = os.path.join(root_dir, "artifacts", "snapshots", model_dir)
            logging.info(f"Using dataset-specific model: {inf_args.load_path}")
        else:
            logging.warning(f"Unknown dataset '{args_validate_crc.dataset}', using default load_path: {inf_args.load_path}")
    
    save_path = os.path.join(inf_args.log_path, f"crc_{time_str}")
    os.makedirs(save_path, exist_ok=True)

    # Check if model is supported
    model_name = inf_args.model_to_infer.lower()
    if model_name not in BenchmarkConfig.MODEL_TO_QUERY_TYPE:
        logging.error(f"Model '{inf_args.model_to_infer}' is not supported yet.")
        logging.error(f"Supported models: {list(BenchmarkConfig.MODEL_TO_QUERY_TYPE.keys())}")
        logging.error("Implementation needed for other query types.")
        sys.exit(1)

    # Log the selected mode
    logging.info(f"Running in mode: {args_validate_crc.mode}")
    logging.info(f"Model: {inf_args.model_to_infer}")
    logging.info(f"Query type: {BenchmarkConfig.MODEL_TO_QUERY_TYPE[model_name]}")
    if args_validate_crc.mode in ["calibration", "both"]:
        logging.info(f"Maximum calibration queries: {args_validate_crc.max_calibration_queries}")
    if args_validate_crc.lambdas_file:
        logging.info(f"Using pre-calibrated lambda values from: {args_validate_crc.lambdas_file}")
    else:
        logging.info("No lambda file provided, will run calibration")

    # Initialize benchmark validator with model-specific config
    config = BenchmarkConfig(
        confidence=args_validate_crc.confidence,
        max_calibration_queries=args_validate_crc.max_calibration_queries,
        max_queries_per_file=args_validate_crc.max_eval_queries,
        lambdas_file_path=args_validate_crc.lambdas_file,
        query_type=BenchmarkConfig.MODEL_TO_QUERY_TYPE[model_name],
        dataset=args_validate_crc.dataset
    )
    
    logging.info(f"Using dataset: {args_validate_crc.dataset}")
    logging.info(f"Calibration data path: {config.calibration_data_path}")
    logging.info(f"Test queries path: {config.test_queries_path}")
    validator = CRCBenchmarkValidator(config)
    
    # Get dataset statistics and update model args
    dataset_stats = get_dataset_statistics(args_validate_crc.dataset)
    inf_args.num_entities = dataset_stats["num_entities"]
    inf_args.num_relations = dataset_stats["num_relations"]
    inf_args.dataset = args_validate_crc.dataset  # Pass dataset for dataset-aware caching
    logging.info(f"Dataset statistics - Entities: {inf_args.num_entities}, Relations: {inf_args.num_relations}")
    
    # Setup model and calibration
    model_factory_pipeline, db_controller = validator.setup_model_and_calibration(inf_args)
    
    # Run benchmark queries
    logging.info("Running benchmark queries with calibrated 3-hop model...")
    
    # Get graph data for the model
    graph_data = get_graph(db_controller, inf_args.device, augment_inverse_edges=True, relation_graph=True)
    
    # Update calibration generator with actual num_entities from graph_data
    if hasattr(model_factory_pipeline.model, 'calibration_generator'):
        model_factory_pipeline.model.calibration_generator.num_entities = graph_data.num_nodes
        logging.info(f"Updated calibration generator num_entities to {graph_data.num_nodes} from graph_data")
    
    # Process queries based on selected mode
    if args_validate_crc.mode in ["calibration", "both"]:
        logging.info("Processing calibration queries...")
        validator.process_calibration_queries(model_factory_pipeline.model, graph_data)
    else:
        logging.info("Skipping calibration queries (mode: benchmark only)")
    
    if args_validate_crc.mode in ["benchmark", "both"]:
        logging.info("Processing benchmark queries...")
        # Get inference batch size from args
        inference_batch_size = getattr(inf_args, 'inference_batch_size', 1)
        logging.info(f"Using inference batch size: {inference_batch_size}")
        # Process queries for the configured query type
        validator.process_queries(model_factory_pipeline.model, graph_data, inference_batch_size=inference_batch_size)
    else:
        logging.info("Skipping benchmark queries (mode: calibration only)")
    
    # Print summary statistics (only if benchmark queries were run)
    if args_validate_crc.mode in ["benchmark", "both"]:
        validator.print_summary_statistics()
    
    logging.info("Query processing completed!")


if __name__ == "__main__":
    main()