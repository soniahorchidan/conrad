#!/usr/bin/env python3
"""
Run baseline benchmarks for comparison with ThreeHopPipeline results.

Three baselines:
1. DBExecModel (Neo4j symbolic execution) - runs full query in Neo4j
2. Ultra with static threshold (neural) - runs hops with static threshold (e.g., 0.75)
3. Hybrid (neural + symbolic) with static thresholds - uses MultiHopPredictor with static thresholds

This script:
- Loads the same benchmark queries as run_crc_benchmark_auto.py
- Runs selected baselines
- Collects precision, recall, F1, and execution time
- Saves results in a format comparable to results_summary.csv
"""

import os
import sys
import argparse
import logging
import json
import time
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add parent dirs to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'benchmark'))

from inference import ModelFactory
from utils import set_logger, parse_time, get_graph
from graph_handler import Neo4JBackendDBController
from conformal_prediction.utils import compute_fnr_metrics
import argparse as argparse_module

# TODO(Sonia): split this into multiple files under src/baseline_runners/

@dataclass
class QueryResult:
    """Container for query evaluation results."""
    precision: float
    recall: float
    f1: float
    predicted_values: List[int]
    ground_truth: List[int]
    execution_time_ms: float
    is_abstention: bool = False  # True if model abstained (empty prediction)
    neo4j_calls: int = 0  # Total Neo4j (1-hop) invocations for this query
    ultra_calls: int = 0  # Total ULTRA (neural) invocations for this query


# TODO(Sonia): split this into multiple runners
class BaselineRunner:
    """
    Runs baseline benchmarks on the same queries used by ThreeHopPipeline.
    """
    
    def __init__(self, config_path: str, query_dir: str, max_queries: int = 1000, 
                 ultra_batch_size: int = 16, use_ultraquery: bool = False):
        """
        Initialize baseline runner.
        
        Args:
            config_path: Path to orb config JSON file
            query_dir: Directory containing benchmark queries (should end with _pipeline)
            max_queries: Maximum number of queries to process
            ultra_batch_size: Batch size for Ultra intermediate hops (to avoid GPU OOM)
            use_ultraquery: If True, use UltraQuery checkpoint instead of dataset-specific models
        """
        self.config_path = config_path
        self.query_dir = query_dir
        self.max_queries = max_queries
        self.ultra_batch_size = ultra_batch_size
        self.use_ultraquery = use_ultraquery
        
        # Detect template from directory name (more reliable than query string)
        if query_dir.endswith("3p_pipeline"):
            self.template_from_dir = "3p"
        elif query_dir.endswith("2u_pipeline"):
            self.template_from_dir = "2u"
        elif query_dir.endswith("2ip_pipeline"):
            self.template_from_dir = "2ip"
        else:
            # Fallback to query string detection
            self.template_from_dir = None
        
        # Results storage
        self.results = {
            'neo4j_symbolic': [],
            'ultra_neural': [],
            'hybrid_static': []
        }
        
        # Store threshold for reporting
        self.threshold = None
        self.hybrid_threshold = None  # Store actual hybrid threshold value
        
        # Cache for Ultra scores (to avoid recomputing inference for multiple thresholds)
        self.ultra_scores_cache = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_query_template(self, query: str) -> str:
        """Detect query template type from query string."""
        if "'2u'" in query or '"2u"' in query:
            return "2u"
        elif "'2ip'" in query or '"2ip"' in query:
            return "2ip"
        else:
            return "3p"
    
    def parse_query(self, query: str) -> Tuple:
        """Parse query string based on template type.
        
        Returns:
            - For 3p: (entity_id, [rel1, rel2, rel3])
            - For 2u: (anchor1, rel1, anchor2, rel2)
            - For 2ip: (anchor1, rel1, anchor2, rel2, rel3)
        """
        template = self.detect_query_template(query)
        
        if template == "2u":
            return self.parse_query_2u(query)
        elif template == "2ip":
            return self.parse_query_2ip(query)
        else:
            return self.parse_query_3p(query)
    
    def parse_query_3p(self, query: str) -> Tuple[int, List[int]]:
        """Parse 3p query: query((entity_id, (rel1, rel2, rel3)))"""
        # Try Python tuple format first (from calibration split)
        tuple_match = re.search(r'query\(\((\d+),\s*\(([0-9,\s]+)\)\)\)', query)
        if tuple_match:
            entity_id = int(tuple_match.group(1))
            rel_str = tuple_match.group(2)
            rel_types = [int(r.strip()) for r in rel_str.split(',')]
            if len(rel_types) != 3:
                raise ValueError(f"Expected 3 relations, found {len(rel_types)} in query: {query}")
            return entity_id, rel_types
        
        # Try Cypher format (from benchmark queries)
        entity_match = re.search(r'Entity \{id: (\d+)\}', query)
        if not entity_match:
            raise ValueError(f"Could not extract entity ID from query: {query}")
        entity_id = int(entity_match.group(1))
        
        relation_matches = re.findall(r'Relation \{type: (\d+)\}', query)
        if len(relation_matches) != 3:
            raise ValueError(f"Expected 3 relations, found {len(relation_matches)} in query: {query}")
        
        rel_types = [int(r) for r in relation_matches]
        
        return entity_id, rel_types
    
    def parse_query_2u(self, query: str) -> Tuple[int, int, int, int]:
        """Parse 2u query: query(((anchor1, rel1), (anchor2, rel2), '2u'))"""
        # Match: query(((anchor1, rel1), (anchor2, rel2), '2u'))
        match = re.search(r'query\(\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\),\s*[\'"]2u[\'"]\)\)', query)
        if match:
            return (int(match.group(1)), int(match.group(2)), 
                   int(match.group(3)), int(match.group(4)))
        raise ValueError(f"Could not parse 2u query: {query}")
    
    def parse_query_2ip(self, query: str) -> Tuple[int, int, int, int, int]:
        """Parse 2ip query: query(((anchor1, rel1), (anchor2, rel2), rel3, '2ip'))"""
        # Match: query(((anchor1, rel1), (anchor2, rel2), rel3, '2ip'))
        match = re.search(r'query\(\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\),\s*(\d+),\s*[\'"]2ip[\'"]\)\)', query)
        if match:
            return (int(match.group(1)), int(match.group(2)), 
                   int(match.group(3)), int(match.group(4)), int(match.group(5)))
        raise ValueError(f"Could not parse 2ip query: {query}")
    
    def load_queries_from_dir(self) -> List[str]:
        """Load queries directly from query_dir/queries (for 3p_pipeline structure)."""
        query_path = os.path.join(self.query_dir, "queries")
        with open(query_path, "r") as f:
            queries = [q.strip() for q in f.readlines() if q.strip()]
        return queries[:self.max_queries]
    
    def load_ground_truth_from_dir(self, gt_filename: str = "gt") -> List[List[int]]:
        """Load ground truth directly from query_dir/gt (for 3p_pipeline structure)."""
        gt_path = os.path.join(self.query_dir, gt_filename)
        gt_values = []
        
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    gt_values.append([int(x) for x in line.split()])
                else:
                    gt_values.append([])
        
        return gt_values[:self.max_queries]
    
    def _normalize_ground_truth(self, gt: List) -> List[int]:
        """Normalize ground truth to list of ints."""
        return [int(x) for x in gt] if gt else []
    
    def _apply_threshold(self, scores: torch.Tensor, threshold: float) -> List[int]:
        """Apply threshold to scores and return node indices."""
        return torch.nonzero(scores > threshold, as_tuple=True)[0].tolist()
    
    def _create_abstention_result(self, gt: List, execution_time_ms: float,
                                   neo4j_calls: int = 0, ultra_calls: int = 0) -> QueryResult:
        """Create a QueryResult for an abstention (empty prediction)."""
        gt_normalized = self._normalize_ground_truth(gt)
        return QueryResult(
            precision=0.0 if len(gt_normalized) > 0 else 1.0,
            recall=0.0,
            f1=0.0,
            predicted_values=[],
            ground_truth=gt_normalized,
            execution_time_ms=execution_time_ms,
            is_abstention=True,
            neo4j_calls=neo4j_calls,
            ultra_calls=ultra_calls
        )
    
    def _create_error_result(self, gt: List) -> QueryResult:
        """Create a QueryResult for a failed query."""
        return self._create_abstention_result(gt, 0.0)
    
    def _compute_metrics_from_predictions(self, predictions: List[int], gt: List) -> Tuple[float, float, float, float]:
        """Compute metrics from predictions and ground truth. Returns (precision, recall, f1, fnr)."""
        gt_normalized = self._normalize_ground_truth(gt)
        fnr, precision, f1 = compute_fnr_metrics([predictions], [gt_normalized])
        recall = 1.0 - fnr
        return precision, recall, f1, fnr
    
    def _process_hop_batch(self, ultra_model, graph_data, nodes: List[int], 
                          rel_type: int, threshold: float, batch_size: int) -> Tuple[List[int], int]:
        """Process a batch of nodes through a hop. Returns (aggregated candidate nodes, num_ultra_calls)."""
        candidates_set = set()
        for batch_start in range(0, len(nodes), batch_size):
            batch_end = min(batch_start + batch_size, len(nodes))
            batch_nodes = nodes[batch_start:batch_end]
            
            queries = torch.tensor([[node, rel_type] for node in batch_nodes], 
                                  dtype=torch.long).to(graph_data.edge_index.device)
            with torch.no_grad():
                scores = ultra_model.forward(graph_data, queries)
            
            for j in range(len(batch_nodes)):
                candidates = self._apply_threshold(scores[j], threshold)
                candidates_set.update(candidates)
        # Total ULTRA 1-hop inferences = len(nodes)
        return list(candidates_set), len(nodes)
    
    def _process_hop_batch_get_scores(self, ultra_model, graph_data, nodes: List[int], 
                                      rel_type: int, batch_size: int) -> Dict[int, torch.Tensor]:
        """Process a batch of nodes through a hop and return all scores (no threshold).
        Returns dict mapping node -> scores tensor (on CPU to save GPU memory)."""
        node_scores = {}
        batch_count = 0
        for batch_start in range(0, len(nodes), batch_size):
            batch_end = min(batch_start + batch_size, len(nodes))
            batch_nodes = nodes[batch_start:batch_end]
            
            queries = torch.tensor([[node, rel_type] for node in batch_nodes], 
                                  dtype=torch.long).to(graph_data.edge_index.device)
            with torch.no_grad():
                scores = ultra_model.forward(graph_data, queries)
            
            for j, node in enumerate(batch_nodes):
                # CRITICAL FIX: Move to CPU immediately to free GPU memory
                node_scores[node] = scores[j].cpu().clone()
            
            # Explicitly delete GPU tensors
            del scores, queries
            # Only clear cache periodically to avoid overhead (every 10 batches)
            batch_count += 1
            if batch_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final cache clear at end
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return node_scores
    
    def run_neo4j_baseline(self, db_controller, queries: List[str], 
                          ground_truths: List[List[int]]) -> List[QueryResult]:
        """
        Run Neo4j symbolic baseline. Routes to appropriate method based on query template.
        """
        # Use template from directory if available, otherwise detect from query string
        if hasattr(self, 'template_from_dir') and self.template_from_dir:
            template = self.template_from_dir
        else:
            template = self.detect_query_template(queries[0]) if queries else "3p"
        
        if template == "2u":
            return self.run_neo4j_baseline_2u(db_controller, queries, ground_truths)
        elif template == "2ip":
            return self.run_neo4j_baseline_2ip(db_controller, queries, ground_truths)
        else:
            return self.run_neo4j_baseline_3p(db_controller, queries, ground_truths)
    
    def run_neo4j_baseline_3p(self, db_controller, queries: List[str], 
                              ground_truths: List[List[int]]) -> List[QueryResult]:
        """
        Run Neo4j symbolic baseline using DBExecModel for 3p queries.
        
        This executes the full 3-hop query directly in Neo4j.
        """
        from models.db_exec.model import DBExecModel
        
        self.logger.info("Running Neo4j Symbolic Baseline (3p)...")
        args = argparse.Namespace(db_controller=db_controller)
        dbexec_model = DBExecModel(args, num_relations=237, device='cpu')
        
        results = []
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Neo4j Symbolic")):
            try:
                entity_id, rel_types = self.parse_query_3p(query_str)
                query_tensor = torch.tensor([[entity_id] + rel_types], dtype=torch.long)
                
                start_time = time.time()
                predictions, _ = dbexec_model.predict(query_tensor)
                execution_time_ms = (time.time() - start_time) * 1000
                
                pred_values = predictions[0] if predictions else []
                precision, recall, f1, _ = self._compute_metrics_from_predictions(pred_values, gt)
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=pred_values,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(pred_values) == 0,
                    neo4j_calls=1,
                    ultra_calls=0
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def run_neo4j_baseline_2u(self, db_controller, queries: List[str], 
                             ground_truths: List[List[int]]) -> List[QueryResult]:
        """
        Run Neo4j symbolic baseline for 2u (union) queries.
        
        Executes two 1-hop queries and unions the results.
        """
        self.logger.info("Running Neo4j Symbolic Baseline (2u)...")
        
        ONE_HOP_TEMPLATE = "MATCH (a:Entity {id: %s})-[f:Relation {type: %s}]->(r:Entity) RETURN DISTINCT r.id"
        
        results = []
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Neo4j Symbolic (2u)")):
            try:
                anchor1, rel1, anchor2, rel2 = self.parse_query_2u(query_str)
                
                start_time = time.time()
                
                # Execute branch 1
                cypher1 = ONE_HOP_TEMPLATE % (anchor1, rel1)
                res1 = db_controller.execute_query(cypher1)
                set1 = {record["r.id"] for record in res1}
                
                # Execute branch 2
                cypher2 = ONE_HOP_TEMPLATE % (anchor2, rel2)
                res2 = db_controller.execute_query(cypher2)
                set2 = {record["r.id"] for record in res2}
                
                # Union
                pred_values = list(set1 | set2)
                execution_time_ms = (time.time() - start_time) * 1000
                
                precision, recall, f1, _ = self._compute_metrics_from_predictions(pred_values, gt)
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=pred_values,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(pred_values) == 0,
                    neo4j_calls=2,
                    ultra_calls=0
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def run_neo4j_baseline_2ip(self, db_controller, queries: List[str], 
                              ground_truths: List[List[int]]) -> List[QueryResult]:
        """
        Run Neo4j symbolic baseline for 2ip (intersect-project) queries.
        
        Executes two 1-hop queries, intersects, then projects from intersection.
        """
        self.logger.info("Running Neo4j Symbolic Baseline (2ip)...")
        
        ONE_HOP_TEMPLATE = "MATCH (a:Entity {id: %s})-[f:Relation {type: %s}]->(r:Entity) RETURN DISTINCT r.id"
        PROJECT_TEMPLATE = "MATCH (i:Entity)-[f:Relation {type: %s}]->(r:Entity) WHERE i.id IN %s RETURN DISTINCT r.id"
        
        results = []
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Neo4j Symbolic (2ip)")):
            try:
                anchor1, rel1, anchor2, rel2, rel3 = self.parse_query_2ip(query_str)
                
                start_time = time.time()
                
                # Execute branch 1
                cypher1 = ONE_HOP_TEMPLATE % (anchor1, rel1)
                res1 = db_controller.execute_query(cypher1)
                set1 = {record["r.id"] for record in res1}
                
                # Execute branch 2
                cypher2 = ONE_HOP_TEMPLATE % (anchor2, rel2)
                res2 = db_controller.execute_query(cypher2)
                set2 = {record["r.id"] for record in res2}
                
                # Intersect
                intersection = set1 & set2
                
                if not intersection:
                    pred_values = []
                else:
                    # Project from intersection - need to build query for each node or use UNWIND
                    intersection_list = list(intersection)
                    if len(intersection_list) == 1:
                        cypher3 = f"MATCH (i:Entity {{id: {intersection_list[0]}}})-[f:Relation {{type: {rel3}}}]->(r:Entity) RETURN DISTINCT r.id"
                        res3 = db_controller.execute_query(cypher3)
                    else:
                        # Use UNWIND for multiple nodes
                        node_ids_str = ', '.join(map(str, intersection_list))
                        cypher3 = f"UNWIND [{node_ids_str}] AS node_id MATCH (i:Entity {{id: node_id}})-[f:Relation {{type: {rel3}}}]->(r:Entity) RETURN DISTINCT r.id"
                        res3 = db_controller.execute_query(cypher3)
                    pred_values = [record["r.id"] for record in res3]
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                precision, recall, f1, _ = self._compute_metrics_from_predictions(pred_values, gt)
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=pred_values,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(pred_values) == 0,
                    neo4j_calls=3,
                    ultra_calls=0
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def compute_ultra_scores(self, ultra_model, queries: List[str], 
                             ground_truths: List[List[int]], 
                             graph_data: Any,
                             min_threshold: float = 0.0,
                             batch_size: int = 32) -> List[Dict]:
        """
        Compute Ultra scores for all queries without applying thresholds.
        Routes to appropriate method based on query template.
        """
        # Use template from directory if available, otherwise detect from query string
        if hasattr(self, 'template_from_dir') and self.template_from_dir:
            template = self.template_from_dir
        else:
            template = self.detect_query_template(queries[0]) if queries else "3p"
        
        if template == "2u":
            return self.compute_ultra_scores_2u(ultra_model, queries, ground_truths, graph_data, min_threshold, batch_size)
        elif template == "2ip":
            return self.compute_ultra_scores_2ip(ultra_model, queries, ground_truths, graph_data, min_threshold, batch_size)
        else:
            return self.compute_ultra_scores_3p(ultra_model, queries, ground_truths, graph_data, min_threshold, batch_size)
    
    def compute_ultra_scores_3p(self, ultra_model, queries: List[str], 
                                ground_truths: List[List[int]], 
                                graph_data: Any,
                                min_threshold: float = 0.0,
                                batch_size: int = 32) -> List[Dict]:
        """
        Compute Ultra scores for 3p queries without applying thresholds.
        """
        self.logger.info(f"Computing Ultra scores (3p, inference only, min_threshold={min_threshold})...")
        
        ultra_model.eval()
        device = graph_data.edge_index.device
        scores_data = []
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Computing Ultra scores")):
            try:
                entity_id, rel_types = self.parse_query_3p(query_str)
                start_time = time.time()  # Track inference time
                
                # Hop 1: compute scores for all nodes
                hop1_query = torch.tensor([[entity_id, rel_types[0]]], dtype=torch.long).to(device)
                with torch.no_grad():
                    hop1_scores_gpu = ultra_model.forward(graph_data, hop1_query)[0]
                
                # Hop 2: compute scores only for nodes that pass min_threshold
                # Apply threshold while still on GPU, then move to CPU
                hop1_candidates = self._apply_threshold(hop1_scores_gpu, min_threshold)
                
                # CRITICAL FIX: Move hop1_scores to CPU immediately after threshold application
                hop1_scores = hop1_scores_gpu.cpu().clone()
                del hop1_scores_gpu, hop1_query
                
                hop2_scores = self._process_hop_batch_get_scores(ultra_model, graph_data, 
                                                                 hop1_candidates, rel_types[1], batch_size)
                del hop1_candidates
                
                # Hop 3: compute scores only for nodes from hop2 that pass min_threshold
                hop2_candidates = []
                for node, scores in hop2_scores.items():
                    # scores are already on CPU (from _process_hop_batch_get_scores), move to GPU for threshold check
                    candidates = self._apply_threshold(scores.to(device), min_threshold)
                    hop2_candidates.extend(candidates)
                hop2_candidates = list(set(hop2_candidates))  # Remove duplicates
                hop3_scores = self._process_hop_batch_get_scores(ultra_model, graph_data,
                                                                 hop2_candidates, rel_types[2], batch_size)
                del hop2_candidates
                
                execution_time_ms = (time.time() - start_time) * 1000  # Store inference time
                
                scores_data.append({
                    'hop1_scores': hop1_scores,  # Already on CPU
                    'hop2_scores': hop2_scores,  # Already on CPU (from _process_hop_batch_get_scores)
                    'hop3_scores': hop3_scores,  # Already on CPU (from _process_hop_batch_get_scores)
                    'rel_types': rel_types,
                    'entity_id': entity_id,
                    'ground_truth': gt,
                    'execution_time_ms': execution_time_ms,
                    'template': '3p'
                })
            except Exception as e:
                self.logger.error(f"Error computing scores for query {i}: {e}")
                # Explicitly delete any remaining GPU tensors
                if 'hop1_scores_gpu' in locals():
                    del hop1_scores_gpu
                if 'hop1_query' in locals():
                    del hop1_query
                scores_data.append({
                    'hop1_scores': None,
                    'hop2_scores': {},
                    'hop3_scores': {},
                    'rel_types': [],
                    'entity_id': 0,
                    'ground_truth': gt,
                    'execution_time_ms': 0.0,
                    'template': '3p'
                })
            finally:
                # Explicitly delete tensors
                if 'hop1_scores_gpu' in locals():
                    del hop1_scores_gpu
                if 'hop1_query' in locals():
                    del hop1_query
                # Clear GPU cache periodically (every 10 queries) to prevent memory accumulation
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final cache clear at end
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return scores_data
    
    def compute_ultra_scores_2u(self, ultra_model, queries: List[str], 
                               ground_truths: List[List[int]], 
                               graph_data: Any,
                               min_threshold: float = 0.0,
                               batch_size: int = 32) -> List[Dict]:
        """
        Compute Ultra scores for 2u queries without applying thresholds.
        """
        self.logger.info(f"Computing Ultra scores (2u, inference only, min_threshold={min_threshold})...")
        
        ultra_model.eval()
        device = graph_data.edge_index.device
        scores_data = []
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Computing Ultra scores (2u)")):
            try:
                anchor1, rel1, anchor2, rel2 = self.parse_query_2u(query_str)
                start_time = time.time()  # Track inference time
                
                # Branch 1: compute scores for all nodes
                branch1_query = torch.tensor([[anchor1, rel1]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch1_scores_gpu = ultra_model.forward(graph_data, branch1_query)[0]
                
                # Branch 2: compute scores for all nodes
                branch2_query = torch.tensor([[anchor2, rel2]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch2_scores_gpu = ultra_model.forward(graph_data, branch2_query)[0]
                
                # CRITICAL FIX: Move scores to CPU immediately to free GPU memory
                branch1_scores = branch1_scores_gpu.cpu().clone()
                branch2_scores = branch2_scores_gpu.cpu().clone()
                del branch1_scores_gpu, branch2_scores_gpu, branch1_query, branch2_query
                
                execution_time_ms = (time.time() - start_time) * 1000  # Store inference time
                
                scores_data.append({
                    'branch1_scores': branch1_scores,  # On CPU
                    'branch2_scores': branch2_scores,  # On CPU
                    'anchor1': anchor1,
                    'rel1': rel1,
                    'anchor2': anchor2,
                    'rel2': rel2,
                    'ground_truth': gt,
                    'execution_time_ms': execution_time_ms,
                    'template': '2u'
                })
            except Exception as e:
                self.logger.error(f"Error computing scores for query {i}: {e}")
                # Explicitly delete any remaining GPU tensors
                if 'branch1_scores_gpu' in locals():
                    del branch1_scores_gpu
                if 'branch2_scores_gpu' in locals():
                    del branch2_scores_gpu
                if 'branch1_query' in locals():
                    del branch1_query
                if 'branch2_query' in locals():
                    del branch2_query
                scores_data.append({
                    'branch1_scores': None,
                    'branch2_scores': None,
                    'anchor1': 0,
                    'rel1': 0,
                    'anchor2': 0,
                    'rel2': 0,
                    'ground_truth': gt,
                    'execution_time_ms': 0.0,
                    'template': '2u'
                })
            finally:
                # Clear GPU cache periodically (every 10 queries) to prevent memory accumulation
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final cache clear at end
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return scores_data
    
    def compute_ultra_scores_2ip(self, ultra_model, queries: List[str], 
                                 ground_truths: List[List[int]], 
                                 graph_data: Any,
                                 min_threshold: float = 0.0,
                                 batch_size: int = 32) -> List[Dict]:
        """
        Compute Ultra scores for 2ip queries without applying thresholds.
        """
        self.logger.info(f"Computing Ultra scores (2ip, inference only, min_threshold={min_threshold})...")
        
        ultra_model.eval()
        device = graph_data.edge_index.device
        scores_data = []
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Computing Ultra scores (2ip)")):
            try:
                anchor1, rel1, anchor2, rel2, rel3 = self.parse_query_2ip(query_str)
                start_time = time.time()  # Track inference time
                
                # Branch 1: compute scores for all nodes
                branch1_query = torch.tensor([[anchor1, rel1]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch1_scores_gpu = ultra_model.forward(graph_data, branch1_query)[0]
                
                # Branch 2: compute scores for all nodes
                branch2_query = torch.tensor([[anchor2, rel2]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch2_scores_gpu = ultra_model.forward(graph_data, branch2_query)[0]
                
                # Compute intersection candidates (nodes that pass min_threshold in both branches)
                # Apply threshold while still on GPU
                branch1_candidates = self._apply_threshold(branch1_scores_gpu, min_threshold)
                branch2_candidates = self._apply_threshold(branch2_scores_gpu, min_threshold)
                intersection_candidates = list(set(branch1_candidates) & set(branch2_candidates))
                
                # CRITICAL FIX: Move scores to CPU immediately after threshold application
                branch1_scores = branch1_scores_gpu.cpu().clone()
                branch2_scores = branch2_scores_gpu.cpu().clone()
                del branch1_scores_gpu, branch2_scores_gpu, branch1_query, branch2_query
                del branch1_candidates, branch2_candidates
                
                # Project scores from intersection candidates
                proj_scores = {}
                if intersection_candidates:
                    proj_scores = self._process_hop_batch_get_scores(ultra_model, graph_data,
                                                                     intersection_candidates, rel3, batch_size)
                del intersection_candidates
                
                execution_time_ms = (time.time() - start_time) * 1000  # Store inference time
                
                scores_data.append({
                    'branch1_scores': branch1_scores,  # On CPU
                    'branch2_scores': branch2_scores,  # On CPU
                    'proj_scores': proj_scores,  # Dict[node -> scores] for projection (already on CPU)
                    'anchor1': anchor1,
                    'rel1': rel1,
                    'anchor2': anchor2,
                    'rel2': rel2,
                    'rel3': rel3,
                    'ground_truth': gt,
                    'execution_time_ms': execution_time_ms,
                    'template': '2ip'
                })
            except Exception as e:
                self.logger.error(f"Error computing scores for query {i}: {e}")
                # Explicitly delete any remaining GPU tensors
                if 'branch1_scores_gpu' in locals():
                    del branch1_scores_gpu
                if 'branch2_scores_gpu' in locals():
                    del branch2_scores_gpu
                if 'branch1_query' in locals():
                    del branch1_query
                if 'branch2_query' in locals():
                    del branch2_query
                scores_data.append({
                    'branch1_scores': None,
                    'branch2_scores': None,
                    'proj_scores': {},
                    'anchor1': 0,
                    'rel1': 0,
                    'anchor2': 0,
                    'rel2': 0,
                    'rel3': 0,
                    'ground_truth': gt,
                    'execution_time_ms': 0.0,
                    'template': '2ip'
                })
            finally:
                # Explicitly delete tensors
                if 'branch1_scores_gpu' in locals():
                    del branch1_scores_gpu
                if 'branch2_scores_gpu' in locals():
                    del branch2_scores_gpu
                if 'branch1_query' in locals():
                    del branch1_query
                if 'branch2_query' in locals():
                    del branch2_query
                # Clear GPU cache periodically (every 10 queries) to prevent memory accumulation
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final cache clear at end
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return scores_data
    
    def apply_threshold_to_scores(self, scores_data: List[Dict], threshold: float,
                                  batch_size: int = 32) -> List[QueryResult]:
        """
        Apply threshold to pre-computed scores and return results.
        Routes to appropriate method based on template.
        """
        if not scores_data:
            return []
        
        template = scores_data[0].get('template', '3p')
        
        if template == "2u":
            return self.apply_threshold_to_scores_2u(scores_data, threshold)
        elif template == "2ip":
            return self.apply_threshold_to_scores_2ip(scores_data, threshold)
        else:
            return self.apply_threshold_to_scores_3p(scores_data, threshold)
    
    def apply_threshold_to_scores_3p(self, scores_data: List[Dict], threshold: float,
                                     batch_size: int = 32) -> List[QueryResult]:
        """
        Apply threshold to pre-computed 3p scores and return results.
        """
        results = []
        
        for i, data in enumerate(tqdm(scores_data, desc=f"Applying threshold {threshold}")):
            try:
                if data.get('hop1_scores') is None:
                    results.append(self._create_error_result(data['ground_truth']))
                    continue
                
                # Use stored inference time, not threshold application time
                execution_time_ms = data.get('execution_time_ms', 0.0)
                
                # Hop 1: apply threshold (scores are on CPU now)
                hop1_nodes = self._apply_threshold(data['hop1_scores'], threshold)
                if not hop1_nodes:
                    results.append(self._create_abstention_result(data['ground_truth'], execution_time_ms))
                    continue
                
                # Hop 2: apply threshold to cached scores (already on CPU)
                hop2_nodes_set = set()
                for hop1_node in hop1_nodes:
                    if hop1_node in data.get('hop2_scores', {}):
                        candidates = self._apply_threshold(data['hop2_scores'][hop1_node], threshold)
                        hop2_nodes_set.update(candidates)
                hop2_nodes = list(hop2_nodes_set)
                
                if not hop2_nodes:
                    results.append(self._create_abstention_result(data['ground_truth'], execution_time_ms))
                    continue
                
                # Hop 3: apply threshold to cached scores (already on CPU)
                hop3_nodes_set = set()
                for hop2_node in hop2_nodes:
                    if hop2_node in data.get('hop3_scores', {}):
                        candidates = self._apply_threshold(data['hop3_scores'][hop2_node], threshold)
                        hop3_nodes_set.update(candidates)
                hop3_nodes = list(hop3_nodes_set)
                
                precision, recall, f1, _ = self._compute_metrics_from_predictions(hop3_nodes, data['ground_truth'])
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=hop3_nodes,
                    ground_truth=self._normalize_ground_truth(data['ground_truth']),
                    execution_time_ms=execution_time_ms,  # Use stored inference time
                    is_abstention=len(hop3_nodes) == 0
                ))
            except Exception as e:
                self.logger.error(f"Error applying threshold to query {i}: {e}")
                results.append(self._create_error_result(data['ground_truth']))
        
        return results
    
    def apply_threshold_to_scores_2u(self, scores_data: List[Dict], threshold: float) -> List[QueryResult]:
        """
        Apply threshold to pre-computed 2u scores and return results.
        """
        results = []
        
        for i, data in enumerate(tqdm(scores_data, desc=f"Applying threshold {threshold} (2u)")):
            try:
                if data.get('branch1_scores') is None or data.get('branch2_scores') is None:
                    results.append(self._create_error_result(data['ground_truth']))
                    continue
                
                # Use stored inference time
                execution_time_ms = data.get('execution_time_ms', 0.0)
                
                # Apply threshold to branch 1
                branch1_nodes = self._apply_threshold(data['branch1_scores'], threshold)
                
                # Apply threshold to branch 2
                branch2_nodes = self._apply_threshold(data['branch2_scores'], threshold)
                
                # Union
                union_nodes = list(set(branch1_nodes) | set(branch2_nodes))
                
                precision, recall, f1, _ = self._compute_metrics_from_predictions(union_nodes, data['ground_truth'])
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=union_nodes,
                    ground_truth=self._normalize_ground_truth(data['ground_truth']),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(union_nodes) == 0
                ))
            except Exception as e:
                self.logger.error(f"Error applying threshold to query {i}: {e}")
                results.append(self._create_error_result(data['ground_truth']))
        
        return results
    
    def apply_threshold_to_scores_2ip(self, scores_data: List[Dict], threshold: float) -> List[QueryResult]:
        """
        Apply threshold to pre-computed 2ip scores and return results.
        Uses the same threshold for both intersect and project.
        """
        results = []
        
        for i, data in enumerate(tqdm(scores_data, desc=f"Applying threshold {threshold} (2ip)")):
            try:
                if data.get('branch1_scores') is None or data.get('branch2_scores') is None:
                    results.append(self._create_error_result(data['ground_truth']))
                    continue
                
                # Use stored inference time
                execution_time_ms = data.get('execution_time_ms', 0.0)
                
                # Apply threshold to branch 1
                branch1_nodes = self._apply_threshold(data['branch1_scores'], threshold)
                
                # Apply threshold to branch 2
                branch2_nodes = self._apply_threshold(data['branch2_scores'], threshold)
                
                # Intersect
                intersection_nodes = list(set(branch1_nodes) & set(branch2_nodes))
                
                if not intersection_nodes:
                    results.append(self._create_abstention_result(data['ground_truth'], execution_time_ms))
                    continue
                
                # Apply threshold to projection scores from intersection nodes
                proj_nodes_set = set()
                for inter_node in intersection_nodes:
                    if inter_node in data.get('proj_scores', {}):
                        candidates = self._apply_threshold(data['proj_scores'][inter_node], threshold)
                        proj_nodes_set.update(candidates)
                proj_nodes = list(proj_nodes_set)
                
                precision, recall, f1, _ = self._compute_metrics_from_predictions(proj_nodes, data['ground_truth'])
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=proj_nodes,
                    ground_truth=self._normalize_ground_truth(data['ground_truth']),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(proj_nodes) == 0
                ))
            except Exception as e:
                self.logger.error(f"Error applying threshold to query {i}: {e}")
                results.append(self._create_error_result(data['ground_truth']))
        
        return results
    
    def run_ultra_pipeline_baseline(self, ultra_model, queries: List[str], 
                                    ground_truths: List[List[int]], 
                                    graph_data: Any,
                                    static_threshold: float = 0.75,
                                    batch_size: int = 32,
                                    scores_cache: Optional[List[Dict]] = None) -> List[QueryResult]:
        """
        Run Ultra neural baseline with static threshold.
        Routes to appropriate method based on query template.
        
        If scores_cache is provided, applies threshold to cached scores.
        Otherwise, computes inference and applies threshold.
        
        Args:
            scores_cache: Pre-computed scores (from compute_ultra_scores)
        """
        if scores_cache is not None:
            self.logger.info(f"Applying threshold {static_threshold} to cached Ultra scores...")
            return self.apply_threshold_to_scores(scores_cache, static_threshold, batch_size)
        
        # Route to template-specific implementation
        # Use template from directory if available, otherwise detect from query string
        if hasattr(self, 'template_from_dir') and self.template_from_dir:
            template = self.template_from_dir
        else:
            template = self.detect_query_template(queries[0]) if queries else "3p"
        
        if template == "2u":
            return self.run_ultra_pipeline_baseline_2u(ultra_model, queries, ground_truths, graph_data, static_threshold, batch_size)
        elif template == "2ip":
            return self.run_ultra_pipeline_baseline_2ip(ultra_model, queries, ground_truths, graph_data, static_threshold, batch_size)
        else:
            return self.run_ultra_pipeline_baseline_3p(ultra_model, queries, ground_truths, graph_data, static_threshold, batch_size)
    
    def run_ultra_pipeline_baseline_3p(self, ultra_model, queries: List[str], 
                                      ground_truths: List[List[int]], 
                                      graph_data: Any,
                                      static_threshold: float = 0.75,
                                      batch_size: int = 32) -> List[QueryResult]:
        """
        Run Ultra neural baseline for 3p queries with static threshold.
        """
        self.logger.info(f"Running Ultra Neural Baseline (3p, threshold={static_threshold})...")
        
        results = []
        ultra_model.eval()
        device = graph_data.edge_index.device
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Ultra Neural (3p)")):
            try:
                entity_id, rel_types = self.parse_query_3p(query_str)
                start_time = time.time()
                
                # Hop 1
                hop1_query = torch.tensor([[entity_id, rel_types[0]]], dtype=torch.long).to(device)
                with torch.no_grad():
                    hop1_scores = ultra_model.forward(graph_data, hop1_query)
                hop1_nodes = self._apply_threshold(hop1_scores[0], static_threshold)
                
                if not hop1_nodes:
                    results.append(self._create_abstention_result(gt, (time.time() - start_time) * 1000, ultra_calls=1))
                    continue
                
                # Hop 2
                hop2_nodes, hop2_calls = self._process_hop_batch(ultra_model, graph_data, hop1_nodes, 
                                                     rel_types[1], static_threshold, batch_size)
                if not hop2_nodes:
                    results.append(self._create_abstention_result(gt, (time.time() - start_time) * 1000, ultra_calls=1 + hop2_calls))
                    continue
                
                # Hop 3
                hop3_nodes, hop3_calls = self._process_hop_batch(ultra_model, graph_data, hop2_nodes, 
                                                     rel_types[2], static_threshold, batch_size)
                
                execution_time_ms = (time.time() - start_time) * 1000
                precision, recall, f1, _ = self._compute_metrics_from_predictions(hop3_nodes, gt)
                ultra_calls = 1 + hop2_calls + hop3_calls
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=hop3_nodes,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(hop3_nodes) == 0,
                    neo4j_calls=0,
                    ultra_calls=ultra_calls
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def run_ultra_pipeline_baseline_2u(self, ultra_model, queries: List[str], 
                                       ground_truths: List[List[int]], 
                                       graph_data: Any,
                                       static_threshold: float = 0.75,
                                       batch_size: int = 32) -> List[QueryResult]:
        """
        Run Ultra neural baseline for 2u queries with static threshold.
        """
        self.logger.info(f"Running Ultra Neural Baseline (2u, threshold={static_threshold})...")
        
        results = []
        ultra_model.eval()
        device = graph_data.edge_index.device
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Ultra Neural (2u)")):
            try:
                anchor1, rel1, anchor2, rel2 = self.parse_query_2u(query_str)
                start_time = time.time()
                
                # Branch 1
                branch1_query = torch.tensor([[anchor1, rel1]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch1_scores = ultra_model.forward(graph_data, branch1_query)
                branch1_nodes = self._apply_threshold(branch1_scores[0], static_threshold)
                
                # Branch 2
                branch2_query = torch.tensor([[anchor2, rel2]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch2_scores = ultra_model.forward(graph_data, branch2_query)
                branch2_nodes = self._apply_threshold(branch2_scores[0], static_threshold)
                
                # Union
                union_nodes = list(set(branch1_nodes) | set(branch2_nodes))
                
                execution_time_ms = (time.time() - start_time) * 1000
                precision, recall, f1, _ = self._compute_metrics_from_predictions(union_nodes, gt)
                # 2 branches = 2 ULTRA 1-hop inferences
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=union_nodes,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(union_nodes) == 0,
                    neo4j_calls=0,
                    ultra_calls=2
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def run_ultra_pipeline_baseline_2ip(self, ultra_model, queries: List[str], 
                                        ground_truths: List[List[int]], 
                                        graph_data: Any,
                                        static_threshold: float = 0.75,
                                        batch_size: int = 32) -> List[QueryResult]:
        """
        Run Ultra neural baseline for 2ip queries with static threshold.
        Uses the same threshold for both intersect and project.
        """
        self.logger.info(f"Running Ultra Neural Baseline (2ip, threshold={static_threshold})...")
        
        results = []
        ultra_model.eval()
        device = graph_data.edge_index.device
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Ultra Neural (2ip)")):
            try:
                anchor1, rel1, anchor2, rel2, rel3 = self.parse_query_2ip(query_str)
                start_time = time.time()
                
                # Branch 1
                branch1_query = torch.tensor([[anchor1, rel1]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch1_scores = ultra_model.forward(graph_data, branch1_query)
                branch1_nodes = self._apply_threshold(branch1_scores[0], static_threshold)
                
                # Branch 2
                branch2_query = torch.tensor([[anchor2, rel2]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch2_scores = ultra_model.forward(graph_data, branch2_query)
                branch2_nodes = self._apply_threshold(branch2_scores[0], static_threshold)
                
                # Intersect
                intersection_nodes = list(set(branch1_nodes) & set(branch2_nodes))
                
                if not intersection_nodes:
                    results.append(self._create_abstention_result(gt, (time.time() - start_time) * 1000, ultra_calls=2))
                    continue
                
                # Project from intersection
                proj_nodes, proj_calls = self._process_hop_batch(ultra_model, graph_data, intersection_nodes, 
                                                     rel3, static_threshold, batch_size)
                
                execution_time_ms = (time.time() - start_time) * 1000
                precision, recall, f1, _ = self._compute_metrics_from_predictions(proj_nodes, gt)
                ultra_calls = 2 + proj_calls  # 2 branches + projection
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=proj_nodes,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(proj_nodes) == 0,
                    neo4j_calls=0,
                    ultra_calls=ultra_calls
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    
    def _has_neo4j_results(self, skip_neo4j: bool = False) -> bool:
        """Check if we have Neo4j results to save.
        
        Note: skip_neo4j only controls whether to RUN neo4j, not whether to SAVE existing results.
        If results exist in memory, they should be saved regardless of skip_neo4j flag.
        """
        return len(self.results['neo4j_symbolic']) > 0
    
    def _has_ultra_results(self) -> bool:
        """Check if we have Ultra Neural results to save."""
        return len(self.results['ultra_neural']) > 0
    
    def _has_hybrid_results(self) -> bool:
        """Check if we have Hybrid Static results to save."""
        return len(self.results['hybrid_static']) > 0
    
    def _create_pipeline_model(self, ultra_model, dbexec_model, args: argparse.Namespace, 
                              device: str, model_name: str):
        """Create the appropriate pipeline model based on model name."""
        from models.topology.three_hop_pipeline import ThreeHopPipeline
        from models.topology.two_union_pipeline import TwoUnionPipeline
        from models.topology.two_intersect_project_pipeline import TwoIntersectProjectPipeline
        
        if model_name == "ThreeHopPipeline":
            return ThreeHopPipeline(ultra_model, dbexec_model, args, device)
        elif model_name == "TwoUnionPipeline":
            return TwoUnionPipeline(ultra_model, dbexec_model, args, device)
        elif model_name == "TwoIntersectProjectPipeline":
            return TwoIntersectProjectPipeline(ultra_model, dbexec_model, args, device)
        else:
            raise ValueError(f"Unknown pipeline model: {model_name}")
    
    def run_hybrid_baseline(self, pipeline_model, queries: List[str], 
                           ground_truths: List[List[int]], 
                           graph_data: Any,
                           static_thresholds: List[float]) -> List[QueryResult]:
        """
        Run hybrid baseline (neural + symbolic) with static thresholds using pipeline classes.
        Routes to appropriate method based on query template.
        """
        # Use template from directory if available, otherwise detect from query string
        if hasattr(self, 'template_from_dir') and self.template_from_dir:
            template = self.template_from_dir
        else:
            template = self.detect_query_template(queries[0]) if queries else "3p"
        
        if template == "2u":
            return self.run_hybrid_baseline_2u(pipeline_model, queries, ground_truths, graph_data, static_thresholds)
        elif template == "2ip":
            return self.run_hybrid_baseline_2ip(pipeline_model, queries, ground_truths, graph_data, static_thresholds)
        else:
            return self.run_hybrid_baseline_3p(pipeline_model, queries, ground_truths, graph_data, static_thresholds)
    
    def run_hybrid_baseline_3p(self, pipeline_model, queries: List[str], 
                               ground_truths: List[List[int]], 
                               graph_data: Any,
                               static_thresholds: List[float]) -> List[QueryResult]:
        """
        Run hybrid baseline for 3p queries with static thresholds using ThreeHopPipeline.
        Uses the pipeline's predict_with_thresholds method with static thresholds.
        """
        self.logger.info(f"Running Hybrid Baseline (3p, thresholds={static_thresholds})...")
        
        if len(static_thresholds) < 3:
            # Default thresholds if not enough provided
            static_thresholds = static_thresholds + [0.4] * (3 - len(static_thresholds))
        
        threshold1, threshold2, threshold3 = static_thresholds[0], static_thresholds[1], static_thresholds[2]
        lamhat = [threshold1, threshold2, threshold3]
        
        results = []
        pipeline_model.eval()
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Hybrid (3p)")):
            try:
                entity_id, rel_types = self.parse_query_3p(query_str)
                query_tensor = torch.tensor([[entity_id] + rel_types], dtype=torch.long)
                
                start_time = time.time()
                # Use pipeline's predict_with_thresholds method
                result = pipeline_model.predict_with_thresholds(
                    query_tensor, lamhat, graph_data
                )
                predictions, neo4j_per_query, ultra_per_query = result[0], result[1], result[2]
                execution_time_ms = (time.time() - start_time) * 1000
                
                pred_values = predictions[0] if predictions else []
                precision, recall, f1, _ = self._compute_metrics_from_predictions(pred_values, gt)
                neo4j_calls = neo4j_per_query[0] if neo4j_per_query else 0
                ultra_calls = ultra_per_query[0] if ultra_per_query else 0
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=pred_values,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(pred_values) == 0,
                    neo4j_calls=neo4j_calls,
                    ultra_calls=ultra_calls
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def run_hybrid_baseline_2u(self, pipeline_model, queries: List[str], 
                               ground_truths: List[List[int]], 
                               graph_data: Any,
                               static_thresholds: List[float]) -> List[QueryResult]:
        """
        Run hybrid baseline for 2u queries with static thresholds using TwoUnionPipeline.
        """
        self.logger.info(f"Running Hybrid Baseline (2u, thresholds={static_thresholds})...")
        
        if len(static_thresholds) < 2:
            # Default thresholds if not enough provided
            static_thresholds = static_thresholds + [0.4] * (2 - len(static_thresholds))
        
        threshold1, threshold2 = static_thresholds[0], static_thresholds[1]
        lamhat = [threshold1, threshold2]
        
        results = []
        pipeline_model.eval()
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Hybrid (2u)")):
            try:
                anchor1, rel1, anchor2, rel2 = self.parse_query_2u(query_str)
                query_tensor = torch.tensor([[anchor1, rel1, anchor2, rel2]], dtype=torch.long)
                
                start_time = time.time()
                # Use pipeline's predict_with_thresholds method
                result = pipeline_model.predict_with_thresholds(
                    query_tensor, lamhat, graph_data
                )
                predictions, neo4j_per_query, ultra_per_query = result[0], result[1], result[2]
                execution_time_ms = (time.time() - start_time) * 1000
                
                pred_values = predictions[0] if predictions else []
                precision, recall, f1, _ = self._compute_metrics_from_predictions(pred_values, gt)
                neo4j_calls = neo4j_per_query[0] if neo4j_per_query else 0
                ultra_calls = ultra_per_query[0] if ultra_per_query else 0
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=pred_values,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(pred_values) == 0,
                    neo4j_calls=neo4j_calls,
                    ultra_calls=ultra_calls
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def run_hybrid_baseline_2ip(self, pipeline_model, queries: List[str], 
                                ground_truths: List[List[int]], 
                                graph_data: Any,
                                static_thresholds: List[float]) -> List[QueryResult]:
        """
        Run hybrid baseline for 2ip queries with static thresholds using TwoIntersectProjectPipeline.
        """
        self.logger.info(f"Running Hybrid Baseline (2ip, thresholds={static_thresholds})...")
        
        if len(static_thresholds) < 3:
            # Default thresholds if not enough provided
            static_thresholds = static_thresholds + [0.4] * (3 - len(static_thresholds))
        
        threshold_branch1, threshold_branch2, threshold_proj = static_thresholds[0], static_thresholds[1], static_thresholds[2]
        lamhat = [threshold_branch1, threshold_branch2, threshold_proj]
        
        results = []
        pipeline_model.eval()
        
        for i, (query_str, gt) in enumerate(tqdm(zip(queries, ground_truths), 
                                                   total=len(queries),
                                                   desc="Hybrid (2ip)")):
            try:
                anchor1, rel1, anchor2, rel2, rel3 = self.parse_query_2ip(query_str)
                query_tensor = torch.tensor([[anchor1, rel1, anchor2, rel2, rel3]], dtype=torch.long)
                
                start_time = time.time()
                # Use pipeline's predict_with_thresholds method
                result = pipeline_model.predict_with_thresholds(
                    query_tensor, lamhat, graph_data
                )
                predictions, neo4j_per_query, ultra_per_query = result[0], result[1], result[2]
                execution_time_ms = (time.time() - start_time) * 1000
                
                pred_values = predictions[0] if predictions else []
                precision, recall, f1, _ = self._compute_metrics_from_predictions(pred_values, gt)
                neo4j_calls = neo4j_per_query[0] if neo4j_per_query else 0
                ultra_calls = ultra_per_query[0] if ultra_per_query else 0
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=pred_values,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(pred_values) == 0,
                    neo4j_calls=neo4j_calls,
                    ultra_calls=ultra_calls
                ))
            except Exception as e:
                self.logger.error(f"Error processing query {i}: {e}")
                results.append(self._create_error_result(gt))
        
        return results
    
    def _has_hybrid_results(self) -> bool:
        """Check if we have Hybrid results to save."""
        return len(self.results['hybrid_static']) > 0
    
    def _write_csv_line(self, f, baseline_type: str, threshold: str, stats: Dict[str, float]):
        """Write a single CSV line for baseline results."""
        f.write(f"{baseline_type},{threshold},"
               f"{stats['precision']:.4f},{stats['recall']:.4f},{stats['f1']:.4f},"
               f"{stats['avg_time_ms']:.2f},{stats['num_queries']},{stats['num_abstentions']},"
               f"{stats['abstention_rate']:.4f},{stats['precision_excl_abstentions']:.4f},"
               f"{stats['recall_excl_abstentions']:.4f},{stats['f1_excl_abstentions']:.4f},"
               f"{stats.get('avg_neo4j_calls', 0):.2f},{stats.get('avg_ultra_calls', 0):.2f},"
               f"{stats.get('total_neo4j_calls', 0)},{stats.get('total_ultra_calls', 0)}\n")
    
    def _serialize_result(self, result: QueryResult) -> Dict:
        """Serialize a single QueryResult to dict."""
        return {
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'execution_time_ms': result.execution_time_ms,
            'num_predicted': len(result.predicted_values),
            'num_ground_truth': len(result.ground_truth),
            'is_abstention': result.is_abstention,
            'neo4j_calls': result.neo4j_calls,
            'ultra_calls': result.ultra_calls
        }
    
    def _print_stats(self, name: str, stats: Dict[str, float]):
        """Print statistics for a baseline."""
        self.logger.info(f"{name}:")
        self.logger.info(f"  Precision: {stats['precision']:.4f}")
        self.logger.info(f"  Recall:    {stats['recall']:.4f}")
        self.logger.info(f"  F1:        {stats['f1']:.4f}")
        self.logger.info(f"  Avg Time:  {stats['avg_time_ms']:.2f} ms")
        self.logger.info(f"  Queries:   {stats['num_queries']}")
        self.logger.info(f"  Neo4j calls: total {stats.get('total_neo4j_calls', 0)}, avg {stats.get('avg_neo4j_calls', 0):.2f}/query")
        self.logger.info(f"  ULTRA calls: total {stats.get('total_ultra_calls', 0)}, avg {stats.get('avg_ultra_calls', 0):.2f}/query")
        if stats.get('num_excluded', 0) > 0:
            self.logger.info(f"  Excluded (empty GT): {stats['num_excluded']}")
        if stats.get('num_abstentions', 0) > 0:
            self.logger.info(f"  Abstentions: {stats['num_abstentions']} ({stats['abstention_rate']*100:.2f}%)")
            self.logger.info(f"  Precision (excl. abstentions): {stats['precision_excl_abstentions']:.4f}")
            self.logger.info(f"  Recall (excl. abstentions):    {stats['recall_excl_abstentions']:.4f}")
            self.logger.info(f"  F1 (excl. abstentions):        {stats['f1_excl_abstentions']:.4f}")
    
    def _load_queries(self) -> Tuple[List[str], List[List[int]]]:
        """Load queries and ground truths from directory."""
        self.logger.info("Loading benchmark queries...")
        self.logger.info(f"Loading from: {self.query_dir}")
        queries = self.load_queries_from_dir()
        ground_truths = self.load_ground_truth_from_dir()
        min_len = min(len(queries), len(ground_truths))
        queries, ground_truths = queries[:min_len], ground_truths[:min_len]
        self.logger.info(f"Loaded {len(queries)} queries")
        return queries, ground_truths
    
    def _load_models_and_queries(self, load_path: Optional[str], device: str,
                                 neo4j_host: str, neo4j_bolt_port: int,
                                 node_unique_id: str, relation_unique_id: str,
                                 scores_cache: Optional[List[Dict]], skip_neo4j: bool, skip_ultra: bool,
                                 skip_hybrid: bool = True):
        """
        Load models, graph data, and queries.
        
        Returns:
            Tuple of (ultra_model, graph_data, db_controller, queries, ground_truths, hybrid_model)
        """
        # Skip model/graph loading if we have cached scores AND skipping neo4j
        # (only need to apply thresholds, no model inference or neo4j queries needed)
        if scores_cache is not None and skip_neo4j and skip_hybrid:
            self.logger.info("Using cached scores - skipping model/graph loading")
            queries, ground_truths = self._load_queries()
            return None, None, None, queries, ground_truths, None
        
        # Setup models
        self.logger.info("Setting up models...")
        
        # Setup DB controller (needed for Neo4j baseline)
        db_controller = Neo4JBackendDBController(
            f"neo4j://{neo4j_host}:{neo4j_bolt_port}",
            node_unique_id,
            relation_unique_id,
        )
        
        ultra_model = None
        graph_data = None
        
        # Only load Ultra model if we're running Ultra baseline
        if not skip_ultra:
            # Create inference args namespace for Ultra
            inf_args = argparse.Namespace(
                load_path=load_path,
                device=device,
                neo4j_host=neo4j_host,
                neo4j_bolt_port=neo4j_bolt_port,
                node_unique_id=node_unique_id,
                relation_unique_id=relation_unique_id,
                model_to_infer="ULTRA",
                db_controller=db_controller,
            )
            
            # Resolve load_path
            inf_args.load_path = self._resolve_load_path(load_path)
            if not inf_args.load_path:
                raise ValueError("--load-path is required for Ultra baseline. Please specify the path to pretrained model directory.")
            self.logger.info(f"Using load_path: {inf_args.load_path}")
            
            # Set msp_threshold=0.0 for UltraQuery (requirement from UltraQuery paper)
            if self.use_ultraquery:
                inf_args.msp_threshold = 0.0
                self.logger.info("Setting msp_threshold=0.0 for UltraQuery checkpoint")
            
            # Load Ultra model
            self.logger.info("Loading Ultra model...")
            inf_args_ultra = argparse.Namespace(**vars(inf_args))
            inf_args_ultra.model_to_infer = "ULTRA"
            model_factory_ultra = ModelFactory(inf_args_ultra)
            model_factory_ultra.prepare_model()
            ultra_model = model_factory_ultra.model
            
            # Load graph data (needed for Ultra)
            self.logger.info("Loading graph data...")
            graph_data = get_graph(db_controller, device)
        else:
            self.logger.info("Skipping Ultra model loading (symbolic baseline only)")
        
        # Load hybrid pipeline model if needed
        hybrid_model = None
        if not skip_hybrid:
            self.logger.info("Loading hybrid pipeline model...")
            if ultra_model is None or graph_data is None:
                # Need to load Ultra model first
                if ultra_model is None:
                    inf_args = argparse.Namespace(
                        load_path=load_path,
                        device=device,
                        neo4j_host=neo4j_host,
                        neo4j_bolt_port=neo4j_bolt_port,
                        node_unique_id=node_unique_id,
                        relation_unique_id=relation_unique_id,
                        model_to_infer="ULTRA",
                        db_controller=db_controller,
                    )
                    inf_args.load_path = self._resolve_load_path(load_path)
                    if not inf_args.load_path:
                        raise ValueError("--load-path is required for hybrid baseline.")
                    # Set msp_threshold=0.0 for UltraQuery (requirement from UltraQuery paper)
                    if self.use_ultraquery:
                        inf_args.msp_threshold = 0.0
                        self.logger.info("Setting msp_threshold=0.0 for UltraQuery checkpoint")
                    inf_args_ultra = argparse.Namespace(**vars(inf_args))
                    inf_args_ultra.model_to_infer = "ULTRA"
                    model_factory_ultra = ModelFactory(inf_args_ultra)
                    model_factory_ultra.prepare_model()
                    ultra_model = model_factory_ultra.model
                
                if graph_data is None:
                    graph_data = get_graph(db_controller, device)
            
            # Create DBExecModel
            from models.db_exec.model import DBExecModel
            num_relations = db_controller.get_num_relations()
            dbexec_args = argparse.Namespace(db_controller=db_controller)
            dbexec_model = DBExecModel(dbexec_args, num_relations=num_relations, device=device)
            
            # Determine pipeline model name from query directory
            model_name = _get_model_from_query_dir(self.query_dir)
            
            # Create pipeline model using the factory method
            num_entities = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else None
            pipeline_args = argparse.Namespace(
                device=device,
                db_controller=db_controller,
                max_internal_batch=getattr(self, 'ultra_batch_size', 64),
                calib_batch_size=getattr(self, 'ultra_batch_size', 64),
                num_entities=num_entities,
            )
            hybrid_model = self._create_pipeline_model(ultra_model, dbexec_model, pipeline_args, device, model_name)
        
        # Load queries
        queries, ground_truths = self._load_queries()
        
        return ultra_model, graph_data, db_controller, queries, ground_truths, hybrid_model
    
    def _run_baselines(self, skip_neo4j: bool, skip_ultra: bool, skip_hybrid: bool, compute_scores_only: bool,
                      db_controller, queries: List[str], ground_truths: List[List[int]],
                      ultra_model, graph_data, static_threshold: float,
                      scores_cache: Optional[List[Dict]], min_threshold: float,
                      output_dir: str, hybrid_model=None, hybrid_thresholds: Optional[List[float]] = None) -> Optional[Dict]:
        """
        Run Neo4j, Ultra, and/or Hybrid baselines based on skip flags.
        
        Returns:
            Dict with 'scores_cache' if compute_scores_only is True, None otherwise
        """
        # Run Neo4j baseline
        if not skip_neo4j:
            self.logger.info("\n" + "-"*80)
            self.logger.info("Running Neo4j Symbolic Baseline")
            self.logger.info("-"*80)
            neo4j_results = self.run_neo4j_baseline(db_controller, queries, ground_truths)
            self.results['neo4j_symbolic'].extend(neo4j_results)
        
        # Run Ultra baseline
        if not skip_ultra:
            if compute_scores_only:
                # Only compute scores, don't apply threshold
                self.logger.info("\n" + "-"*80)
                self.logger.info("Computing all Ultra scores")
                self.logger.info("-"*80)
                self.ultra_scores_cache = self.compute_ultra_scores(
                    ultra_model, queries, ground_truths, graph_data,
                    min_threshold=min_threshold,
                    batch_size=self.ultra_batch_size
                )
                
                # Still save Neo4j results if we ran it
                if not skip_neo4j:
                    self.save_results(output_dir, skip_neo4j=False)
                
                return {'scores_cache': self.ultra_scores_cache}
            else:
                # Apply threshold to scores (either from cache or compute on the fly)
                self.logger.info("\n" + "-"*80)
                self.logger.info("Running Ultra Neural Baseline")
                self.logger.info("-"*80)
                ultra_results = self.run_ultra_pipeline_baseline(
                    ultra_model, queries, ground_truths, graph_data, static_threshold,
                    batch_size=self.ultra_batch_size, scores_cache=scores_cache
                )
                self.results['ultra_neural'].extend(ultra_results)
        
        # Run Hybrid baseline
        if not skip_hybrid and hybrid_model is not None:
            self.logger.info("\n" + "-"*80)
            self.logger.info("Running Hybrid Baseline (neural + symbolic)")
            self.logger.info("-"*80)
            if hybrid_thresholds is None:
                # Default thresholds based on query type
                if hasattr(self, 'template_from_dir') and self.template_from_dir:
                    template = self.template_from_dir
                else:
                    template = self.detect_query_template(queries[0]) if queries else "3p"
                
                if template == "2u":
                    hybrid_thresholds = [static_threshold, static_threshold]
                elif template == "2ip":
                    hybrid_thresholds = [static_threshold, static_threshold, static_threshold]
                elif template == "3p":
                    hybrid_thresholds = [static_threshold, static_threshold, static_threshold]
            
            # Store the first threshold for reporting (all thresholds are typically the same)
            self.hybrid_threshold = hybrid_thresholds[0] if hybrid_thresholds else None
            
            hybrid_results = self.run_hybrid_baseline(
                hybrid_model, queries, ground_truths, graph_data, hybrid_thresholds
            )
            self.results['hybrid_static'].extend(hybrid_results)
            return None
    
    def _resolve_load_path(self, load_path: Optional[str]) -> Optional[str]:
        """Resolve load_path from provided value or auto-detect."""
        if load_path:
            return load_path
        
        # If use_ultraquery flag is set, use UltraQuery checkpoint
        if self.use_ultraquery:
            repo_root = _get_repo_root()
            ultraquery_dir = os.path.join(repo_root, "artifacts", "snapshots", "ultraquery")
            if os.path.exists(ultraquery_dir):
                self.logger.info(f"Using UltraQuery checkpoint directory: {ultraquery_dir}")
                return ultraquery_dir
            else:
                self.logger.error(f"UltraQuery directory not found at: {ultraquery_dir}")
                raise ValueError(f"UltraQuery directory not found at: {ultraquery_dir}. Please ensure artifacts/snapshots/ultraquery/ exists.")
        
        # Auto-detect from snapshots
        for search_path in ["./artifacts/snapshots", "../artifacts/snapshots"]:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    if "ultra_args.json" in files:
                        return root
        
        return None
    
    def compute_summary_stats(self, results: List[QueryResult]) -> Dict[str, float]:
        """Compute summary statistics from results, excluding queries with empty ground truth."""
        # Default empty stats
        empty_stats = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'avg_time_ms': 0.0,
            'num_queries': 0, 'num_excluded': 0, 'num_abstentions': 0,
            'abstention_rate': 0.0, 'precision_excl_abstentions': 0.0,
            'recall_excl_abstentions': 0.0, 'f1_excl_abstentions': 0.0,
            'total_neo4j_calls': 0, 'total_ultra_calls': 0,
            'avg_neo4j_calls': 0.0, 'avg_ultra_calls': 0.0,
        }
        
        if not results:
            return empty_stats
        
        valid_results = [r for r in results if len(r.ground_truth) > 0]
        num_excluded = len(results) - len(valid_results)
        
        if not valid_results:
            return {**empty_stats, 'num_excluded': num_excluded}
        
        abstentions = [r for r in valid_results if r.is_abstention]
        num_abstentions = len(abstentions)
        abstention_rate = num_abstentions / len(valid_results)
        
        non_abstention_results = [r for r in valid_results if not r.is_abstention]
        if non_abstention_results:
            precision_excl = np.mean([r.precision for r in non_abstention_results])
            recall_excl = np.mean([r.recall for r in non_abstention_results])
            f1_excl = np.mean([r.f1 for r in non_abstention_results])
        else:
            precision_excl = recall_excl = f1_excl = 0.0
        
        total_neo4j = sum(r.neo4j_calls for r in valid_results)
        total_ultra = sum(r.ultra_calls for r in valid_results)
        nq = len(valid_results)
        return {
            'precision': np.mean([r.precision for r in valid_results]),
            'recall': np.mean([r.recall for r in valid_results]),
            'f1': np.mean([r.f1 for r in valid_results]),
            'avg_time_ms': np.mean([r.execution_time_ms for r in valid_results]),
            'num_queries': nq,
            'num_excluded': num_excluded,
            'num_abstentions': num_abstentions,
            'abstention_rate': abstention_rate,
            'precision_excl_abstentions': precision_excl,
            'recall_excl_abstentions': recall_excl,
            'f1_excl_abstentions': f1_excl,
            'total_neo4j_calls': total_neo4j,
            'total_ultra_calls': total_ultra,
            'avg_neo4j_calls': total_neo4j / nq if nq else 0.0,
            'avg_ultra_calls': total_ultra / nq if nq else 0.0,
        }
    
    def save_results(self, output_dir: str, skip_neo4j: bool = False, json_suffix: str = None):
        """Save results to files.
        
        Args:
            output_dir: Directory to save results
            skip_neo4j: If True, skip Neo4j results (deprecated, now auto-detected)
            json_suffix: Optional suffix for the detailed JSON filename (e.g., "_0.7" -> baseline_detailed_results_0.7.json)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save baseline results summary
        baseline_summary_path = os.path.join(output_dir, "baseline_results_summary.csv")
        file_exists = os.path.exists(baseline_summary_path)
        
        # Use append mode if file exists, write mode if it doesn't
        mode = 'a' if file_exists else 'w'
        with open(baseline_summary_path, mode) as f:
            # Write header only if file doesn't exist
            if not file_exists:
                f.write("baseline,threshold,precision,recall,f1,avg_time_ms,num_queries,num_abstentions,abstention_rate,precision_excl_abstentions,recall_excl_abstentions,f1_excl_abstentions,avg_neo4j_calls,avg_ultra_calls,total_neo4j_calls,total_ultra_calls\n")
            
            # Save neo4j results if they exist (regardless of skip_neo4j flag)
            if self._has_neo4j_results():
                neo4j_stats = self.compute_summary_stats(self.results['neo4j_symbolic'])
                self._write_csv_line(f, "symbolic", "N/A", neo4j_stats)
            
            # Save Ultra Neural results only if they exist
            if self._has_ultra_results():
                ultra_stats = self.compute_summary_stats(self.results['ultra_neural'])
                threshold_str = f"{self.threshold:.2f}" if self.threshold else "0.75"
                self._write_csv_line(f, "neural", threshold_str, ultra_stats)
            
            # Save Hybrid results only if they exist
            if self._has_hybrid_results():
                hybrid_stats = self.compute_summary_stats(self.results['hybrid_static'])
                # Use stored hybrid threshold or fallback to static_threshold
                if self.hybrid_threshold is not None:
                    threshold_str = f"{self.hybrid_threshold:.2f}"
                elif self.threshold is not None:
                    threshold_str = f"{self.threshold:.2f}"
                else:
                    threshold_str = "0.75"
                self._write_csv_line(f, "hybrid", threshold_str, hybrid_stats)
        
        self.logger.info(f"Baseline summary saved to: {baseline_summary_path}")
        
        # Save detailed results as JSON
        detailed_results = {}
        # Save neo4j results if they exist (regardless of skip_neo4j flag)
        if self._has_neo4j_results():
            detailed_results['neo4j_symbolic'] = [
                self._serialize_result(r) for r in self.results['neo4j_symbolic']
            ]
        # Save Ultra Neural results only if they exist
        if self._has_ultra_results():
            detailed_results['ultra_neural'] = [
                self._serialize_result(r) for r in self.results['ultra_neural']
            ]
        # Save Hybrid results only if they exist
        if self._has_hybrid_results():
            detailed_results['hybrid_static'] = [
                self._serialize_result(r) for r in self.results['hybrid_static']
            ]
        
        json_filename = f"baseline_detailed_results{json_suffix}.json" if json_suffix else "baseline_detailed_results.json"
        detailed_path = os.path.join(output_dir, json_filename)
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        self.logger.info(f"Detailed results saved to: {detailed_path}")
        
        # Print summary
        self.logger.info("\n" + "="*80)
        self.logger.info("BASELINE RESULTS SUMMARY")
        self.logger.info("="*80)
        
        # Print neo4j results if they exist (regardless of skip_neo4j flag)
        if self._has_neo4j_results():
            neo4j_stats = self.compute_summary_stats(self.results['neo4j_symbolic'])
            self._print_stats("Neo4j Symbolic", neo4j_stats)
        
        # Print Ultra Neural results only if they exist
        if self._has_ultra_results():
            ultra_stats = self.compute_summary_stats(self.results['ultra_neural'])
            threshold_str = f"{self.threshold:.2f}" if self.threshold else "0.75"
            self._print_stats(f"Ultra Neural (threshold={threshold_str})", ultra_stats)
        
        # Print Hybrid results only if they exist
        if self._has_hybrid_results():
            hybrid_stats = self.compute_summary_stats(self.results['hybrid_static'])
            threshold_str = f"{self.threshold:.2f}" if self.threshold else "0.75"
            self._print_stats(f"Hybrid Static (threshold={threshold_str})", hybrid_stats)
    
    def run(self, output_dir: str, static_threshold: float = 0.75, 
            load_path: str = None, device: str = "cuda",
            neo4j_host: str = "localhost", neo4j_bolt_port: int = 7687,
            node_unique_id: str = "id", relation_unique_id: str = "type",
            skip_neo4j: bool = False, skip_ultra: bool = False, skip_hybrid: bool = True,
            compute_scores_only: bool = False, scores_cache: Optional[List[Dict]] = None,
            min_threshold: float = 0.0, hybrid_thresholds: Optional[List[float]] = None,
            json_suffix: str = None):
        """
        Run both baselines on benchmark queries.
        
        Args:
            output_dir: Directory to save results
            static_threshold: Threshold for Ultra neural baseline
            load_path: Path to pretrained model directory
            device: Device to use (cuda, cpu, mps)
            neo4j_host: Neo4j host
            neo4j_bolt_port: Neo4j bolt port
            node_unique_id: Node unique identifier field
            relation_unique_id: Relation unique identifier field
            skip_neo4j: If True, skip Neo4j baseline (only run Ultra)
            skip_ultra: If True, skip Ultra baseline (only run Neo4j)
            compute_scores_only: If True, only compute scores (no threshold application)
            scores_cache: Pre-computed Ultra scores (from previous run)
            min_threshold: Minimum threshold for computing scores (to avoid computing for all nodes)
            json_suffix: Optional suffix for the detailed JSON filename (e.g., "_0.7")
        """
        # Store threshold for reporting
        self.threshold = static_threshold
        # Store hybrid threshold if provided
        if hybrid_thresholds is not None and len(hybrid_thresholds) > 0:
            self.hybrid_threshold = hybrid_thresholds[0]
        
        self.logger.info("="*80)
        self.logger.info("BASELINE BENCHMARK RUNNER")
        self.logger.info("="*80)
        self.logger.info(f"Query directory: {self.query_dir}")
        self.logger.info(f"Max queries: {self.max_queries}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Threshold: {static_threshold}")
        self.logger.info("="*80)
        
        # Load models and queries
        ultra_model, graph_data, db_controller, queries, ground_truths, hybrid_model = self._load_models_and_queries(
            load_path, device, neo4j_host, neo4j_bolt_port,
            node_unique_id, relation_unique_id, scores_cache, skip_neo4j, skip_ultra, skip_hybrid
        )
                
        # Run baselines
        result = self._run_baselines(
            skip_neo4j, skip_ultra, skip_hybrid, compute_scores_only,
            db_controller, queries, ground_truths,
            ultra_model, graph_data, static_threshold,
            scores_cache, min_threshold, output_dir,
            hybrid_model, hybrid_thresholds
        )
        
        # If compute_scores_only, return early (results already saved if needed)
        if result is not None:
            return result
        
        # Save results
        self.logger.info("\n" + "="*80)
        self.logger.info("Saving results...")
        self.logger.info("="*80)
        self.save_results(output_dir, skip_neo4j=skip_neo4j, json_suffix=json_suffix)
                
        return self.results


def _get_repo_root() -> str:
    """Get the repository root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def _get_model_from_query_dir(query_dir: str) -> str:
    """
    Extract model name from query directory path.
    
    Maps:
    - 3p_pipeline → ThreeHopPipeline
    - 2u_pipeline → TwoUnionPipeline
    - 2ip_pipeline → TwoIntersectProjectPipeline
    """
    query_type_to_model = {
        "3p_pipeline": "ThreeHopPipeline",
        "2u_pipeline": "TwoUnionPipeline",
        "2ip_pipeline": "TwoIntersectProjectPipeline",
    }
    
    for query_type, model_name in query_type_to_model.items():
        if query_dir.endswith(query_type):
            return model_name
    
    # Fallback: try to extract from path
    basename = os.path.basename(query_dir.rstrip('/'))
    if basename in query_type_to_model:
        return query_type_to_model[basename]
    
    # Default fallback
    return "UnknownModel"


def _setup_output_directory(output_dir: str, dataset: str = None, 
                           baseline_type: str = None, model_to_infer: str = None,
                           incompleteness: int = None) -> str:
    """
    Setup output directory path.
    
    If output_dir is None, constructs it from dataset, baseline_type, model_to_infer, and incompleteness.
    Otherwise, resolves relative paths to be relative to repo root.
    
    Format: <baseline_name>_bench_<dataset>_<model_to_infer>_<incompleteness>
    """
    repo_root = _get_repo_root()
    
    if output_dir is None:
        if not all([dataset, baseline_type, model_to_infer, incompleteness is not None]):
            raise ValueError("dataset, baseline_type, model_to_infer, and incompleteness are required when output_dir is not provided")
        
        folder_name = f"{baseline_type}_bench_{dataset}_{model_to_infer}_{incompleteness}"
        benchmark_dir = os.path.join(repo_root, "artifacts", "benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        return os.path.join(benchmark_dir, folder_name)
    
    # Resolve relative paths
    if not os.path.isabs(output_dir):
        return os.path.join(repo_root, output_dir)
    
    return output_dir


def _get_skip_flags_from_baseline_type(baseline_type: str) -> Tuple[bool, bool]:
    """
    Get skip_neo4j and skip_ultra flags based on baseline type.
    
    Returns:
        Tuple of (skip_neo4j, skip_ultra)
    """
    baseline_configs = {
        "neural": (True, False),   # Skip Neo4j, run Ultra
        "symbolic": (False, True),  # Run Neo4j, skip Ultra
        "hybrid": (True, True),     # Skip both individual baselines, run hybrid
    }
    return baseline_configs.get(baseline_type, (False, False))


def _print_configuration(dataset: str = None, baseline_type: str = None, 
                        incompleteness: int = None, output_dir: str = None,
                        model_to_infer: str = None):
    """Print benchmark configuration."""
    print("="*80)
    print("BASELINE BENCHMARK RUNNER")
    print("="*80)
    if dataset:
        print(f"Dataset: {dataset}")
    if baseline_type:
        print(f"Baseline type: {baseline_type}")
    if model_to_infer:
        print(f"Model to infer: {model_to_infer}")
    if incompleteness is not None:
        print(f"Incompleteness level: {incompleteness}%")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print("="*80)
    print()


def _run_with_multiple_thresholds(runner: BaselineRunner, args, skip_neo4j: bool, skip_ultra: bool):
    """Run benchmarks with multiple thresholds (compute scores once, apply all thresholds)."""
    # Use first threshold as min_threshold if not explicitly provided (and not 0.0)
    min_threshold = args.min_threshold if args.min_threshold != 0.0 else args.ultra_thresholds[0]
    
    # Compute scores once
    result = runner.run(
        output_dir=args.output_dir,
        static_threshold=args.ultra_thresholds[0],  # Dummy, will be overridden
        load_path=args.load_path,
        device=args.device,
        neo4j_host=args.neo4j_host,
        neo4j_bolt_port=args.neo4j_bolt_port,
        node_unique_id=args.node_unique_id,
        relation_unique_id=args.relation_unique_id,
        skip_neo4j=skip_neo4j,
        skip_ultra=skip_ultra,
        skip_hybrid=True,  # Skip hybrid for multiple threshold sweeps (Ultra only)
        compute_scores_only=True,
        min_threshold=min_threshold
    )
    scores_cache = result.get('scores_cache')
    
    # Apply each threshold - save all results in the same directory
    # CSV is appended automatically, JSON gets threshold suffix
    for threshold in args.ultra_thresholds:
        runner.results['ultra_neural'] = []  # Clear previous results
        runner.run(
            output_dir=args.output_dir,
            static_threshold=threshold,
            load_path=args.load_path,
            device=args.device,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            node_unique_id=args.node_unique_id,
            relation_unique_id=args.relation_unique_id,
            skip_neo4j=True,  # Skip Neo4j for threshold runs (already computed once)
            skip_ultra=skip_ultra,
            skip_hybrid=True,  # Skip hybrid for threshold runs
            scores_cache=scores_cache,
            min_threshold=min_threshold,
            json_suffix=f"_{threshold}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline benchmarks for comparison with ThreeHopPipeline"
    )
    parser.add_argument("--query-dir", type=str, 
                       default="../queries/test/3p_pipeline",
                       help="Directory containing test queries (default: ../queries/test/3p_pipeline)")
    parser.add_argument("--max-queries", type=int, default=1000,
                       help="Maximum number of queries to process")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: artifacts/benchmark/<baseline_type>_bench_<dataset>_<model_to_infer>_<incompleteness>)")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["fb15k-237", "nell-955", "yago310"],
                       help="Dataset name: fb15k-237, nell-955, or yago310")
    parser.add_argument("--baseline-type", type=str, default=None,
                       choices=["neural", "symbolic", "hybrid"],
                       help="Baseline type: neural (Ultra), symbolic (Neo4j), or hybrid (neural + symbolic)")
    parser.add_argument("--incompleteness", type=int, default=None,
                       help="Data incompleteness level (e.g., 20 for 20% missing)")
    parser.add_argument("--threshold", type=float, default=0.75,
                       help="Static threshold for neural baseline (and default for hybrid if --hybrid-thresholds not provided)")
    parser.add_argument("--ultra-batch-size", type=int, default=4,
                       help="Batch size for Ultra intermediate hops (to avoid GPU OOM, default: 16)")
    parser.add_argument("--load-path", type=str, default=None,
                       help="Path to pretrained model directory (required for Ultra baseline)")
    parser.add_argument("--use-ultraquery", action="store_true", default=False,
                       help="Use the official UltraQuery checkpoint (ultraquery.pth) instead of dataset-specific models")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--neo4j-host", type=str, default="localhost",
                       help="Neo4j host (default: localhost)")
    parser.add_argument("--neo4j-bolt-port", type=int, default=7687,
                       help="Neo4j bolt port (default: 7687)")
    parser.add_argument("--node-unique-id", type=str, default="id",
                       help="Node unique identifier field (default: id)")
    parser.add_argument("--relation-unique-id", type=str, default="type",
                       help="Relation unique identifier field (default: type)")
    parser.add_argument("--skip-neo4j", action="store_true",
                       help="Skip Neo4j baseline (only run Ultra)")
    parser.add_argument("--compute-scores-only", action="store_true",
                       help="Only compute Ultra scores, don't apply threshold")
    parser.add_argument("--min-threshold", type=float, default=0.0,
                       help="Minimum threshold for computing scores (to avoid computing for all nodes)")
    parser.add_argument("--ultra-thresholds", type=float, nargs='+', default=None,
                       help="Multiple thresholds to apply (if provided, computes scores once and applies all thresholds)")
    parser.add_argument("--skip-hybrid", action="store_true",
                       help="Skip hybrid baseline (neural + symbolic with static thresholds)")
    parser.add_argument("--hybrid-thresholds", type=float, nargs='+', default=None,
                       help="Static thresholds for hybrid baseline (default: use --threshold for all hops)")
    
    args = parser.parse_args()
    
    # Extract model name from query directory
    model_to_infer = _get_model_from_query_dir(args.query_dir) if args.query_dir else None
    
    # Setup output directory
    try:
        args.output_dir = _setup_output_directory(
            args.output_dir, args.dataset, args.baseline_type, model_to_infer, args.incompleteness
        )
    except ValueError as e:
        parser.error(str(e))
    
    # Set skip flags based on baseline type
    skip_neo4j, skip_ultra = _get_skip_flags_from_baseline_type(args.baseline_type) if args.baseline_type else (args.skip_neo4j, False)
    if args.baseline_type:
        args.skip_neo4j = skip_neo4j
        if args.baseline_type == "hybrid":
            skip_hybrid = False  # Run hybrid when baseline_type is "hybrid"
        elif args.baseline_type in ("symbolic", "neural"):
            skip_hybrid = True  # Skip hybrid when running symbolic or neural only
        else:
            skip_hybrid = args.skip_hybrid
    else:
        skip_hybrid = args.skip_hybrid
    
    # Print configuration
    _print_configuration(args.dataset, args.baseline_type, args.incompleteness, args.output_dir, model_to_infer)
    
    # Create baseline runner
    runner = BaselineRunner(
        config_path=None,  # No longer needed
        query_dir=args.query_dir,
        max_queries=args.max_queries,
        ultra_batch_size=args.ultra_batch_size,
        use_ultraquery=args.use_ultraquery
    )
    
    # Run benchmarks
    if args.ultra_thresholds and len(args.ultra_thresholds) >= 1:
        _run_with_multiple_thresholds(runner, args, args.skip_neo4j, skip_ultra)
    else:
        # Determine json_suffix for hybrid baselines (to create separate JSON files per threshold)
        json_suffix = None
        if args.baseline_type == "hybrid" and args.hybrid_thresholds and len(args.hybrid_thresholds) > 0:
            json_suffix = f"_{args.hybrid_thresholds[0]}"
        
        runner.run(
            output_dir=args.output_dir,
            static_threshold=args.threshold,
            load_path=args.load_path,
            device=args.device,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            node_unique_id=args.node_unique_id,
            relation_unique_id=args.relation_unique_id,
            skip_neo4j=args.skip_neo4j,
            skip_ultra=skip_ultra,
            skip_hybrid=skip_hybrid,
            compute_scores_only=args.compute_scores_only,
            min_threshold=args.min_threshold,
            hybrid_thresholds=args.hybrid_thresholds,
            json_suffix=json_suffix
        )


if __name__ == "__main__":
    main()

