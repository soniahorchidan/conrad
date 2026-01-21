#!/usr/bin/env python3
"""
Run baseline benchmarks for comparison with ThreeHopPipeline results.

Two baselines:
1. DBExecModel (Neo4j symbolic execution) - runs full 3-hop query in Neo4j
2. Ultra with static threshold (neural) - runs 3 separate hops with static threshold (e.g., 0.75)

This script:
- Loads the same benchmark queries as run_crc_benchmark_auto.py
- Runs both baselines
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

from inference import ModelFactory, parse_args_inference
from utils import merge_args, set_logger, parse_time, get_embeddings, get_graph
from graph_handler import Neo4JBackendDBController
from conformal_prediction.utils import compute_fnr_metrics
import argparse as argparse_module


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


# TODO(Sonia): split this into multiple runners
class BaselineRunner:
    """
    Runs baseline benchmarks on the same queries used by ThreeHopPipeline.
    """
    
    def __init__(self, config_path: str, query_dir: str, max_queries: int = 1000, 
                 ultra_batch_size: int = 16):
        """
        Initialize baseline runner.
        
        Args:
            config_path: Path to orb config JSON file
            query_dir: Directory containing benchmark queries (should end with _pipeline)
            max_queries: Maximum number of queries to process
            ultra_batch_size: Batch size for Ultra intermediate hops (to avoid GPU OOM)
        """
        self.config_path = config_path
        self.query_dir = query_dir
        self.max_queries = max_queries
        self.ultra_batch_size = ultra_batch_size
        
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
            'ultra_neural': []
        }
        
        # Store threshold for reporting
        self.ultra_threshold = None
        
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
    
    def _create_abstention_result(self, gt: List, execution_time_ms: float) -> QueryResult:
        """Create a QueryResult for an abstention (empty prediction)."""
        gt_normalized = self._normalize_ground_truth(gt)
        return QueryResult(
            precision=0.0 if len(gt_normalized) > 0 else 1.0,
            recall=0.0,
            f1=0.0,
            predicted_values=[],
            ground_truth=gt_normalized,
            execution_time_ms=execution_time_ms,
            is_abstention=True
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
                          rel_type: int, threshold: float, batch_size: int) -> List[int]:
        """Process a batch of nodes through a hop. Returns aggregated candidate nodes."""
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
        
        return list(candidates_set)
    
    def _process_hop_batch_get_scores(self, ultra_model, graph_data, nodes: List[int], 
                                      rel_type: int, batch_size: int) -> Dict[int, torch.Tensor]:
        """Process a batch of nodes through a hop and return all scores (no threshold).
        Returns dict mapping node -> scores tensor."""
        node_scores = {}
        for batch_start in range(0, len(nodes), batch_size):
            batch_end = min(batch_start + batch_size, len(nodes))
            batch_nodes = nodes[batch_start:batch_end]
            
            queries = torch.tensor([[node, rel_type] for node in batch_nodes], 
                                  dtype=torch.long).to(graph_data.edge_index.device)
            with torch.no_grad():
                scores = ultra_model.forward(graph_data, queries)
            
            for j, node in enumerate(batch_nodes):
                node_scores[node] = scores[j]
            
            # Clear GPU cache after each batch to prevent memory accumulation
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
                    is_abstention=len(pred_values) == 0
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
                    is_abstention=len(pred_values) == 0
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
                    is_abstention=len(pred_values) == 0
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
                    hop1_scores = ultra_model.forward(graph_data, hop1_query)[0]
                
                # Hop 2: compute scores only for nodes that pass min_threshold
                hop1_candidates = self._apply_threshold(hop1_scores, min_threshold)
                hop2_scores = self._process_hop_batch_get_scores(ultra_model, graph_data, 
                                                                 hop1_candidates, rel_types[1], batch_size)
                
                # Hop 3: compute scores only for nodes from hop2 that pass min_threshold
                hop2_candidates = []
                for node, scores in hop2_scores.items():
                    candidates = self._apply_threshold(scores, min_threshold)
                    hop2_candidates.extend(candidates)
                hop2_candidates = list(set(hop2_candidates))  # Remove duplicates
                hop3_scores = self._process_hop_batch_get_scores(ultra_model, graph_data,
                                                                 hop2_candidates, rel_types[2], batch_size)
                
                execution_time_ms = (time.time() - start_time) * 1000  # Store inference time
                
                scores_data.append({
                    'hop1_scores': hop1_scores,
                    'hop2_scores': hop2_scores,
                    'hop3_scores': hop3_scores,
                    'rel_types': rel_types,
                    'entity_id': entity_id,
                    'ground_truth': gt,
                    'execution_time_ms': execution_time_ms,
                    'template': '3p'
                })
            except Exception as e:
                self.logger.error(f"Error computing scores for query {i}: {e}")
                # Clear GPU cache to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                # Clear GPU cache after each query to prevent memory accumulation
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
                    branch1_scores = ultra_model.forward(graph_data, branch1_query)[0]
                
                # Branch 2: compute scores for all nodes
                branch2_query = torch.tensor([[anchor2, rel2]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch2_scores = ultra_model.forward(graph_data, branch2_query)[0]
                
                execution_time_ms = (time.time() - start_time) * 1000  # Store inference time
                
                scores_data.append({
                    'branch1_scores': branch1_scores,
                    'branch2_scores': branch2_scores,
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
                # Clear GPU cache to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                # Clear GPU cache after each query to prevent memory accumulation
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
                    branch1_scores = ultra_model.forward(graph_data, branch1_query)[0]
                
                # Branch 2: compute scores for all nodes
                branch2_query = torch.tensor([[anchor2, rel2]], dtype=torch.long).to(device)
                with torch.no_grad():
                    branch2_scores = ultra_model.forward(graph_data, branch2_query)[0]
                
                # Compute intersection candidates (nodes that pass min_threshold in both branches)
                branch1_candidates = self._apply_threshold(branch1_scores, min_threshold)
                branch2_candidates = self._apply_threshold(branch2_scores, min_threshold)
                intersection_candidates = list(set(branch1_candidates) & set(branch2_candidates))
                
                # Project scores from intersection candidates
                proj_scores = {}
                if intersection_candidates:
                    proj_scores = self._process_hop_batch_get_scores(ultra_model, graph_data,
                                                                     intersection_candidates, rel3, batch_size)
                
                execution_time_ms = (time.time() - start_time) * 1000  # Store inference time
                
                scores_data.append({
                    'branch1_scores': branch1_scores,
                    'branch2_scores': branch2_scores,
                    'proj_scores': proj_scores,  # Dict[node -> scores] for projection
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
                
                # Hop 1: apply threshold
                hop1_nodes = self._apply_threshold(data['hop1_scores'], threshold)
                if not hop1_nodes:
                    results.append(self._create_abstention_result(data['ground_truth'], execution_time_ms))
                    continue
                
                # Hop 2: apply threshold to cached scores
                hop2_nodes_set = set()
                for hop1_node in hop1_nodes:
                    if hop1_node in data.get('hop2_scores', {}):
                        candidates = self._apply_threshold(data['hop2_scores'][hop1_node], threshold)
                        hop2_nodes_set.update(candidates)
                hop2_nodes = list(hop2_nodes_set)
                
                if not hop2_nodes:
                    results.append(self._create_abstention_result(data['ground_truth'], execution_time_ms))
                    continue
                
                # Hop 3: apply threshold to cached scores
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
                    results.append(self._create_abstention_result(gt, (time.time() - start_time) * 1000))
                    continue
                
                # Hop 2
                hop2_nodes = self._process_hop_batch(ultra_model, graph_data, hop1_nodes, 
                                                     rel_types[1], static_threshold, batch_size)
                if not hop2_nodes:
                    results.append(self._create_abstention_result(gt, (time.time() - start_time) * 1000))
                    continue
                
                # Hop 3
                hop3_nodes = self._process_hop_batch(ultra_model, graph_data, hop2_nodes, 
                                                     rel_types[2], static_threshold, batch_size)
                
                execution_time_ms = (time.time() - start_time) * 1000
                precision, recall, f1, _ = self._compute_metrics_from_predictions(hop3_nodes, gt)
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=hop3_nodes,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(hop3_nodes) == 0
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
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=union_nodes,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(union_nodes) == 0
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
                    results.append(self._create_abstention_result(gt, (time.time() - start_time) * 1000))
                    continue
                
                # Project from intersection
                proj_nodes = self._process_hop_batch(ultra_model, graph_data, intersection_nodes, 
                                                     rel3, static_threshold, batch_size)
                
                execution_time_ms = (time.time() - start_time) * 1000
                precision, recall, f1, _ = self._compute_metrics_from_predictions(proj_nodes, gt)
                
                results.append(QueryResult(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    predicted_values=proj_nodes,
                    ground_truth=self._normalize_ground_truth(gt),
                    execution_time_ms=execution_time_ms,
                    is_abstention=len(proj_nodes) == 0
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
    
    def _write_csv_line(self, f, baseline_type: str, threshold: str, stats: Dict[str, float]):
        """Write a single CSV line for baseline results."""
        f.write(f"{baseline_type},{threshold},"
               f"{stats['precision']:.4f},{stats['recall']:.4f},{stats['f1']:.4f},"
               f"{stats['avg_time_ms']:.2f},{stats['num_queries']},{stats['num_abstentions']},"
               f"{stats['abstention_rate']:.4f},{stats['precision_excl_abstentions']:.4f},"
               f"{stats['recall_excl_abstentions']:.4f},{stats['f1_excl_abstentions']:.4f}\n")
    
    def _serialize_result(self, result: QueryResult) -> Dict:
        """Serialize a single QueryResult to dict."""
        return {
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'execution_time_ms': result.execution_time_ms,
            'num_predicted': len(result.predicted_values),
            'num_ground_truth': len(result.ground_truth),
            'is_abstention': result.is_abstention
        }
    
    def _print_stats(self, name: str, stats: Dict[str, float]):
        """Print statistics for a baseline."""
        self.logger.info(f"{name}:")
        self.logger.info(f"  Precision: {stats['precision']:.4f}")
        self.logger.info(f"  Recall:    {stats['recall']:.4f}")
        self.logger.info(f"  F1:        {stats['f1']:.4f}")
        self.logger.info(f"  Avg Time:  {stats['avg_time_ms']:.2f} ms")
        self.logger.info(f"  Queries:   {stats['num_queries']}")
        if stats.get('num_excluded', 0) > 0:
            self.logger.info(f"  Excluded (empty GT): {stats['num_excluded']}")
        if stats.get('num_abstentions', 0) > 0:
            self.logger.info(f"  Abstentions: {stats['num_abstentions']} ({stats['abstention_rate']*100:.2f}%)")
            self.logger.info(f"  Precision (excl. abstentions): {stats['precision_excl_abstentions']:.4f}")
            self.logger.info(f"  Recall (excl. abstentions):    {stats['recall_excl_abstentions']:.4f}")
            self.logger.info(f"  F1 (excl. abstentions):        {stats['f1_excl_abstentions']:.4f}")
    
    def _resolve_load_path(self, args_file: str, load_path: Optional[str], inf_args) -> str:
        """Resolve load_path from provided value, config file, or auto-detect."""
        if load_path:
            return load_path
        
        if hasattr(inf_args, 'load_path') and inf_args.load_path:
            return inf_args.load_path
        
        # Try config file
        with open(args_file, 'r') as f:
            config = json.load(f)
        if 'inference_engine' in config and 'load_path' in config['inference_engine']:
            config_path = config['inference_engine']['load_path']
            # Handle relative paths
            if config_path.startswith('./'):
                parent_path = os.path.join('..', config_path[2:])
                return parent_path if os.path.exists(parent_path) else config_path
            return config_path
        
        # Auto-detect from snapshots
        for search_path in ["./artifacts/snapshots", "../artifacts/snapshots"]:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    if "ultra_args.json" in files:
                        return root
        
        raise FileNotFoundError(
            "Could not find pretrained model. Please specify --load-path argument.\n"
            "Available snapshots can be found in ./artifacts/snapshots/ or ../artifacts/snapshots/"
        )
    
    def compute_summary_stats(self, results: List[QueryResult]) -> Dict[str, float]:
        """Compute summary statistics from results, excluding queries with empty ground truth."""
        # Default empty stats
        empty_stats = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'avg_time_ms': 0.0,
            'num_queries': 0, 'num_excluded': 0, 'num_abstentions': 0,
            'abstention_rate': 0.0, 'precision_excl_abstentions': 0.0,
            'recall_excl_abstentions': 0.0, 'f1_excl_abstentions': 0.0,
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
        
        return {
            'precision': np.mean([r.precision for r in valid_results]),
            'recall': np.mean([r.recall for r in valid_results]),
            'f1': np.mean([r.f1 for r in valid_results]),
            'avg_time_ms': np.mean([r.execution_time_ms for r in valid_results]),
            'num_queries': len(valid_results),
            'num_excluded': num_excluded,
            'num_abstentions': num_abstentions,
            'abstention_rate': abstention_rate,
            'precision_excl_abstentions': precision_excl,
            'recall_excl_abstentions': recall_excl,
            'f1_excl_abstentions': f1_excl,
        }
    
    def save_results(self, output_dir: str, skip_neo4j: bool = False):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save baseline results summary
        baseline_summary_path = os.path.join(output_dir, "baseline_results_summary.csv")
        with open(baseline_summary_path, 'w') as f:
            f.write("baseline,threshold,precision,recall,f1,avg_time_ms,num_queries,num_abstentions,abstention_rate,precision_excl_abstentions,recall_excl_abstentions,f1_excl_abstentions\n")
            
            # Save neo4j results if they exist (regardless of skip_neo4j flag)
            if self._has_neo4j_results():
                neo4j_stats = self.compute_summary_stats(self.results['neo4j_symbolic'])
                self._write_csv_line(f, "symbolic", "N/A", neo4j_stats)
            
            # Save Ultra Neural results only if they exist
            if self._has_ultra_results():
                ultra_stats = self.compute_summary_stats(self.results['ultra_neural'])
                threshold_str = f"{self.ultra_threshold:.2f}" if self.ultra_threshold else "0.75"
                self._write_csv_line(f, "neural", threshold_str, ultra_stats)
        
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
        
        detailed_path = os.path.join(output_dir, "baseline_detailed_results.json")
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
            threshold_str = f"{self.ultra_threshold:.2f}" if self.ultra_threshold else "0.75"
            self._print_stats(f"Ultra Neural (threshold={threshold_str})", ultra_stats)
    
    def run(self, args_file: str, output_dir: str, static_threshold: float = 0.75, 
            load_path: str = None, skip_neo4j: bool = False, 
            compute_scores_only: bool = False, scores_cache: Optional[List[Dict]] = None,
            min_threshold: float = 0.0):
        """
        Run both baselines on benchmark queries.
        
        Args:
            args_file: Path to args JSON file for model setup
            output_dir: Directory to save results
            static_threshold: Threshold for Ultra neural baseline
            load_path: Path to pretrained model directory (overrides config file)
            skip_neo4j: If True, skip Neo4j baseline (only run Ultra)
            compute_scores_only: If True, only compute scores (no threshold application)
            scores_cache: Pre-computed Ultra scores (from previous run)
            min_threshold: Minimum threshold for computing scores (to avoid computing for all nodes)
        """
        # Store threshold for reporting
        self.ultra_threshold = static_threshold
        
        self.logger.info("="*80)
        self.logger.info("BASELINE BENCHMARK RUNNER")
        self.logger.info("="*80)
        self.logger.info(f"Query directory: {self.query_dir}")
        self.logger.info(f"Max queries: {self.max_queries}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Ultra threshold: {static_threshold}")
        self.logger.info("="*80)
        
        # Skip model/graph loading if we have cached scores AND skipping neo4j
        # (only need to apply thresholds, no model inference or neo4j queries needed)
        if scores_cache is not None and skip_neo4j:
            self.logger.info("Using cached scores - skipping model/graph loading")
            ultra_model = None
            graph_data = None
            db_controller = None
            # Still load queries/ground_truths for compatibility (they're fast)
            self.logger.info("Loading benchmark queries...")
            self.logger.info(f"Loading from: {self.query_dir}")
            queries = self.load_queries_from_dir()
            ground_truths = self.load_ground_truth_from_dir()
            min_len = min(len(queries), len(ground_truths))
            queries, ground_truths = queries[:min_len], ground_truths[:min_len]
            self.logger.info(f"Loaded {len(queries)} queries")
        else:
            # Setup models
            self.logger.info("Setting up models...")
            inf_args = merge_args(
                parse_args_inference,
                "args_file",
                ["core", "inference_engine"],
                ["--args_file", args_file]
            )
            
            # Resolve load_path
            inf_args.load_path = self._resolve_load_path(args_file, load_path, inf_args)
            self.logger.info(f"Using load_path: {inf_args.load_path}")
            
            # Setup DB controller and models
            db_controller = Neo4JBackendDBController(
                f"neo4j://{inf_args.neo4j_host}:{inf_args.neo4j_bolt_port}",
                inf_args.node_unique_id,
                inf_args.relation_unique_id,
            )
            inf_args.db_controller = db_controller
            
            self.logger.info("Loading Ultra model...")
            inf_args_ultra = argparse.Namespace(**vars(inf_args))
            inf_args_ultra.model_to_infer = "ULTRA"
            model_factory_ultra = ModelFactory(inf_args_ultra)
            model_factory_ultra.prepare_model()
            ultra_model = model_factory_ultra.model
            
            self.logger.info("Loading graph data...")
            graph_data = get_graph(db_controller, inf_args.device)
            
            # Load queries
            self.logger.info("Loading benchmark queries...")
            self.logger.info(f"Loading from: {self.query_dir}")
            queries = self.load_queries_from_dir()
            ground_truths = self.load_ground_truth_from_dir()
            min_len = min(len(queries), len(ground_truths))
            queries, ground_truths = queries[:min_len], ground_truths[:min_len]
            self.logger.info(f"Loaded {len(queries)} queries")
                
        # Run baselines
        if not skip_neo4j:
                self.logger.info("\n" + "-"*80)
                self.logger.info("Running Neo4j Symbolic Baseline")
                self.logger.info("-"*80)
                neo4j_results = self.run_neo4j_baseline(db_controller, queries, ground_truths)
                self.results['neo4j_symbolic'].extend(neo4j_results)
        else:
            self.logger.info("\nSkipping Neo4j baseline (already run)")
                
                # Run Ultra baseline
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
            # Scores cache kept in memory only (not written to disk)
            
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
        
        # Save results
        self.logger.info("\n" + "="*80)
        self.logger.info("Saving results...")
        self.logger.info("="*80)
        self.save_results(output_dir, skip_neo4j=skip_neo4j)
                
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline benchmarks for comparison with ThreeHopPipeline"
    )
    parser.add_argument("--args_file", type=str, required=True,
                       help="Path to orb config JSON file")
    parser.add_argument("--query-dir", type=str, 
                       default="../queries/test/3p_pipeline",
                       help="Directory containing test queries (default: ../queries/test/3p_pipeline)")
    parser.add_argument("--max-queries", type=int, default=1000,
                       help="Maximum number of queries to process")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--ultra-threshold", type=float, default=0.75,
                       help="Static threshold for Ultra neural baseline")
    parser.add_argument("--ultra-batch-size", type=int, default=4,
                       help="Batch size for Ultra intermediate hops (to avoid GPU OOM, default: 16)")
    parser.add_argument("--load-path", type=str, default=None,
                       help="Path to pretrained model directory (overrides config file)")
    parser.add_argument("--skip-neo4j", action="store_true",
                       help="Skip Neo4j baseline (only run Ultra)")
    parser.add_argument("--compute-scores-only", action="store_true",
                       help="Only compute Ultra scores, don't apply threshold")
    parser.add_argument("--min-threshold", type=float, default=0.0,
                       help="Minimum threshold for computing scores (to avoid computing for all nodes)")
    parser.add_argument("--ultra-thresholds", type=float, nargs='+', default=None,
                       help="Multiple thresholds to apply (if provided, computes scores once and applies all thresholds)")
    
    args = parser.parse_args()
    
    # Create baseline runner
    runner = BaselineRunner(
        config_path=args.args_file,
        query_dir=args.query_dir,
        max_queries=args.max_queries,
        ultra_batch_size=args.ultra_batch_size
    )
    
    # If thresholds provided via --ultra-thresholds, compute scores once and apply all thresholds
    if args.ultra_thresholds and len(args.ultra_thresholds) >= 1:
        # Compute scores once
        result = runner.run(
            args_file=args.args_file,
            output_dir=args.output_dir,
            static_threshold=args.ultra_thresholds[0],  # Dummy, will be overridden
            load_path=args.load_path,
            skip_neo4j=args.skip_neo4j,
            compute_scores_only=True,
            min_threshold=args.min_threshold
        )
        scores_cache = result.get('scores_cache')
        
        # Apply each threshold
        for threshold in args.ultra_thresholds:
            threshold_output_dir = os.path.join(args.output_dir, f"threshold_{threshold}")
            runner.results['ultra_neural'] = []  # Clear previous results
            runner.run(
                args_file=args.args_file,
                output_dir=threshold_output_dir,
                static_threshold=threshold,
                load_path=args.load_path,
                skip_neo4j=True,
                scores_cache=scores_cache,
                min_threshold=args.min_threshold
            )
    else:
        # Single threshold or compute-scores-only mode (using --ultra-threshold)
        runner.run(
            args_file=args.args_file,
            output_dir=args.output_dir,
            static_threshold=args.ultra_threshold,
            load_path=args.load_path,
            skip_neo4j=args.skip_neo4j,
            compute_scores_only=args.compute_scores_only,
            min_threshold=args.min_threshold
        )


if __name__ == "__main__":
    main()

