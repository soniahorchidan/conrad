import torch
import torch.nn as nn
from ..common import ModelUtils
from argparse import Namespace
import logging
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import json
import random

class DBExecModel(nn.Module, ModelUtils):

    def __init__(self, args: Namespace, num_relations: int, device: str):
        super(DBExecModel, self).__init__()
        self.args = Namespace()
        for key, value in vars(args).items():
            setattr(self.args, key, value)
        self.device = device
        self.backend_controller = self.args.db_controller
        self.non_overlap_set = set()
        self.ONE_HOP_TEMPLATE = (
            "MATCH (a:Entity {id: %s})-[f1:Relation {type: %s}]->(r:Entity) "
            "RETURN DISTINCT r.id;"
        )
        self.TWO_HOP_TEMPLATE = (
            "MATCH (a:Entity {id: %s})-[f1:Relation {type: %s}]->(b:Entity)"
            "-[f2:Relation {type: %s}]->(r:Entity) RETURN DISTINCT r.id;"
        )
        self.THREE_HOP_TEMPLATE = (
            "MATCH (a:Entity {id: %s})-[f1:Relation {type: %s}]->(b:Entity)"
            "-[f2:Relation {type: %s}]->(c:Entity)-[f3:Relation {type: %s}]->(r:Entity) "
            "RETURN DISTINCT r.id;"
        )

        # TODO(sonia): dont hardcode num edges and nodes
        self.node_degree_popularity = {
            str(i): {"popularity": random.uniform(0.5, 1.0)} 
            for i in range(1, 14542)
        }

        self.edge_popularity_map = {
            str(i): {"popularity": random.uniform(0.5, 1.0)} 
            for i in range(1, 1001)
        }

        # Tiny epsilon fallback when popularity stats are missing
        self._pop_epsilon = 1e-2

    def _compute_scores(self, results, rel_type):
        """
        Compute confidence scores for query results based on edge and node popularity.
        
        Args:
            results: List of node IDs from query results
            rel_type: The relation type (edge type) - can be tensor or string
            
        Returns:
            Dictionary mapping node_id -> confidence score
        """
        # Get edge popularity with fallback
        edge_popularity = self.edge_popularity_map.get(str(rel_type), {}).get("popularity")
        if not edge_popularity:
            edge_popularity = self._pop_epsilon
        
        # Compute scores for each result node
        query_scores = {}
        for r in results:
            node_pop = self.node_degree_popularity.get(str(r), {}).get("popularity")
            if not node_pop:
                node_pop = self._pop_epsilon
            confidence = 0.5 * edge_popularity + 0.5 * node_pop
            query_scores[r] = confidence
        return query_scores

    @staticmethod
    def preprocess(general_args: Namespace, args: Namespace):
        logging.info(
            "Skipping preprocess for DBExecModel."
        )
    
    def postprocess(model, general_args: Namespace, args: Namespace):
        pass


    def predict(
        self, query
    ):
        results = []
        scores = []
        
        for q in query:
            q = q.tolist()
            if len(q) == 2:
                cypher = self.ONE_HOP_TEMPLATE % (q[0], q[1])
            elif len(q) == 3:
                cypher = self.TWO_HOP_TEMPLATE % (q[0], q[1], q[2])
            elif len(q) == 4:
                cypher = self.THREE_HOP_TEMPLATE % (q[0], q[1], q[2], q[3])
            res = self.backend_controller.execute_query(cypher)
            res = [record["r.id"] for record in res]
            results.append(res)
            
            if len(res) > 0:
                scores.append(self._compute_scores(res, q[1]))
            else:
                scores.append({})
                
        return results, scores

    def predict_batched(self, query):
        """
        Optimized batch version that groups similar queries together for more efficient Neo4j calls.
        """
        results = []
        scores = []
        
        # Group queries by hop count and relation type for batching
        query_groups = {}
        query_indices = {}
        
        for i, q in enumerate(query):
            q = q.tolist()
            hop_count = len(q) - 1
            rel_type = q[1] if len(q) > 1 else None
            group_key = (hop_count, rel_type)
            
            if group_key not in query_groups:
                query_groups[group_key] = []
                query_indices[group_key] = []
            
            query_groups[group_key].append(q)
            query_indices[group_key].append(i)
        
        # Initialize results and scores arrays
        for _ in range(len(query)):
            results.append([])
            scores.append({})
        
        # Process each group
        for (hop_count, rel_type), group_queries in query_groups.items():
            indices = query_indices[(hop_count, rel_type)]
            
            if hop_count == 1:
                # Batch 1-hop queries with same relation type
                self._process_one_hop_batch(group_queries, indices, results, scores)
            elif hop_count == 2:
                # Batch 2-hop queries with same relation types
                self._process_two_hop_batch(group_queries, indices, results, scores)
            elif hop_count == 3:
                # Batch 3-hop queries with same relation types
                self._process_three_hop_batch(group_queries, indices, results, scores)
            else:
                # Fallback to individual processing
                for q, idx in zip(group_queries, indices):
                    if hop_count == 1:
                        cypher = self.ONE_HOP_TEMPLATE % (q[0], q[1])
                    elif hop_count == 2:
                        cypher = self.TWO_HOP_TEMPLATE % (q[0], q[1], q[2])
                    elif hop_count == 3:
                        cypher = self.THREE_HOP_TEMPLATE % (q[0], q[1], q[2], q[3])
                    
                    res = self.backend_controller.execute_query(cypher)
                    res = [record["r.id"] for record in res]
                    results[idx] = res
                    
                    if len(res) > 0:
                        scores[idx] = self._compute_scores(res, rel_type)
                    else:
                        scores[idx] = {}
        
        return results, scores

    def _process_one_hop_batch(self, queries, indices, results, scores):
        """Process a batch of 1-hop queries with the same relation type."""
        if not queries:
            return
            
        rel_type = queries[0][1]
        
        # Create a single batched query using UNWIND
        node_ids = [q[0] for q in queries]
        cypher = f"""
        UNWIND $node_ids AS node_id
        MATCH (a:Entity {{id: node_id}})-[f1:Relation {{type: {rel_type}}}]->(r:Entity)
        RETURN node_id, r.id
        """
        
        res = self.backend_controller.execute_query(cypher, node_ids=node_ids)
        
        # Group results by node_id
        grouped_results = {}
        for record in res:
            node_id = record["node_id"]
            if node_id not in grouped_results:
                grouped_results[node_id] = []
            grouped_results[node_id].append(record["r.id"])
        
        # Assign results to each query
        for q, idx in zip(queries, indices):
            node_id = q[0]
            query_results = grouped_results.get(node_id, [])
            results[idx] = query_results
            
            if len(query_results) > 0:
                scores[idx] = self._compute_scores(query_results, rel_type)
            else:
                scores[idx] = {}

    def _process_two_hop_batch(self, queries, indices, results, scores):
        """Process a batch of 2-hop queries with the same relation types."""
        if not queries:
            return
            
        rel1_type = queries[0][1]
        rel2_type = queries[0][2]
        
        # Create a single batched query using UNWIND
        node_ids = [q[0] for q in queries]
        cypher = f"""
        UNWIND $node_ids AS node_id
        MATCH (a:Entity {{id: node_id}})-[f1:Relation {{type: {rel1_type}}}]->(b:Entity)
        -[f2:Relation {{type: {rel2_type}}}]->(r:Entity)
        RETURN node_id, r.id
        """
        
        res = self.backend_controller.execute_query(cypher, node_ids=node_ids)
        
        # Group results by node_id
        grouped_results = {}
        for record in res:
            node_id = record["node_id"]
            if node_id not in grouped_results:
                grouped_results[node_id] = []
            grouped_results[node_id].append(record["r.id"])
        
        # Assign results to each query
        for q, idx in zip(queries, indices):
            node_id = q[0]
            query_results = grouped_results.get(node_id, [])
            results[idx] = query_results
            
            if len(query_results) > 0:
                scores[idx] = self._compute_scores(query_results, rel1_type)
            else:
                scores[idx] = {}

    def _process_three_hop_batch(self, queries, indices, results, scores):
        """Process a batch of 3-hop queries with the same relation types."""
        if not queries:
            return
            
        rel1_type = queries[0][1]
        rel2_type = queries[0][2]
        rel3_type = queries[0][3]
        
        # Create a single batched query using UNWIND
        node_ids = [q[0] for q in queries]
        cypher = f"""
        UNWIND $node_ids AS node_id
        MATCH (a:Entity {{id: node_id}})-[f1:Relation {{type: {rel1_type}}}]->(b:Entity)
        -[f2:Relation {{type: {rel2_type}}}]->(c:Entity)-[f3:Relation {{type: {rel3_type}}}]->(r:Entity)
        RETURN node_id, r.id
        """
        
        res = self.backend_controller.execute_query(cypher, node_ids=node_ids)
        
        # Group results by node_id
        grouped_results = {}
        for record in res:
            node_id = record["node_id"]
            if node_id not in grouped_results:
                grouped_results[node_id] = []
            grouped_results[node_id].append(record["r.id"])
        
        # Assign results to each query
        for q, idx in zip(queries, indices):
            node_id = q[0]
            query_results = grouped_results.get(node_id, [])
            results[idx] = query_results
            
            if len(query_results) > 0:
                scores[idx] = self._compute_scores(query_results, rel1_type)
            else:
                scores[idx] = {}


    def forward(self, graph_data, query):
        pass
    
    def generateCalibrateSamples(
        self, queries, answers, batch_size=32
    ):
        self.eval()
        scores = []
        hop1_scores = []
        hop2_scores = []
        hop3_scores = []
        
        # Process queries in batches
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
            batch_queries = queries[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]
            
            # Process each query to get intermediate hop results
            for query in batch_queries:
                q = query.tolist()
                num_entities = 14541  # Should match the graph size
                
                # Initialize score tensors for each hop
                prob_hop1 = torch.zeros(num_entities)
                prob_hop2 = torch.zeros(num_entities)
                prob_hop3 = torch.zeros(num_entities)
                
                # Hop 1: Execute first relation
                if len(q) >= 2:
                    cypher_hop1 = self.ONE_HOP_TEMPLATE % (q[0], q[1])
                    res_hop1 = self.backend_controller.execute_query(cypher_hop1)
                    res_hop1 = [record["r.id"] for record in res_hop1]
                    if len(res_hop1) > 0:
                        query_scores_hop1 = self._compute_scores(res_hop1, q[1])
                        for node_id, score in query_scores_hop1.items():
                            prob_hop1[node_id] = score
                
                # Hop 2: Execute first two relations
                if len(q) >= 3:
                    cypher_hop2 = self.TWO_HOP_TEMPLATE % (q[0], q[1], q[2])
                    res_hop2 = self.backend_controller.execute_query(cypher_hop2)
                    res_hop2 = [record["r.id"] for record in res_hop2]
                    if len(res_hop2) > 0:
                        query_scores_hop2 = self._compute_scores(res_hop2, q[2])
                        for node_id, score in query_scores_hop2.items():
                            prob_hop2[node_id] = score
                
                # Hop 3: Execute all three relations (final results)
                if len(q) >= 4:
                    cypher_hop3 = self.THREE_HOP_TEMPLATE % (q[0], q[1], q[2], q[3])
                    res_hop3 = self.backend_controller.execute_query(cypher_hop3)
                    res_hop3 = [record["r.id"] for record in res_hop3]
                    if len(res_hop3) > 0:
                        query_scores_hop3 = self._compute_scores(res_hop3, q[3])
                        for node_id, score in query_scores_hop3.items():
                            prob_hop3[node_id] = score
                
                # Store scores for each hop
                hop1_scores.append(prob_hop1)
                hop2_scores.append(prob_hop2)
                hop3_scores.append(prob_hop3)
                scores.append(prob_hop3)  # Final scores (hop3) for backward compatibility

        final_scores = torch.stack(scores)
        hop1_tensor = torch.stack(hop1_scores)
        hop2_tensor = torch.stack(hop2_scores)
        hop3_tensor = torch.stack(hop3_scores)
        
        return final_scores, hop1_tensor, hop2_tensor, hop3_tensor 

    def load_all_components(self, load_path, ignore_components: list = []):
        pass

    def trainModel(self, general_args, args):
        pass