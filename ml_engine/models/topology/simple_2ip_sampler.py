#!/usr/bin/env python3
"""
Script to generate Intersection-followed-by-Projection (2ip) queries.

Query pattern: 2 projections from different anchors → intersection → 1 projection
Example: "Where did Canadian citizens with a Turing Award graduate?"
- Anchor1 entity → Project along R1 (e.g., has_nationality → Canadian) → set S1
- Anchor2 entity → Project along R2 (e.g., has_award → Turing Award) → set S2  
- Intersect S1 ∩ S2
- Project from intersection along R3 (e.g., graduated_from) → final answers

Query format: ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
"""

import os
import sys
import logging
import pickle
import random
from collections import defaultdict
from argparse import ArgumentParser

# Add the ml_engine directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph_handler.backend_db.neo4j.controller import Neo4JBackendDBController
from utils import get_graph
import tqdm


def build_ent_out_structure(graph_data):
    """Build ent_out structure from graph data."""
    from collections import defaultdict
    
    if hasattr(graph_data, 'ent_out'):
        return graph_data.ent_out
    
    ent_out = defaultdict(lambda: defaultdict(set))
    if hasattr(graph_data, 'edge_index') and hasattr(graph_data, 'edge_type'):
        for e1, rel, e2 in zip(graph_data.edge_index[0], graph_data.edge_type, graph_data.edge_index[1]):
            ent_out[e1.item()][rel.item()].add(e2.item())
    else:
        logging.warning("Could not extract ent_out from graph_data")
        return None
    
    return ent_out


def execute_2ip_query(anchor1, rel1, anchor2, rel2, rel3, ent_out):
    """
    Execute a 2ip query: project R1 from anchor1, project R2 from anchor2, intersect, then project R3.
    
    Args:
        anchor1: First starting entity ID
        rel1: First projection relation type
        anchor2: Second starting entity ID
        rel2: Second projection relation type  
        rel3: Final projection relation type
        ent_out: Graph structure (entity -> relation -> set of entities)
    
    Returns:
        Set of answer entities
    """
    # Project along R1 from anchor1
    set1 = set()
    if anchor1 in ent_out and rel1 in ent_out[anchor1]:
        set1 = ent_out[anchor1][rel1].copy()
    
    # Project along R2 from anchor2
    set2 = set()
    if anchor2 in ent_out and rel2 in ent_out[anchor2]:
        set2 = ent_out[anchor2][rel2].copy()
    
    # Intersect the two sets
    intersection = set1 & set2
    
    # Project from intersection along R3
    answers = set()
    for entity in intersection:
        if entity in ent_out and rel3 in ent_out[entity]:
            answers.update(ent_out[entity][rel3])
    
    return answers


def generate_2ip_queries(
    graph_data,
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    max_hop_size=None,
    save_path="./calibration_data/2ip_queries"
):
    """
    Generate 2ip queries from the graph.
    
    Args:
        graph_data: Graph data object
        num_queries: Number of queries to generate
        max_answers: Maximum number of answers per query
        min_answers: Minimum number of answers per query
        max_hop_size: Maximum number of nodes allowed in any intermediate set (set1, set2, or intersection). 
                      If None, no filtering by intermediate set size.
        save_path: Path to save the generated queries
    
    Returns:
        Tuple of (queries list, answers dict)
    """
    logging.info("Building graph structure...")
    ent_out = build_ent_out_structure(graph_data)
    if ent_out is None:
        raise ValueError("Could not build graph structure")
    
    # Build reverse index: for each entity, find all (anchor, relation) pairs that lead to it
    # This helps us find entities that can serve as intersections
    logging.info("Building reverse index for intersection finding...")
    entity_to_anchors = defaultdict(list)  # entity -> [(anchor, rel), ...]
    for anchor in ent_out:
        for rel in ent_out[anchor]:
            for target_entity in ent_out[anchor][rel]:
                entity_to_anchors[target_entity].append((anchor, rel))
    
    # Find entities that have at least 2 different (anchor, relation) pairs
    # These are potential intersection points
    intersection_candidates = {
        entity: pairs for entity, pairs in entity_to_anchors.items()
        if len(pairs) >= 2
    }
    
    if not intersection_candidates:
        raise ValueError("No entities found that can serve as intersections. Graph may be too sparse.")
    
    logging.info(f"Found {len(intersection_candidates)} entities that can serve as intersections")
    
    # Collect all relation types
    all_relations = set()
    for entity in ent_out:
        all_relations.update(ent_out[entity].keys())
    all_relations = list(all_relations)
    
    if len(all_relations) < 3:
        raise ValueError(f"Need at least 3 relation types, found {len(all_relations)}")
    
    logging.info(f"Found {len(ent_out)} entities and {len(all_relations)} relation types")
    
    queries = []
    answers = {}
    num_sampled = 0
    num_try = 0
    num_empty = 0
    num_too_many_answers = 0
    num_no_intersection = 0
    
    max_attempts = num_queries * 20  # Allow many attempts
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    sample_progress = tqdm.tqdm(
        total=num_queries,
        desc=f"Generating 2ip queries: {num_sampled}/{num_queries}"
    )
    
    while num_sampled < num_queries and num_try < max_attempts and consecutive_failures < max_consecutive_failures:
        num_try += 1
        sample_progress.set_description(
            f"Generating 2ip queries: {num_sampled}/{num_queries} (tried: {num_try}, failures: {consecutive_failures})"
        )
        
        # Sample an intersection entity (one that has at least 2 incoming paths)
        intersection_entity = random.choice(list(intersection_candidates.keys()))
        anchor_rel_pairs = intersection_candidates[intersection_entity]
        
        # Deduplicate pairs to ensure we sample from distinct (anchor, rel) combinations
        unique_pairs = list(set(anchor_rel_pairs))
        
        # Sample two different (anchor, relation) pairs that lead to this intersection
        if len(unique_pairs) < 2:
            consecutive_failures += 1
            continue
        
        (anchor1, rel1), (anchor2, rel2) = random.sample(unique_pairs, 2)
        
        # Verify the intersection (should always be non-empty now)
        set1 = ent_out[anchor1].get(rel1, set())
        set2 = ent_out[anchor2].get(rel2, set())
        intersection = set1 & set2
        
        # Double-check intersection contains our target entity
        if intersection_entity not in intersection or len(intersection) == 0:
            consecutive_failures += 1
            continue
        
        # Check intermediate set sizes if max_hop_size is specified
        if max_hop_size is not None:
            if len(set1) > max_hop_size or len(set2) > max_hop_size or len(intersection) > max_hop_size:
                consecutive_failures += 1
                continue
        
        # Sample a third relation for the final projection
        # Filter to relations that exist from the intersection entities
        available_rel3 = []
        for entity in intersection:
            if entity in ent_out:
                available_rel3.extend(ent_out[entity].keys())
        
        if not available_rel3:
            consecutive_failures += 1
            continue
        
        rel3 = random.choice(available_rel3)
        
        # Execute the query
        answer_set = execute_2ip_query(anchor1, rel1, anchor2, rel2, rel3, ent_out)
        
        # Check validity
        if len(answer_set) == 0:
            num_empty += 1
            consecutive_failures += 1
            continue
        
        # Check final answer set size against both max_answers and max_hop_size
        if len(answer_set) > max_answers:
            num_too_many_answers += 1
            consecutive_failures += 1
            continue
        
        # Also check final answer set against max_hop_size (if specified)
        if max_hop_size is not None and len(answer_set) > max_hop_size:
            num_too_many_answers += 1
            consecutive_failures += 1
            continue
        
        if len(answer_set) < min_answers:
            consecutive_failures += 1
            continue
        
        # Create query tuple: ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
        query = ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
        
        # Check for duplicates
        if query in answers:
            consecutive_failures += 1
            continue
        
        # Successfully generated a valid query
        queries.append(query)
        answers[query] = answer_set
        num_sampled += 1
        consecutive_failures = 0
        
        if num_sampled % max(1, int(num_queries * 0.1)) == 0:
            logging.info(
                f"Generated {num_sampled}/{num_queries} queries "
                f"(tried: {num_try}, empty: {num_empty}, too_many: {num_too_many_answers}, "
                f"no_intersection: {num_no_intersection})"
            )
        
        sample_progress.update(1)
    
    sample_progress.close()
    
    logging.info(
        f"Generated {num_sampled}/{num_queries} queries "
        f"(total attempts: {num_try}, empty: {num_empty}, "
        f"too_many_answers: {num_too_many_answers}, no_intersection: {num_no_intersection})"
    )
    
    # Save queries and answers (skip if save_path is None)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        queries_path = os.path.join(save_path, "queries.pkl")
        answers_path = os.path.join(save_path, "answers.pkl")
        
        with open(queries_path, "wb") as f:
            pickle.dump(queries, f)
        
        with open(answers_path, "wb") as f:
            pickle.dump(answers, f)
        
        logging.info(f"Saved {len(queries)} queries to {save_path}")
    
    return queries, answers


def sample_2ip_calibration_data(
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    neo4j_host="localhost",
    neo4j_bolt_port=7687,
    neo4j_database="neo4j",
    save_path="./calibration_data/2ip_queries"
):
    """
    Main function to sample 2ip calibration data.
    
    Args:
        num_queries: Number of queries to generate
        max_answers: Maximum answers per query
        min_answers: Minimum answers per query
        neo4j_host: Neo4j host
        neo4j_bolt_port: Neo4j bolt port
        neo4j_database: Neo4j database name
        save_path: Path to save the data
    """
    logging.info("Starting 2ip query generation...")
    
    # Initialize database controller
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_bolt_port}"
    db_controller = Neo4JBackendDBController(
        uri=neo4j_uri,
        nodeUID="id",
        relUID="type"
    )
    
    # Get graph data
    logging.info("Loading graph data from database...")
    graph_data = get_graph(
        db_controller,
        device="cpu",
        augment_inverse_edges=True,
        relation_graph=True
    )
    logging.info(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Generate queries
    queries, answers = generate_2ip_queries(
        graph_data,
        num_queries=num_queries,
        max_answers=max_answers,
        min_answers=min_answers,
        save_path=save_path
    )
    
    # Save metadata
    metadata = {
        "query_type": "2ip",
        "num_queries": len(queries),
        "num_answers": sum(len(ans) for ans in answers.values()),
        "graph_num_nodes": graph_data.num_nodes,
        "graph_num_edges": graph_data.num_edges,
        "max_answers": max_answers,
        "min_answers": min_answers,
    }
    
    metadata_path = os.path.join(save_path, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    logging.info(f"2ip query generation completed!")
    logging.info(f"Generated {len(queries)} queries with {metadata['num_answers']} total answers")
    logging.info(f"Data saved to: {save_path}")
    
    return save_path


def main():
    """Main function to run the 2ip query generation."""
    parser = ArgumentParser(description='Generate 2ip (Intersection-followed-by-Projection) queries')
    parser.add_argument('--num-queries', type=int, default=1000,
                        help='Number of queries to generate (default: 1000)')
    parser.add_argument('--max-answers', type=int, default=1000,
                        help='Maximum number of answers per query (default: 1000)')
    parser.add_argument('--min-answers', type=int, default=1,
                        help='Minimum number of answers per query (default: 1)')
    parser.add_argument('--neo4j-host', type=str, default='localhost',
                        help='Neo4j host (default: localhost)')
    parser.add_argument('--neo4j-bolt-port', type=int, default=7687,
                        help='Neo4j bolt port (default: 7687)')
    parser.add_argument('--neo4j-database', type=str, default='neo4j',
                        help='Neo4j database name (default: neo4j)')
    parser.add_argument('--save-path', type=str, default='./calibration_data/2ip_queries',
                        help='Path to save generated queries (default: ./calibration_data/2ip_queries)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if not args.verbose:
        logging.getLogger('neo4j').setLevel(logging.WARNING)
        logging.getLogger('neo4j.io').setLevel(logging.WARNING)
        logging.getLogger('neo4j.pool').setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        save_path = sample_2ip_calibration_data(
            num_queries=args.num_queries,
            max_answers=args.max_answers,
            min_answers=args.min_answers,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            neo4j_database=args.neo4j_database,
            save_path=args.save_path
        )
        print(f"\n2ip queries saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error during 2ip query generation: {e}")
        raise


if __name__ == "__main__":
    main()

