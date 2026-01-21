#!/usr/bin/env python3
"""
Simple script to sample calibration data from Neo4j database.
This version avoids complex model imports and focuses just on data sampling.

OUTPUT STRUCTURE:
The script creates a directory structure with:
- Shared data: graph_data.pkl at the base calibration_data/ directory (shared by all query types)
- Query-specific data: Each query type (3p_pipeline/, 2u_pipeline/, etc.) has:
  - queries.pkl: Generated queries
  - answers.pkl: Ground truth answers
  - intermediate_hops/hop_X/: Directory for each hop (1, 2, 3) with queries.pkl and answers.pkl
  - metadata.pkl: Detailed statistics including answer counts per query per hop

Each hop's metadata includes:
- Total queries and answers for that hop
- Answer distribution statistics (min, max, mean, std, median answers per query)

Hop results:
- For each multi-hop query (e.g., 3-hop), extracts results at each step (1-hop, 2-hop, 3-hop)
- Shows how answer sets evolve as you traverse the graph step by step
- All hop results are stored in intermediate_hops/hop_X/ directories
- Useful for analyzing query complexity and intermediate reasoning steps

CONFIGURATION OPTIONS:
The number of queries generated is controlled by several parameters:

1. --num-queries: Number of queries to generate per hop count (default: 1000)
   - This directly controls the gen_num parameter passed to QueryGenerator
   - Higher values = more queries but longer generation time

2. --size-ratio: Size ratio multiplier (default: 0.1)
   - This is a multiplier applied to the base query count calculation
   - 0.1 = 10% of the calculated base number
   - 1.0 = 100% of the calculated base number
   - Lower values = fewer queries, faster generation

3. --min-hops/--max-hops: Hop range for queries (default: 3-3)
   - Controls the complexity of generated queries
   - More hops = more complex queries but fewer valid ones

4. --max-hop-size: Maximum nodes in any hop, including final answers (default: 50)
   - Controls both intermediate hop sizes and final answer sizes
   - Filters out queries where any hop (1 to max_hops, including the final hop) exceeds this size
   - Also used as max_num_ans parameter for QueryGenerator to filter queries during generation
   - Lower values = stricter filtering, fewer but simpler queries

USAGE EXAMPLES:
# Generate 500 queries per hop with 5% size ratio
python calibration_sampler.py --num-queries 500 --size-ratio 0.05

# Generate 2000 queries per hop with 20% size ratio  
python calibration_sampler.py --num-queries 2000 --size-ratio 0.2

# Generate 2-hop and 3-hop queries
python calibration_sampler.py --min-hops 2 --max-hops 3

# Filter queries with intermediate hops and final answers having more than 30 nodes
python calibration_sampler.py --max-hop-size 30

# Verbose output for debugging
python calibration_sampler.py --verbose

# Generate 3-hop queries AND 2ip queries
python calibration_sampler.py --generate-2ip --num-2ip-queries 1000

# Generate only 2ip queries (use sampler_2ip.py directly)
python sampler_2ip.py --num-queries 1000 --max-answers 1000
"""

import os
import sys
import logging
import pickle
from argparse import Namespace, ArgumentParser

# Add the src directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_handler.backend_db.neo4j.controller import Neo4JBackendDBController
from utils import get_graph
import time

# Import query generators
from .sampler_3p import generate_3p_queries, build_ent_out_structure
from .sampler_2ip import generate_2ip_queries
from .sampler_2u import generate_2u_queries
import random


def format_query_string(query, query_type):
    """
    Format a query as a string for benchmark format.
    
    Args:
        query: Query tuple
        query_type: Type of query ("3p", "2ip", or "2u")
    
    Returns:
        Formatted query string
    """
    if query_type == "3p":
        # Format: (entity, (rel1, rel2, rel3))
        entity_id, rel_types = query
        return f"query(({entity_id}, ({', '.join(map(str, rel_types))})))"
    elif query_type == "2ip":
        # Format: ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
        (anchor1, rel1), (anchor2, rel2), rel3, _ = query
        return f"query((({anchor1}, {rel1}), ({anchor2}, {rel2}), {rel3}, '2ip'))"
    elif query_type == "2u":
        # Format: ((anchor1, rel1), (anchor2, rel2), "2u")
        (anchor1, rel1), (anchor2, rel2), _ = query
        return f"query((({anchor1}, {rel1}), ({anchor2}, {rel2}), '2u'))"
    else:
        raise ValueError(f"Unknown query type: {query_type}")


def split_and_save_queries(
    queries,
    answers,
    query_type,
    calib_split,
    calib_path,
    test_path,
    intermediate_hop_results=None,
):
    """
    Split queries and save calibration and test sets.
    
    Args:
        queries: List of queries
        answers: Dict mapping queries to answer sets
        query_type: Type of query ("3p", "2ip", or "2u")
        calib_split: Fraction for calibration (0.5 = 50% calibration, 50% test)
        calib_path: Path to save calibration queries
        test_path: Path to save test queries
        intermediate_hop_results: Optional dict of intermediate hop results (for 3p queries)
    """
    if not queries:
        return
    
    # Shuffle queries for random split
    shuffled_queries = queries.copy()
    random.shuffle(shuffled_queries)
    
    # Split queries
    n_calib = int(len(shuffled_queries) * calib_split)
    calib_queries = shuffled_queries[:n_calib]
    test_queries = shuffled_queries[n_calib:]
    
    logging.info("="*60)
    logging.info(f"Splitting {query_type} queries:")
    logging.info(f"  Calibration: {len(calib_queries)} queries ({calib_split*100:.0f}%)")
    logging.info(f"  Test:        {len(test_queries)} queries ({(1-calib_split)*100:.0f}%)")
    logging.info("="*60)
    
    # Save calibration queries and answers
    os.makedirs(calib_path, exist_ok=True)
    with open(os.path.join(calib_path, "queries.pkl"), "wb") as f:
        pickle.dump(calib_queries, f)
    
    calib_answers = {q: answers[q] for q in calib_queries if q in answers}
    with open(os.path.join(calib_path, "answers.pkl"), "wb") as f:
        pickle.dump(calib_answers, f)
    
    # Save intermediate hop results if provided
    # Format: intermediate_hops/hop_X/{queries.pkl,answers.pkl}
    if intermediate_hop_results is not None:
        intermediate_dir = os.path.join(calib_path, "intermediate_hops")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        for hop in sorted(intermediate_hop_results.keys()):
            hop_dir = os.path.join(intermediate_dir, f"hop_{hop}")
            os.makedirs(hop_dir, exist_ok=True)
            
            # Filter to only calibration queries
            hop_answers_map = intermediate_hop_results.get(hop, {}) or {}
            calib_hop_queries = [q for q in calib_queries if q in hop_answers_map]
            calib_hop_answers = {q: hop_answers_map[q] for q in calib_hop_queries}
            
            with open(os.path.join(hop_dir, "queries.pkl"), "wb") as f:
                pickle.dump(calib_hop_queries, f)
            with open(os.path.join(hop_dir, "answers.pkl"), "wb") as f:
                pickle.dump(calib_hop_answers, f)
    
    # Save test queries in benchmark format (only if test_path is provided)
    if test_path is not None:
        os.makedirs(test_path, exist_ok=True)
        queries_file = os.path.join(test_path, "queries")
        with open(queries_file, "w") as f:
            for query in test_queries:
                query_str = format_query_string(query, query_type)
                f.write(query_str + "\n")
        
        # Save test ground truth
        gt_file = os.path.join(test_path, "gt")
        with open(gt_file, "w") as f:
            for query in test_queries:
                if query in answers:
                    gt_nodes = sorted(list(answers[query]))
                    f.write(" ".join(map(str, gt_nodes)) + "\n")
                else:
                    f.write("\n")  # Empty GT
        
        logging.info(f"  Calibration data saved to: {calib_path}")
        logging.info(f"  Test data saved to: {test_path}")
    else:
        logging.info(f"  Calibration data saved to: {calib_path}")


def sample_calibration_data(num_queries_per_hop=1000, size_ratio=0.1, min_hops=3, max_hops=3, extract_intermediate=True, max_hop_size=50, generate_3hop=True, generate_2ip=False, generate_2u=False, num_2ip_queries=1000, num_2u_queries=1000, calib_path="./calibration_data", calib_split=None, test_path=None):
    """
    Sample calibration data from the database and save to disk.
    
    Args:
        num_queries_per_hop (int): Number of queries to generate for each hop count
        size_ratio (float): Size ratio multiplier (0.1 = 10% of base calculation)
        min_hops (int): Minimum number of hops for queries
        max_hops (int): Maximum number of hops for queries
        extract_intermediate (bool): Whether to extract intermediate hop results (can be slow)
        max_hop_size (int): Maximum number of nodes allowed in any hop, including the final hop (default: 50)
        generate_3hop (bool): Whether to generate 3p (3-hop traversal) queries
        generate_2ip (bool): Whether to generate 2ip (Intersection-followed-by-Projection) queries
        generate_2u (bool): Whether to generate 2u (Union) queries
        num_2ip_queries (int): Number of 2ip queries to generate if generate_2ip is True
        num_2u_queries (int): Number of 2u queries to generate if generate_2u is True
        calib_path (str): Base path to save calibration data (default: "./calibration_data")
        calib_split (float): Fraction for calibration split (e.g., 0.5 = 50% calibration, 50% test). 
                             If None, all queries go to calibration data.
        test_path (str): Output directory for test queries (required if calib_split is specified)
    """
    if not generate_3hop and not generate_2ip and not generate_2u:
        raise ValueError("At least one of --generate-3p, --generate-2ip, or --generate-2u must be enabled")
    
    if calib_split is not None and test_path is None:
        raise ValueError("--test-path must be specified when --calib-split is provided")
    
    if calib_split is not None and (calib_split <= 0 or calib_split >= 1):
        raise ValueError("--calib-split must be between 0 and 1 (exclusive)")
    
    logging.info("Starting calibration data sampling...")
    if generate_3hop:
        logging.info(f"Will generate {num_queries_per_hop} 3p queries")
    if generate_2ip:
        logging.info(f"Will generate {num_2ip_queries} 2ip queries")
    if generate_2u:
        logging.info(f"Will generate {num_2u_queries} 2u queries")
    if max_hop_size is not None:
        logging.info(f"Max intermediate set size: {max_hop_size} nodes (applies to all query types)")
    
    logging.info("="*60)
    
    # Setup
    device = "cpu"  # or "cuda" if available
    
    # Initialize database controller
    neo4j_uri = f"bolt://localhost:7687"
    db_controller = Neo4JBackendDBController(
        uri=neo4j_uri,
        nodeUID="id",
        relUID="type"
    )
    
    # Get graph data
    logging.info("Loading graph data from database...")
    graph_data = get_graph(
        db_controller, 
        device, 
        augment_inverse_edges=True, 
        relation_graph=True
    )
    logging.info(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    logging.info("="*60)
    
    # Generate 3p queries if requested
    if generate_3hop:
        logging.info("Generating 3p (3-hop traversal) queries...")
        try:
            final_3p_path = os.path.join(calib_path, "3p_pipeline")
            
            # Generate queries without saving (pass None to skip save)
            queries_3p, answers_3p, intermediate_hop_results = generate_3p_queries(
                graph_data,
                num_queries_per_hop=num_queries_per_hop,
                size_ratio=size_ratio,
                min_hops=min_hops,
                max_hops=max_hops,
                extract_intermediate=extract_intermediate,
                max_answers=max_hop_size,  # Use max_hop_size for final answers too
                max_hop_size=max_hop_size,
                save_path=None  # Skip saving, we'll save after split
            )
            unique_3p_queries = len(set(queries_3p))
            total_3p_queries = len(queries_3p)
            logging.info(f"Generated {total_3p_queries} 3p queries (unique: {unique_3p_queries})")
            
            # Always split and save (even if calib_split is None, we'll save all to calib)
            if calib_split is not None and test_path is not None:
                calib_3p_path = final_3p_path
                test_3p_path = os.path.join(test_path, "3p_pipeline")
                split_and_save_queries(queries_3p, answers_3p, "3p", calib_split, calib_3p_path, test_3p_path, intermediate_hop_results)
            else:
                # No split requested, save all to calibration path
                split_and_save_queries(queries_3p, answers_3p, "3p", 1.0, final_3p_path, None, intermediate_hop_results)
        except Exception as e:
            logging.warning(f"Failed to generate 3p queries: {e}")
            logging.warning("Continuing without 3p queries...")
    
    logging.info("="*60)
    
    # Generate 2ip queries if requested
    if generate_2ip:
        logging.info("Generating 2ip (Intersection-followed-by-Projection) queries...")
        try:
            final_2ip_path = os.path.join(calib_path, "2ip_pipeline")
            
            # Generate queries without saving (pass None to skip save)
            queries_2ip, answers_2ip = generate_2ip_queries(
                graph_data,
                num_queries=num_2ip_queries,
                max_answers=max_hop_size,  # Use max_hop_size for final answers too
                min_answers=1,
                max_hop_size=max_hop_size,
                save_path=None  # Skip saving, we'll save after split
            )
            unique_2ip_queries = len(set(queries_2ip))
            total_2ip_queries = len(queries_2ip)
            logging.info(f"Generated {total_2ip_queries} 2ip queries (unique: {unique_2ip_queries})")
            
            # Build intermediate hop results for 2ip if requested.
            # 2ip has 3 graph traversals (projections):
            # - hop_1: first branch projection (anchor1 --rel1--> set1)
            # - hop_2: second branch projection (anchor2 --rel2--> set2)
            # - intersection: set1 ∩ set2 (computed on-demand, not stored)
            # - hop_3: third projection from intersection (intersection --rel3--> final answers)
            # Final answers are also stored in answers.pkl
            intermediate_hop_results_2ip = None
            if extract_intermediate:
                logging.info("Extracting 2ip intermediate results for analysis...")
                ent_out = build_ent_out_structure(graph_data)
                if ent_out is None:
                    logging.warning("Could not build graph structure, skipping 2ip intermediate extraction")
                else:
                    intermediate_hop_results_2ip = {1: {}, 2: {}, 3: {}}
                    for q in queries_2ip:
                        (anchor1, rel1), (anchor2, rel2), rel3, _ = q
                        set1 = ent_out.get(anchor1, {}).get(rel1, set()).copy()
                        set2 = ent_out.get(anchor2, {}).get(rel2, set()).copy()
                        intersection = set1 & set2
                        # hop_3: third projection from intersection
                        set3 = set()
                        for entity in intersection:
                            if entity in ent_out and rel3 in ent_out[entity]:
                                set3.update(ent_out[entity][rel3])
                        intermediate_hop_results_2ip[1][q] = set1
                        intermediate_hop_results_2ip[2][q] = set2
                        intermediate_hop_results_2ip[3][q] = set3

            # Always split and save
            if calib_split is not None and test_path is not None:
                calib_2ip_path = final_2ip_path
                test_2ip_path = os.path.join(test_path, "2ip_pipeline")
                split_and_save_queries(
                    queries_2ip,
                    answers_2ip,
                    "2ip",
                    calib_split,
                    calib_2ip_path,
                    test_2ip_path,
                    intermediate_hop_results_2ip,
                )
            else:
                # No split requested, save all to calibration path
                split_and_save_queries(
                    queries_2ip,
                    answers_2ip,
                    "2ip",
                    1.0,
                    final_2ip_path,
                    None,
                    intermediate_hop_results_2ip,
                )
            
            # Save metadata for 2ip queries
            metadata_2ip = {
                "query_type": "2ip",
                "num_queries": len(queries_2ip),
                "num_answers": sum(len(ans) for ans in answers_2ip.values()),
                "graph_num_nodes": graph_data.num_nodes,
                "graph_num_edges": graph_data.num_edges,
                "max_hop_size": max_hop_size,
                "min_answers": 1,
            }
            
            metadata_path = os.path.join(final_2ip_path, "metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata_2ip, f)
            
            logging.info(f"2ip queries generation completed!")
            logging.info(f"Generated {len(queries_2ip)} queries with {metadata_2ip['num_answers']} total answers")
            logging.info(f"Data saved to: {final_2ip_path}")
        except Exception as e:
            logging.warning(f"Failed to generate 2ip queries: {e}")
            logging.warning("Continuing without 2ip queries...")
    
    logging.info("="*60)
    
    # Generate 2u queries if requested
    if generate_2u:
        logging.info("Generating 2u (Union) queries...")
        try:
            final_2u_path = os.path.join(calib_path, "2u_pipeline")
            
            # Generate queries without saving (pass None to skip save)
            queries_2u, answers_2u = generate_2u_queries(
                graph_data,
                num_queries=num_2u_queries,
                max_answers=max_hop_size,  # Use max_hop_size for final answers too
                min_answers=1,
                max_hop_size=max_hop_size,
                save_path=None  # Skip saving, we'll save after split
            )
            unique_2u_queries = len(set(queries_2u))
            total_2u_queries = len(queries_2u)
            logging.info(f"Generated {total_2u_queries} 2u queries (unique: {unique_2u_queries})")
            
            # Build intermediate hop results for 2u if requested.
            # We store:
            # - hop_1: first branch projection (anchor1 --rel1--> set1)
            # - hop_2: second branch projection (anchor2 --rel2--> set2)
            # Final union (set1 ∪ set2) is stored in answers.pkl
            # Note: 2u has num_components=2, so only 2 hops are saved (no hop_3)
            intermediate_hop_results_2u = None
            if extract_intermediate:
                logging.info("Extracting 2u intermediate results for analysis...")
                ent_out = build_ent_out_structure(graph_data)
                if ent_out is None:
                    logging.warning("Could not build graph structure, skipping 2u intermediate extraction")
                else:
                    intermediate_hop_results_2u = {1: {}, 2: {}}
                    for q in queries_2u:
                        (anchor1, rel1), (anchor2, rel2), _ = q
                        set1 = ent_out.get(anchor1, {}).get(rel1, set()).copy()
                        set2 = ent_out.get(anchor2, {}).get(rel2, set()).copy()
                        intermediate_hop_results_2u[1][q] = set1
                        intermediate_hop_results_2u[2][q] = set2

            # Always split and save
            if calib_split is not None and test_path is not None:
                calib_2u_path = final_2u_path
                test_2u_path = os.path.join(test_path, "2u_pipeline")
                split_and_save_queries(
                    queries_2u,
                    answers_2u,
                    "2u",
                    calib_split,
                    calib_2u_path,
                    test_2u_path,
                    intermediate_hop_results_2u,
                )
            else:
                # No split requested, save all to calibration path
                split_and_save_queries(
                    queries_2u,
                    answers_2u,
                    "2u",
                    1.0,
                    final_2u_path,
                    None,
                    intermediate_hop_results_2u,
                )
            
            # Save metadata for 2u queries
            metadata_2u = {
                "query_type": "2u",
                "num_queries": len(queries_2u),
                "num_answers": sum(len(ans) for ans in answers_2u.values()),
                "graph_num_nodes": graph_data.num_nodes,
                "graph_num_edges": graph_data.num_edges,
                "max_hop_size": max_hop_size,
                "min_answers": 1,
            }
            
            metadata_path = os.path.join(final_2u_path, "metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata_2u, f)
            
            logging.info(f"2u query generation completed!")
            logging.info(f"Generated {len(queries_2u)} queries with {metadata_2u['num_answers']} total answers")
            logging.info(f"Data saved to: {final_2u_path}")
        except Exception as e:
            logging.warning(f"Failed to generate 2u queries: {e}")
            logging.warning("Continuing without 2u queries...")
    
    logging.info("="*60)
    
    # Save graph_data once at the shared location (reused by all query types)
    shared_graph_path = os.path.join(calib_path, "graph_data.pkl")
    if not os.path.exists(shared_graph_path):
        logging.info(f"Saving shared graph_data to {shared_graph_path}...")
        os.makedirs(calib_path, exist_ok=True)
        with open(shared_graph_path, "wb") as f:
            pickle.dump(graph_data, f)
        logging.info(f"Shared graph_data saved ({graph_data.num_nodes} nodes, {graph_data.num_edges} edges)")
    else:
        logging.info(f"Shared graph_data already exists at {shared_graph_path}, skipping save")
    
    # Return the appropriate path(s)
    paths = {}
    if generate_3hop:
        paths["3p_path"] = os.path.join(calib_path, "3p_pipeline")
    if generate_2ip:
        paths["2ip_path"] = os.path.join(calib_path, "2ip_pipeline")
    if generate_2u:
        paths["2u_path"] = os.path.join(calib_path, "2u_pipeline")
    
    if len(paths) == 1:
        return list(paths.values())[0]
    else:
        return paths


def main():
    """Main function to run the calibration sampling."""
    parser = ArgumentParser(description='Sample calibration data from Neo4j database')
    parser.add_argument('--num-queries', type=int, default=1000,
                        help='Number of queries to generate per hop count (default: 1000, min:100)')
    parser.add_argument('--size-ratio', type=float, default=0.1,
                        help='Size ratio multiplier for query generation (default: 0.1)')
    parser.add_argument('--min-hops', type=int, default=3,
                        help='Minimum number of hops for queries (default: 3)')
    parser.add_argument('--max-hops', type=int, default=3,
                        help='Maximum number of hops for queries (default: 3)')
    parser.add_argument('--max-hop-size', type=int, default=50, dest='max_hop_size',
                        help='Maximum number of nodes allowed in any hop, including the final hop (default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--extract-intermediate', action='store_true', default=True,
                        help='Extract intermediate hop results (can be slow for large datasets)')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Skip intermediate hop extraction for faster processing')
    parser.add_argument('--generate-3p', '-3p', action='store_true', default=False,
                        help='Generate 3p (3-hop traversal) queries')
    parser.add_argument('--generate-2ip', '-2ip', action='store_true', default=False,
                        help='Generate 2ip (Intersection-followed-by-Projection) queries')
    parser.add_argument('--generate-2u', '-2u', action='store_true', default=False,
                        help='Generate 2u (Union) queries')
    parser.add_argument('--num-2ip-queries', type=int, default=1000,
                        help='Number of 2ip queries to generate (default: 1000, only used if --generate-2ip is set)')
    parser.add_argument('--num-2u-queries', type=int, default=1000,
                        help='Number of 2u queries to generate (default: 1000, only used if --generate-2u is set)')
    parser.add_argument('--calib-path', type=str, default='./calibration_data',
                        help='Base path to save calibration data (default: ./calibration_data)')
    parser.add_argument('--calib-split', type=float, default=None,
                        help='Fraction for calibration split (e.g., 0.5 = 50%% calibration, 50%% test). If not specified, all queries go to calibration data.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Output directory for test queries (required if --calib-split is specified)')
    
    args = parser.parse_args()
    
    # Handle conflicting arguments
    if args.no_intermediate:
        args.extract_intermediate = False
    
    # If no query type specified, default to 3p for backward compatibility
    if not args.generate_3p and not args.generate_2ip and not args.generate_2u:
        logging.warning("No query type specified, defaulting to 3p queries for backward compatibility")
        args.generate_3p = True
    
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
        calibration_data_path = sample_calibration_data(
            num_queries_per_hop=args.num_queries,
            size_ratio=args.size_ratio,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            extract_intermediate=args.extract_intermediate,
            max_hop_size=args.max_hop_size,
            generate_3hop=args.generate_3p,
            generate_2ip=args.generate_2ip,
            generate_2u=args.generate_2u,
            num_2ip_queries=args.num_2ip_queries,
            num_2u_queries=args.num_2u_queries,
            calib_path=args.calib_path,
            calib_split=args.calib_split,
            test_path=args.test_path
        )
        
        if isinstance(calibration_data_path, dict):
            for key, path in calibration_data_path.items():
                print(f"{key} queries saved to: {path}")
        else:
            print(f"\nCalibration data saved to: {calibration_data_path}")
    except Exception as e:
        logging.error(f"Error during calibration sampling: {e}")
        raise


if __name__ == "__main__":
    main()
