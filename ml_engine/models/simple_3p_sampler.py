#!/usr/bin/env python3
"""
Script to generate 3-hop traversal (3p) queries.

Query pattern: entity → rel1 → rel2 → rel3 → answers
Example: "Which entities are connected to entity X via relation R1, then R2, then R3?"

Query format: (entity, (rel1, rel2, rel3))
"""

import os
import sys
import logging
import pickle
from argparse import Namespace, ArgumentParser
from collections import defaultdict
import numpy as np

# Add the ml_engine directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph_handler.backend_db.neo4j.controller import Neo4JBackendDBController
from graph_handler.graph_manager.graph_sampler.query_generator import QueryGenerator
from utils import get_graph
import shutil


def calculate_answer_stats(answers_dict):
    """Calculate answer statistics for a dictionary of answers."""
    answer_counts = [len(answers) for answers in answers_dict.values()]
    if not answer_counts:
        return {"total_queries": 0, "total_answers": 0, "answers_per_query": {"min": 0, "max": 0, "mean": 0, "std": 0, "median": 0}}
    
    return {
        "total_queries": len(answers_dict),
        "total_answers": sum(answer_counts),
        "answers_per_query": {
            "min": min(answer_counts),
            "max": max(answer_counts),
            "mean": np.mean(answer_counts),
            "std": np.std(answer_counts),
            "median": np.median(answer_counts)
        }
    }


def build_ent_out_structure(graph_data):
    """Build ent_out structure from graph data once."""
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


def get_intermediate_hop_results(query, ent_out, max_hops):
    """Extract intermediate hop results for a multi-hop query using pre-built ent_out."""
    if ent_out is None:
        return {i: set() for i in range(1, max_hops + 1)}
    
    intermediate_results = {}
    ent_set = set([query[0]])  # Start with the anchor entity
    
    # For each hop, compute the intermediate results
    for hop in range(1, max_hops + 1):
        if hop <= len(query[-1]):  # Only if we have relations for this hop
            ent_set_traverse = set()
            for ent in ent_set:
                if ent in ent_out and query[-1][hop-1] in ent_out[ent]:
                    ent_set_traverse.update(ent_out[ent][query[-1][hop-1]])  # More efficient than union
            ent_set = ent_set_traverse
            intermediate_results[hop] = ent_set.copy()
        else:
            intermediate_results[hop] = set()
    
    return intermediate_results


def create_sample_args(num_queries_per_hop=1000, size_ratio=0.1, min_hops=3, max_hops=3, max_answers=1000):
    """Create sample arguments for the calibration sampler.
    
    Args:
        num_queries_per_hop (int): Number of queries to generate for each hop count
        size_ratio (float): Size ratio multiplier (0.1 = 10% of base calculation)
        min_hops (int): Minimum number of hops for queries
        max_hops (int): Maximum number of hops for queries
        max_answers (int): Maximum number of answers per query
    """
    args = Namespace()
    
    # Database configuration
    args.neo4j_host = "localhost"
    args.neo4j_bolt_port = 7687
    args.neo4j_database = "neo4j"
    args.node_unique_id = "id"
    args.relation_unique_id = "type"
    
    # Query generation parameters
    args.train_min_hops = min_hops
    args.train_max_hops = max_hops
    args.max_num_ans = max_answers
    args.query_generator_log_ratio = 0.3
    
    # Configurable query count - this controls how many queries are generated per hop
    # Create comma-separated string for each hop count
    gen_num_values = [str(num_queries_per_hop)] * (max_hops - min_hops + 1)
    args.gen_num = ",".join(gen_num_values)
    args.size_ratio = size_ratio
    
    return args


def generate_3p_queries(
    graph_data,
    num_queries_per_hop=1000,
    size_ratio=0.1,
    min_hops=3,
    max_hops=3,
    extract_intermediate=True,
    max_answers=5,
    max_hop_size=50,
    save_path="./calibration_data/3p_pipeline"
):
    """
    Generate 3-hop traversal (3p) queries from the graph.
    
    Args:
        graph_data: Graph data object
        num_queries_per_hop: Number of queries to generate per hop count
        size_ratio: Size ratio multiplier (0.1 = 10% of base calculation)
        min_hops: Minimum number of hops for queries
        max_hops: Maximum number of hops for queries
        extract_intermediate: Whether to extract intermediate hop results (can be slow)
        max_answers: Maximum number of answers per query during generation
        max_hop_size: Maximum number of nodes allowed in any hop, including the final hop
        save_path: Path to save the generated queries
    
    Returns:
        Tuple of (queries list, answers dict)
    """
    logging.info("Generating 3-hop traversal (3p) queries...")
    logging.info(f"Configuration: {num_queries_per_hop} queries per hop, size_ratio={size_ratio}, hops={min_hops}-{max_hops}")
    logging.info(f"Max hop size (all hops including final): {max_hop_size} nodes")
    
    # Setup generation parameters
    args = create_sample_args(num_queries_per_hop, size_ratio, min_hops, max_hops, max_answers)
    
    gen_num = (
        {
            int(k): float(v)
            for k, v in zip(
                range(args.train_min_hops, args.train_max_hops + 1),
                args.gen_num.split(","),
            )
        }
        if hasattr(args, "gen_num")
        else {}
    )
    
    min_hops = args.train_min_hops
    max_hops = args.train_max_hops
    max_num_ans = args.max_num_ans
    size_ratio = args.size_ratio
    query_generator_log_ratio = 0.3
    non_overlap_dataset = set()
    
    # Create QueryGenerator
    query_generator = QueryGenerator(
        min_hops=min_hops,
        max_hops=max_hops,
        max_num_ans=max_num_ans,
        gen_num=gen_num,
        size_ratio=1.0,  # Use full size for calibration data
        mode="calib",
        query_generator_log_ratio=query_generator_log_ratio,
        non_overlap_dataset=non_overlap_dataset,
    )
    
    # Generate queries
    # QueryGenerator now filters by max_hop_size during generation, so we just call it once
    # Use a subdirectory of save_path for query generator's temp files
    # (we'll load them into memory and delete the subdirectory)
    if save_path is not None:
        save_path_prefix = os.path.join(save_path, "temp_calibration_data")
    else:
        # If save_path is None, we still need somewhere for query generator's files
        # Use a temp directory that we'll clean up
        import tempfile
        save_path_prefix = tempfile.mkdtemp(prefix="3p_temp_")
    os.makedirs(save_path_prefix, exist_ok=True)
    
    logging.info("Generating 3-hop queries (filtering by max_hop_size during generation)...")
    query_files = query_generator.generate_queries(
        graph_data, save_path_prefix, None
    )

    # Load and process queries (into memory)
    all_queries = []
    all_answers = {}
    hop_data = {}  # Store all hop data: {hop: {"queries": [...], "answers": {...}, "stats": {...}}}
    intermediate_hop_results = {}  # Store intermediate results for each hop
    
    # Process each hop count
    for hop in range(min_hops, max_hops + 1):
        # Load queries and answers from files into memory
        queries_hop = pickle.load(open(query_files[hop][0], "rb"))
        queries_hop = list(queries_hop[list(queries_hop.keys())[0]])
        answers_hop = pickle.load(open(query_files[hop][1], "rb"))
        
        # Store data in memory
        all_queries.extend(queries_hop)
        all_answers.update(answers_hop)
        hop_data[hop] = {
            "queries": queries_hop,
            "answers": answers_hop,
            "stats": calculate_answer_stats(answers_hop)
        }
        
        logging.info(f"Loaded {len(queries_hop)} {hop}-hop queries with {hop_data[hop]['stats']['total_answers']} total answers")
    
    # Now delete the temp files since we have everything in memory
    if save_path is not None:
        temp_path = os.path.join(save_path, "temp_calibration_data")
    else:
        temp_path = save_path_prefix
    if temp_path and os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        logging.info("Cleaned up temporary files")
    
    # Extract intermediate hop results for multi-hop queries (optional, can be slow)
    intermediate_hop_results = {}
    
    if extract_intermediate:
        logging.info("Extracting intermediate hop results for analysis...")
        
        # Build ent_out structure once (this is the expensive part)
        logging.info("Building graph structure...")
        ent_out = build_ent_out_structure(graph_data)
        if ent_out is None:
            logging.warning("Could not build graph structure, skipping intermediate hop extraction")
        else:
            max_hop_queries = hop_data[max_hops]["queries"]
            logging.info(f"Processing {len(max_hop_queries)} queries for intermediate hop extraction...")
            
            # Process queries with progress tracking
            for i, query in enumerate(max_hop_queries):
                if i % 100 == 0:  # Log progress every 100 queries
                    logging.info(f"Processing query {i+1}/{len(max_hop_queries)}")
                
                intermediate_results = get_intermediate_hop_results(query, ent_out, max_hops)
                for hop in range(1, max_hops + 1):
                    if hop not in intermediate_hop_results:
                        intermediate_hop_results[hop] = {}
                    intermediate_hop_results[hop][query] = intermediate_results[hop]
            
            logging.info("Intermediate hop extraction completed!")
            
            # Calculate intermediate hop statistics
            for hop in range(1, max_hops):
                if hop in intermediate_hop_results:
                    stats = calculate_answer_stats(intermediate_hop_results[hop])
                    logging.info(f"Intermediate hop {hop}: {stats['total_queries']} queries, "
                                f"answers per query - min: {stats['answers_per_query']['min']}, "
                                f"max: {stats['answers_per_query']['max']}, "
                                f"mean: {stats['answers_per_query']['mean']:.2f}")
    else:
        logging.info("Skipping intermediate hop extraction (use --extract-intermediate to enable)")
    
    # Save 3-hop data (skip if save_path is None)
    if save_path is not None:
        final_3p_path = save_path
        os.makedirs(final_3p_path, exist_ok=True)
        
        # Save main data files
        with open(os.path.join(final_3p_path, "graph_data.pkl"), "wb") as f:
            pickle.dump(graph_data, f)
        
        # Save only the 3-hop queries
        with open(os.path.join(final_3p_path, "queries.pkl"), "wb") as f:
            pickle.dump(hop_data[max_hops]["queries"], f)
        
        # Save answers
        with open(os.path.join(final_3p_path, "answers.pkl"), "wb") as f:
            pickle.dump(hop_data[max_hops]["answers"], f)
        
        # Save intermediate hop results (including hop_3)
        intermediate_dir = os.path.join(final_3p_path, "intermediate_hops")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        intermediate_hop_stats = {}
        for hop in range(1, max_hops + 1):  # Include all hops (1, 2, 3)
            if hop in intermediate_hop_results:
                hop_dir = os.path.join(intermediate_dir, f"hop_{hop}")
                os.makedirs(hop_dir, exist_ok=True)
                
                intermediate_queries = list(intermediate_hop_results[hop].keys())
                intermediate_answers = intermediate_hop_results[hop]
                
                with open(os.path.join(hop_dir, "queries.pkl"), "wb") as f:
                    pickle.dump(intermediate_queries, f)
                with open(os.path.join(hop_dir, "answers.pkl"), "wb") as f:
                    pickle.dump(intermediate_answers, f)
                
                intermediate_hop_stats[hop] = calculate_answer_stats(intermediate_answers)
                logging.info(f"Saved {len(intermediate_queries)} hop {hop} queries to {hop_dir}")
        
        # Save metadata
        metadata_3p = {
            "query_type": "3p",
            "min_hops": min_hops,
            "max_hops": max_hops,
            "num_queries": len(hop_data[max_hops]["queries"]),  # Only 3-hop queries
            "num_answers": hop_data[max_hops]["stats"]["total_answers"],  # Only 3-hop answers
            "graph_num_nodes": graph_data.num_nodes,
            "graph_num_edges": graph_data.num_edges,
            "hop_metadata": intermediate_hop_stats,  # All hop results in one place
        }
        with open(os.path.join(final_3p_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata_3p, f)
        
        logging.info(f"3-hop calibration data sampling completed!")
        unique_3p_queries = len(set(hop_data[max_hops]['queries']))
        total_3p_queries = len(hop_data[max_hops]['queries'])
        logging.info(f"Unique 3p queries: {unique_3p_queries} (total: {total_3p_queries})")
        logging.info(f"Total answers: {hop_data[max_hops]['stats']['total_answers']}")
        logging.info(f"Data saved to: {final_3p_path}")
        
        # Log summary statistics for all hops
        if intermediate_hop_stats:
            logging.info("Hop results:")
            for hop in sorted(intermediate_hop_stats.keys()):
                stats = intermediate_hop_stats[hop]
                logging.info(f"  Hop {hop}: {stats['total_queries']} queries, {stats['total_answers']} answers "
                            f"(avg: {stats['answers_per_query']['mean']:.1f} per query)")

    return hop_data[max_hops]["queries"], hop_data[max_hops]["answers"], intermediate_hop_results


def sample_3p_calibration_data(
    num_queries_per_hop=1000,
    size_ratio=0.1,
    min_hops=3,
    max_hops=3,
    extract_intermediate=True,
    max_answers=5,
    max_hop_size=50,
    neo4j_host="localhost",
    neo4j_bolt_port=7687,
    neo4j_database="neo4j",
    save_path="./calibration_data/3p_pipeline"
):
    """
    Main function to sample 3p calibration data.
    
    Args:
        num_queries_per_hop: Number of queries to generate per hop count
        size_ratio: Size ratio multiplier
        min_hops: Minimum number of hops for queries
        max_hops: Maximum number of hops for queries
        extract_intermediate: Whether to extract intermediate hop results
        max_answers: Maximum answers per query
        max_hop_size: Maximum nodes in any hop
        neo4j_host: Neo4j host
        neo4j_bolt_port: Neo4j bolt port
        neo4j_database: Neo4j database name
        save_path: Path to save the data
    """
    logging.info("Starting 3p query generation...")
    
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
    queries, answers = generate_3p_queries(
        graph_data,
        num_queries_per_hop=num_queries_per_hop,
        size_ratio=size_ratio,
        min_hops=min_hops,
        max_hops=max_hops,
        extract_intermediate=extract_intermediate,
        max_answers=max_answers,
        max_hop_size=max_hop_size,
        save_path=save_path
    )
    
    logging.info(f"3p query generation completed!")
    logging.info(f"Generated {len(queries)} queries with {sum(len(ans) for ans in answers.values())} total answers")
    logging.info(f"Data saved to: {save_path}")
    
    return save_path


def main():
    """Main function to run the 3p query generation."""
    parser = ArgumentParser(description='Generate 3p (3-hop traversal) queries')
    parser.add_argument('--num-queries', type=int, default=1000,
                        help='Number of queries to generate per hop count (default: 1000)')
    parser.add_argument('--size-ratio', type=float, default=0.1,
                        help='Size ratio multiplier for query generation (default: 0.1)')
    parser.add_argument('--min-hops', type=int, default=3,
                        help='Minimum number of hops for queries (default: 3)')
    parser.add_argument('--max-hops', type=int, default=3,
                        help='Maximum number of hops for queries (default: 3)')
    parser.add_argument('--max-answers', type=int, default=50,
                        help='Maximum number of answers per query during generation (default: 50)')
    parser.add_argument('--max-hop-size', type=int, default=50, dest='max_hop_size',
                        help='Maximum number of nodes allowed in any hop, including the final hop (default: 50)')
    parser.add_argument('--extract-intermediate', action='store_true', default=True,
                        help='Extract intermediate hop results (can be slow for large datasets)')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Skip intermediate hop extraction for faster processing')
    parser.add_argument('--neo4j-host', type=str, default='localhost',
                        help='Neo4j host (default: localhost)')
    parser.add_argument('--neo4j-bolt-port', type=int, default=7687,
                        help='Neo4j bolt port (default: 7687)')
    parser.add_argument('--neo4j-database', type=str, default='neo4j',
                        help='Neo4j database name (default: neo4j)')
    parser.add_argument('--save-path', type=str, default='./calibration_data/3p_pipeline',
                        help='Path to save generated queries (default: ./calibration_data/3p_pipeline)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Handle conflicting arguments
    if args.no_intermediate:
        args.extract_intermediate = False
    
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
        save_path = sample_3p_calibration_data(
            num_queries_per_hop=args.num_queries,
            size_ratio=args.size_ratio,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            extract_intermediate=args.extract_intermediate,
            max_answers=args.max_answers,
            max_hop_size=args.max_hop_size,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            neo4j_database=args.neo4j_database,
            save_path=args.save_path
        )
        print(f"\n3p queries saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error during 3p query generation: {e}")
        raise


if __name__ == "__main__":
    main()

