#!/usr/bin/env python3
"""
Script to generate 2-hop traversal (2p) queries.

Query pattern: entity → rel1 → rel2 → answers
Example: "Which entities are connected to entity X via relation R1, then R2?"

Query format: (entity, (rel1, rel2))
"""

import os
import sys
import logging
import pickle
from argparse import Namespace, ArgumentParser
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_handler.backend_db.neo4j.controller import Neo4JBackendDBController
from graph_handler.graph_manager.graph_sampler.query_generator import QueryGenerator
from utils import get_graph
import shutil


def calculate_answer_stats(answers_dict):
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
            "median": np.median(answer_counts),
        },
    }


def build_ent_out_structure(graph_data):
    if hasattr(graph_data, "ent_out"):
        return graph_data.ent_out
    ent_out = defaultdict(lambda: defaultdict(set))
    if hasattr(graph_data, "edge_index") and hasattr(graph_data, "edge_type"):
        for e1, rel, e2 in zip(graph_data.edge_index[0], graph_data.edge_type, graph_data.edge_index[1]):
            ent_out[e1.item()][rel.item()].add(e2.item())
    else:
        logging.warning("Could not extract ent_out from graph_data")
        return None
    return ent_out


def get_intermediate_hop_results(query, ent_out, max_hops):
    if ent_out is None:
        return {i: set() for i in range(1, max_hops + 1)}
    intermediate_results = {}
    ent_set = {query[0]}
    for hop in range(1, max_hops + 1):
        if hop <= len(query[-1]):
            ent_set_traverse = set()
            for ent in ent_set:
                if ent in ent_out and query[-1][hop - 1] in ent_out[ent]:
                    ent_set_traverse.update(ent_out[ent][query[-1][hop - 1]])
            ent_set = ent_set_traverse
            intermediate_results[hop] = ent_set.copy()
        else:
            intermediate_results[hop] = set()
    return intermediate_results


def generate_2p_queries(
    graph_data,
    num_queries_per_hop=1000,
    size_ratio=0.1,
    extract_intermediate=True,
    max_answers=5,
    max_hop_size=50,
    save_path="./calibration_data/2p_pipeline",
):
    """Generate 2-hop traversal (2p) queries from the graph."""
    min_hops = max_hops = 2
    logging.info("Generating 2-hop traversal (2p) queries...")

    args = Namespace()
    args.train_min_hops = min_hops
    args.train_max_hops = max_hops
    args.max_num_ans = max_answers
    args.query_generator_log_ratio = 0.3
    args.gen_num = str(num_queries_per_hop)
    args.size_ratio = size_ratio

    gen_num = {2: float(num_queries_per_hop)}
    query_generator = QueryGenerator(
        min_hops=min_hops,
        max_hops=max_hops,
        max_num_ans=max_answers,
        gen_num=gen_num,
        size_ratio=1.0,
        mode="calib",
        query_generator_log_ratio=0.3,
        non_overlap_dataset=set(),
    )

    if save_path is not None:
        save_path_prefix = os.path.join(save_path, "temp_calibration_data")
    else:
        import tempfile
        save_path_prefix = tempfile.mkdtemp(prefix="2p_temp_")
    os.makedirs(save_path_prefix, exist_ok=True)

    query_files = query_generator.generate_queries(graph_data, save_path_prefix, None)

    queries_hop = pickle.load(open(query_files[2][0], "rb"))
    queries_hop = list(queries_hop[list(queries_hop.keys())[0]])
    answers_hop = pickle.load(open(query_files[2][1], "rb"))

    temp_path = save_path_prefix
    if temp_path and os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        logging.info("Cleaned up temporary files")

    all_queries = queries_hop
    all_answers = answers_hop
    stats = calculate_answer_stats(answers_hop)
    logging.info(f"Loaded {len(queries_hop)} 2-hop queries with {stats['total_answers']} total answers")

    intermediate_hop_results = {}
    if extract_intermediate:
        logging.info("Extracting intermediate hop results...")
        ent_out = build_ent_out_structure(graph_data)
        if ent_out is None:
            logging.warning("Could not build graph structure, skipping intermediate hop extraction")
        else:
            for i, query in enumerate(all_queries):
                if i % 100 == 0:
                    logging.info(f"Processing query {i+1}/{len(all_queries)}")
                results = get_intermediate_hop_results(query, ent_out, max_hops)
                for hop in range(1, max_hops + 1):
                    if hop not in intermediate_hop_results:
                        intermediate_hop_results[hop] = {}
                    intermediate_hop_results[hop][query] = results[hop]

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "queries.pkl"), "wb") as f:
            pickle.dump(all_queries, f)
        with open(os.path.join(save_path, "answers.pkl"), "wb") as f:
            pickle.dump(all_answers, f)

        intermediate_dir = os.path.join(save_path, "intermediate_hops")
        os.makedirs(intermediate_dir, exist_ok=True)

        hop_stats = {}
        for hop in range(1, max_hops + 1):
            if hop in intermediate_hop_results:
                hop_dir = os.path.join(intermediate_dir, f"hop_{hop}")
                os.makedirs(hop_dir, exist_ok=True)
                with open(os.path.join(hop_dir, "queries.pkl"), "wb") as f:
                    pickle.dump(list(intermediate_hop_results[hop].keys()), f)
                with open(os.path.join(hop_dir, "answers.pkl"), "wb") as f:
                    pickle.dump(intermediate_hop_results[hop], f)
                hop_stats[hop] = calculate_answer_stats(intermediate_hop_results[hop])

        metadata = {
            "query_type": "2p",
            "min_hops": min_hops,
            "max_hops": max_hops,
            "num_queries": len(all_queries),
            "num_answers": stats["total_answers"],
            "graph_num_nodes": graph_data.num_nodes,
            "graph_num_edges": graph_data.num_edges,
            "hop_metadata": hop_stats,
        }
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        logging.info(f"2p calibration data saved to: {save_path}")

    return all_queries, all_answers, intermediate_hop_results


def sample_2p_calibration_data(
    num_queries_per_hop=1000,
    size_ratio=0.1,
    extract_intermediate=True,
    max_answers=5,
    max_hop_size=50,
    neo4j_host="localhost",
    neo4j_bolt_port=7687,
    neo4j_database="neo4j",
    save_path="./calibration_data/2p_pipeline",
):
    logging.info("Starting 2p query generation...")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_bolt_port}"
    db_controller = Neo4JBackendDBController(uri=neo4j_uri, nodeUID="id", relUID="type")

    logging.info("Loading graph data from database...")
    graph_data = get_graph(db_controller, device="cpu", augment_inverse_edges=True, relation_graph=True)
    logging.info(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    queries, answers, _ = generate_2p_queries(
        graph_data,
        num_queries_per_hop=num_queries_per_hop,
        size_ratio=size_ratio,
        extract_intermediate=extract_intermediate,
        max_answers=max_answers,
        max_hop_size=max_hop_size,
        save_path=save_path,
    )

    logging.info(f"2p query generation completed! Generated {len(queries)} queries.")
    return save_path


def main():
    parser = ArgumentParser(description="Generate 2p (2-hop traversal) queries")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--size-ratio", type=float, default=0.1)
    parser.add_argument("--max-answers", type=int, default=50)
    parser.add_argument("--max-hop-size", type=int, default=50, dest="max_hop_size")
    parser.add_argument("--extract-intermediate", action="store_true", default=True)
    parser.add_argument("--no-intermediate", action="store_true")
    parser.add_argument("--neo4j-host", type=str, default="localhost")
    parser.add_argument("--neo4j-bolt-port", type=int, default=7687)
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument("--save-path", type=str, default="./calibration_data/2p_pipeline")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.no_intermediate:
        args.extract_intermediate = False

    if not args.verbose:
        logging.getLogger("neo4j").setLevel(logging.WARNING)
        logging.getLogger("neo4j.io").setLevel(logging.WARNING)
        logging.getLogger("neo4j.pool").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        save_path = sample_2p_calibration_data(
            num_queries_per_hop=args.num_queries,
            size_ratio=args.size_ratio,
            extract_intermediate=args.extract_intermediate,
            max_answers=args.max_answers,
            max_hop_size=args.max_hop_size,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            neo4j_database=args.neo4j_database,
            save_path=args.save_path,
        )
        print(f"\n2p queries saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error during 2p query generation: {e}")
        raise


if __name__ == "__main__":
    main()
