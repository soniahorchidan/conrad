#!/usr/bin/env python3
"""
Script to generate union-project (up) queries.

Query pattern: 2 projections from different anchors → union → project
Example: "Where did people who work at Company X OR live in City Y graduate from?"
- Anchor1 → rel1 → set1
- Anchor2 → rel2 → set2
- union_nodes = set1 ∪ set2
- Answers = project(union_nodes, rel3)  [= ∪ {ent_out[u][rel3] for u in union_nodes}]

Query format: ((anchor1, rel1), (anchor2, rel2), rel3, "up")
Tensor format: [anchor1, rel1, anchor2, rel2, rel3]  (matches UnionProjectPipeline indices 0-4)
"""

import os
import sys
import logging
import pickle
import random
from collections import defaultdict
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_handler.backend_db.neo4j.controller import Neo4JBackendDBController
from utils import get_graph
import tqdm


def build_ent_out_structure(graph_data):
    from collections import defaultdict
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


def execute_up_query(anchor1, rel1, anchor2, rel2, rel3, ent_out):
    """Execute an up query: union of two projections, then project via rel3."""
    set1 = ent_out[anchor1].get(rel1, set()) if anchor1 in ent_out else set()
    set2 = ent_out[anchor2].get(rel2, set()) if anchor2 in ent_out else set()
    union_nodes = set1 | set2
    answers = set()
    for entity in union_nodes:
        if entity in ent_out and rel3 in ent_out[entity]:
            answers.update(ent_out[entity][rel3])
    return answers


def generate_up_queries(
    graph_data,
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    max_hop_size=None,
    save_path="./calibration_data/up_pipeline",
):
    """Generate up (union-project) queries from the graph."""
    logging.info("Building graph structure...")
    ent_out = build_ent_out_structure(graph_data)
    if ent_out is None:
        raise ValueError("Could not build graph structure")

    # Pre-compute valid (anchor, rel) pairs with non-empty projections
    logging.info("Pre-computing valid anchor-relation pairs...")
    valid_anchor_rel_pairs = []
    for anchor in ent_out:
        for rel in ent_out[anchor]:
            if len(ent_out[anchor][rel]) > 0:
                valid_anchor_rel_pairs.append((anchor, rel))

    if len(valid_anchor_rel_pairs) < 2:
        raise ValueError("Not enough valid anchor-relation pairs found")

    logging.info(f"Found {len(valid_anchor_rel_pairs)} valid anchor-relation pairs")

    queries = []
    answers = {}
    num_sampled = 0
    num_try = 0
    num_empty = 0
    num_too_many = 0

    max_attempts = num_queries * 20
    consecutive_failures = 0
    max_consecutive_failures = 1000

    sample_progress = tqdm.tqdm(total=num_queries, desc="Generating up queries")

    while num_sampled < num_queries and num_try < max_attempts and consecutive_failures < max_consecutive_failures:
        num_try += 1
        sample_progress.set_description(
            f"Generating up queries: {num_sampled}/{num_queries} (tried: {num_try})"
        )

        # Sample two distinct (anchor, rel) pairs for the union branches
        (anchor1, rel1), (anchor2, rel2) = random.sample(valid_anchor_rel_pairs, 2)

        set1 = ent_out[anchor1][rel1]
        set2 = ent_out[anchor2][rel2]

        if max_hop_size is not None:
            if len(set1) > max_hop_size or len(set2) > max_hop_size:
                consecutive_failures += 1
                continue

        union_nodes = set1 | set2

        if not union_nodes:
            consecutive_failures += 1
            continue

        if max_hop_size is not None and len(union_nodes) > max_hop_size:
            consecutive_failures += 1
            continue

        # Sample rel3 from relations available in the union
        available_rel3 = []
        for entity in union_nodes:
            if entity in ent_out:
                available_rel3.extend(ent_out[entity].keys())

        if not available_rel3:
            consecutive_failures += 1
            continue

        rel3 = random.choice(available_rel3)

        # Execute the query
        answer_set = execute_up_query(anchor1, rel1, anchor2, rel2, rel3, ent_out)

        if len(answer_set) == 0:
            num_empty += 1
            consecutive_failures += 1
            continue

        if len(answer_set) > max_answers:
            num_too_many += 1
            consecutive_failures += 1
            continue

        if max_hop_size is not None and len(answer_set) > max_hop_size:
            consecutive_failures += 1
            continue

        if len(answer_set) < min_answers:
            consecutive_failures += 1
            continue

        # Query format: ((anchor1, rel1), (anchor2, rel2), rel3, "up")
        query = ((anchor1, rel1), (anchor2, rel2), rel3, "up")

        if query in answers:
            consecutive_failures += 1
            continue

        queries.append(query)
        answers[query] = answer_set
        num_sampled += 1
        consecutive_failures = 0

        if num_sampled % max(1, int(num_queries * 0.1)) == 0:
            logging.info(f"Generated {num_sampled}/{num_queries} queries (tried: {num_try})")

        sample_progress.update(1)

    sample_progress.close()
    logging.info(
        f"Generated {num_sampled}/{num_queries} queries "
        f"(total attempts: {num_try}, empty: {num_empty}, too_many: {num_too_many})"
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "queries.pkl"), "wb") as f:
            pickle.dump(queries, f)
        with open(os.path.join(save_path, "answers.pkl"), "wb") as f:
            pickle.dump(answers, f)
        logging.info(f"Saved {len(queries)} queries to {save_path}")

    return queries, answers


def sample_up_calibration_data(
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    neo4j_host="localhost",
    neo4j_bolt_port=7687,
    neo4j_database="neo4j",
    save_path="./calibration_data/up_pipeline",
):
    logging.info("Starting up query generation...")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_bolt_port}"
    db_controller = Neo4JBackendDBController(uri=neo4j_uri, nodeUID="id", relUID="type")

    logging.info("Loading graph data from database...")
    graph_data = get_graph(db_controller, device="cpu", augment_inverse_edges=True, relation_graph=True)
    logging.info(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    queries, answers = generate_up_queries(
        graph_data,
        num_queries=num_queries,
        max_answers=max_answers,
        min_answers=min_answers,
        save_path=save_path,
    )

    metadata = {
        "query_type": "up",
        "num_queries": len(queries),
        "num_answers": sum(len(ans) for ans in answers.values()),
        "graph_num_nodes": graph_data.num_nodes,
        "graph_num_edges": graph_data.num_edges,
        "max_answers": max_answers,
        "min_answers": min_answers,
    }
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    logging.info(f"up query generation completed! Generated {len(queries)} queries.")
    return save_path


def main():
    parser = ArgumentParser(description="Generate up (union-project) queries")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--max-answers", type=int, default=1000)
    parser.add_argument("--min-answers", type=int, default=1)
    parser.add_argument("--neo4j-host", type=str, default="localhost")
    parser.add_argument("--neo4j-bolt-port", type=int, default=7687)
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument("--save-path", type=str, default="./calibration_data/up_pipeline")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger("neo4j").setLevel(logging.WARNING)
        logging.getLogger("neo4j.io").setLevel(logging.WARNING)
        logging.getLogger("neo4j.pool").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        save_path = sample_up_calibration_data(
            num_queries=args.num_queries,
            max_answers=args.max_answers,
            min_answers=args.min_answers,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            neo4j_database=args.neo4j_database,
            save_path=args.save_path,
        )
        print(f"\nup queries saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error during up query generation: {e}")
        raise


if __name__ == "__main__":
    main()
