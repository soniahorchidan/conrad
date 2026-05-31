#!/usr/bin/env python3
"""
Script to generate 3-way intersection (3i) queries.

Query pattern: 3 projections from different anchors → intersection
Example: "Which entities are reachable from X via R1 AND from Y via R2 AND from Z via R3?"
- Anchor1 → rel1 → set1
- Anchor2 → rel2 → set2
- Anchor3 → rel3 → set3
- Answers = set1 ∩ set2 ∩ set3

Query format: ((anchor1, rel1), (anchor2, rel2), (anchor3, rel3), "3i")
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


def generate_3i_queries(
    graph_data,
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    max_hop_size=None,
    save_path="./calibration_data/3i_pipeline",
):
    """Generate 3i queries from the graph."""
    logging.info("Building graph structure...")
    ent_out = build_ent_out_structure(graph_data)
    if ent_out is None:
        raise ValueError("Could not build graph structure")

    # Build reverse index: entity → [(anchor, rel), ...]
    logging.info("Building reverse index for intersection finding...")
    entity_to_anchors = defaultdict(list)
    for anchor in ent_out:
        for rel in ent_out[anchor]:
            for target_entity in ent_out[anchor][rel]:
                entity_to_anchors[target_entity].append((anchor, rel))

    # Need at least 3 distinct (anchor, rel) pairs to form a 3-way intersection
    intersection_candidates = {
        entity: list(set(pairs))
        for entity, pairs in entity_to_anchors.items()
        if len(set(pairs)) >= 3
    }

    if not intersection_candidates:
        raise ValueError("No entities found with ≥3 incoming (anchor, rel) pairs. Graph may be too sparse.")

    logging.info(f"Found {len(intersection_candidates)} potential 3-way intersection entities")

    queries = []
    answers = {}
    num_sampled = 0
    num_try = 0
    num_empty = 0
    num_too_many = 0

    max_attempts = num_queries * 30
    consecutive_failures = 0
    max_consecutive_failures = 1000

    sample_progress = tqdm.tqdm(total=num_queries, desc="Generating 3i queries")

    candidate_list = list(intersection_candidates.keys())

    while num_sampled < num_queries and num_try < max_attempts and consecutive_failures < max_consecutive_failures:
        num_try += 1
        sample_progress.set_description(
            f"Generating 3i queries: {num_sampled}/{num_queries} (tried: {num_try})"
        )

        intersection_entity = random.choice(candidate_list)
        unique_pairs = intersection_candidates[intersection_entity]

        if len(unique_pairs) < 3:
            consecutive_failures += 1
            continue

        (anchor1, rel1), (anchor2, rel2), (anchor3, rel3) = random.sample(unique_pairs, 3)

        set1 = ent_out[anchor1].get(rel1, set())
        set2 = ent_out[anchor2].get(rel2, set())
        set3 = ent_out[anchor3].get(rel3, set())
        intersection = set1 & set2 & set3

        if intersection_entity not in intersection or len(intersection) == 0:
            num_empty += 1
            consecutive_failures += 1
            continue

        if max_hop_size is not None:
            if (len(set1) > max_hop_size or len(set2) > max_hop_size
                    or len(set3) > max_hop_size or len(intersection) > max_hop_size):
                consecutive_failures += 1
                continue

        if len(intersection) > max_answers:
            num_too_many += 1
            consecutive_failures += 1
            continue

        if len(intersection) < min_answers:
            consecutive_failures += 1
            continue

        query = ((anchor1, rel1), (anchor2, rel2), (anchor3, rel3), "3i")

        if query in answers:
            consecutive_failures += 1
            continue

        queries.append(query)
        answers[query] = intersection
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


def sample_3i_calibration_data(
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    neo4j_host="localhost",
    neo4j_bolt_port=7687,
    neo4j_database="neo4j",
    save_path="./calibration_data/3i_pipeline",
):
    logging.info("Starting 3i query generation...")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_bolt_port}"
    db_controller = Neo4JBackendDBController(uri=neo4j_uri, nodeUID="id", relUID="type")

    logging.info("Loading graph data from database...")
    graph_data = get_graph(db_controller, device="cpu", augment_inverse_edges=True, relation_graph=True)
    logging.info(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    queries, answers = generate_3i_queries(
        graph_data,
        num_queries=num_queries,
        max_answers=max_answers,
        min_answers=min_answers,
        save_path=save_path,
    )

    metadata = {
        "query_type": "3i",
        "num_queries": len(queries),
        "num_answers": sum(len(ans) for ans in answers.values()),
        "graph_num_nodes": graph_data.num_nodes,
        "graph_num_edges": graph_data.num_edges,
        "max_answers": max_answers,
        "min_answers": min_answers,
    }
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    logging.info(f"3i query generation completed! Generated {len(queries)} queries.")
    return save_path


def main():
    parser = ArgumentParser(description="Generate 3i (3-way intersection) queries")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--max-answers", type=int, default=1000)
    parser.add_argument("--min-answers", type=int, default=1)
    parser.add_argument("--neo4j-host", type=str, default="localhost")
    parser.add_argument("--neo4j-bolt-port", type=int, default=7687)
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument("--save-path", type=str, default="./calibration_data/3i_pipeline")
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
        save_path = sample_3i_calibration_data(
            num_queries=args.num_queries,
            max_answers=args.max_answers,
            min_answers=args.min_answers,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            neo4j_database=args.neo4j_database,
            save_path=args.save_path,
        )
        print(f"\n3i queries saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error during 3i query generation: {e}")
        raise


if __name__ == "__main__":
    main()
