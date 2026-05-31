#!/usr/bin/env python3
"""
Script to generate project-intersect (pi) queries.

Query pattern: 2-hop chain intersected with 1-hop branch
- Chain: anchor1 → rel1 → intermediates → rel2 → chain_result
- Branch: anchor2 → rel3 → branch_result
- Answers = chain_result ∩ branch_result

Query format: ((anchor1, rel1, rel2), (anchor2, rel3), "pi")
Tensor format: [anchor1, rel1, rel2, anchor2, rel3]  (matches ProjectIntersectPipeline indices 0-4)
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


def execute_chain(anchor1, rel1, rel2, ent_out):
    """Execute a 2-hop chain: anchor1 → rel1 → intermediates → rel2 → result."""
    intermediates = ent_out[anchor1].get(rel1, set()) if anchor1 in ent_out else set()
    result = set()
    for intermediate in intermediates:
        if intermediate in ent_out and rel2 in ent_out[intermediate]:
            result.update(ent_out[intermediate][rel2])
    return intermediates, result


def generate_pi_queries(
    graph_data,
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    max_hop_size=None,
    save_path="./calibration_data/pi_pipeline",
):
    """Generate pi (project-intersect) queries from the graph."""
    logging.info("Building graph structure...")
    ent_out = build_ent_out_structure(graph_data)
    if ent_out is None:
        raise ValueError("Could not build graph structure")

    # Pre-compute valid (anchor, rel) pairs for the 1p branch
    logging.info("Pre-computing valid anchor-relation pairs for 1p branch...")
    valid_1p_pairs = []
    for anchor in ent_out:
        for rel in ent_out[anchor]:
            if len(ent_out[anchor][rel]) > 0:
                valid_1p_pairs.append((anchor, rel))

    if not valid_1p_pairs:
        raise ValueError("No valid anchor-relation pairs found")

    # Build reverse index for finding chain answer candidates
    # We need chain_result ∩ branch_result to be non-empty.
    # Strategy: sample a chain (anchor1, rel1, rel2), compute chain_result,
    # then sample anchor2 and rel3 such that branch_result overlaps with chain_result.
    logging.info("Building reverse index for chain output entities...")
    entity_to_1p_anchors = defaultdict(list)
    for anchor in ent_out:
        for rel in ent_out[anchor]:
            for target in ent_out[anchor][rel]:
                entity_to_1p_anchors[target].append((anchor, rel))

    # Pre-compute entities with at least 2 outgoing relation types (needed for chain hop1)
    entities_with_two_rels = [e for e in ent_out if len(ent_out[e]) >= 1]
    all_entities = list(ent_out.keys())

    logging.info(f"Found {len(all_entities)} entities with outgoing edges")

    queries = []
    answers = {}
    num_sampled = 0
    num_try = 0
    num_empty = 0
    num_too_many = 0
    num_no_overlap = 0

    max_attempts = num_queries * 30
    consecutive_failures = 0
    max_consecutive_failures = 1000

    sample_progress = tqdm.tqdm(total=num_queries, desc="Generating pi queries")

    while num_sampled < num_queries and num_try < max_attempts and consecutive_failures < max_consecutive_failures:
        num_try += 1
        sample_progress.set_description(
            f"Generating pi queries: {num_sampled}/{num_queries} (tried: {num_try})"
        )

        # Sample chain: anchor1 → rel1 → rel2
        anchor1 = random.choice(all_entities)
        if anchor1 not in ent_out or len(ent_out[anchor1]) == 0:
            consecutive_failures += 1
            continue

        rel1 = random.choice(list(ent_out[anchor1].keys()))
        intermediates = ent_out[anchor1].get(rel1, set())
        if not intermediates:
            consecutive_failures += 1
            continue

        if max_hop_size is not None and len(intermediates) > max_hop_size:
            consecutive_failures += 1
            continue

        # Sample rel2 from the outgoing relations of at least one intermediate
        available_rel2 = []
        for inter in intermediates:
            if inter in ent_out:
                available_rel2.extend(ent_out[inter].keys())
        if not available_rel2:
            consecutive_failures += 1
            continue

        rel2 = random.choice(available_rel2)

        # Execute the chain
        intermediates_set, chain_result = execute_chain(anchor1, rel1, rel2, ent_out)

        if not chain_result:
            num_empty += 1
            consecutive_failures += 1
            continue

        if max_hop_size is not None and len(chain_result) > max_hop_size:
            consecutive_failures += 1
            continue

        # Find a 1p branch whose result overlaps with chain_result
        # Sample from entities in chain_result that have incoming 1p pairs
        overlap_anchors = []
        for entity in chain_result:
            if entity in entity_to_1p_anchors:
                overlap_anchors.extend(entity_to_1p_anchors[entity])

        if not overlap_anchors:
            num_no_overlap += 1
            consecutive_failures += 1
            continue

        anchor2, rel3 = random.choice(overlap_anchors)

        # Compute 1p branch result
        branch_result = ent_out[anchor2].get(rel3, set()) if anchor2 in ent_out else set()

        if not branch_result:
            consecutive_failures += 1
            continue

        if max_hop_size is not None and len(branch_result) > max_hop_size:
            consecutive_failures += 1
            continue

        # Compute final intersection
        answer_set = chain_result & branch_result

        if len(answer_set) == 0:
            num_empty += 1
            consecutive_failures += 1
            continue

        if len(answer_set) > max_answers:
            num_too_many += 1
            consecutive_failures += 1
            continue

        if len(answer_set) < min_answers:
            consecutive_failures += 1
            continue

        # Query format: ((anchor1, rel1, rel2), (anchor2, rel3), "pi")
        # Tensor: [anchor1, rel1, rel2, anchor2, rel3]
        query = ((anchor1, rel1, rel2), (anchor2, rel3), "pi")

        if query in answers:
            consecutive_failures += 1
            continue

        queries.append(query)
        answers[query] = answer_set
        num_sampled += 1
        consecutive_failures = 0

        if num_sampled % max(1, int(num_queries * 0.1)) == 0:
            logging.info(
                f"Generated {num_sampled}/{num_queries} queries "
                f"(tried: {num_try}, no_overlap: {num_no_overlap})"
            )

        sample_progress.update(1)

    sample_progress.close()
    logging.info(
        f"Generated {num_sampled}/{num_queries} queries "
        f"(total attempts: {num_try}, empty: {num_empty}, "
        f"too_many: {num_too_many}, no_overlap: {num_no_overlap})"
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "queries.pkl"), "wb") as f:
            pickle.dump(queries, f)
        with open(os.path.join(save_path, "answers.pkl"), "wb") as f:
            pickle.dump(answers, f)
        logging.info(f"Saved {len(queries)} queries to {save_path}")

    return queries, answers


def sample_pi_calibration_data(
    num_queries=1000,
    max_answers=1000,
    min_answers=1,
    neo4j_host="localhost",
    neo4j_bolt_port=7687,
    neo4j_database="neo4j",
    save_path="./calibration_data/pi_pipeline",
):
    logging.info("Starting pi query generation...")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_bolt_port}"
    db_controller = Neo4JBackendDBController(uri=neo4j_uri, nodeUID="id", relUID="type")

    logging.info("Loading graph data from database...")
    graph_data = get_graph(db_controller, device="cpu", augment_inverse_edges=True, relation_graph=True)
    logging.info(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    queries, answers = generate_pi_queries(
        graph_data,
        num_queries=num_queries,
        max_answers=max_answers,
        min_answers=min_answers,
        save_path=save_path,
    )

    metadata = {
        "query_type": "pi",
        "num_queries": len(queries),
        "num_answers": sum(len(ans) for ans in answers.values()),
        "graph_num_nodes": graph_data.num_nodes,
        "graph_num_edges": graph_data.num_edges,
        "max_answers": max_answers,
        "min_answers": min_answers,
    }
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    logging.info(f"pi query generation completed! Generated {len(queries)} queries.")
    return save_path


def main():
    parser = ArgumentParser(description="Generate pi (project-intersect) queries")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--max-answers", type=int, default=1000)
    parser.add_argument("--min-answers", type=int, default=1)
    parser.add_argument("--neo4j-host", type=str, default="localhost")
    parser.add_argument("--neo4j-bolt-port", type=int, default=7687)
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument("--save-path", type=str, default="./calibration_data/pi_pipeline")
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
        save_path = sample_pi_calibration_data(
            num_queries=args.num_queries,
            max_answers=args.max_answers,
            min_answers=args.min_answers,
            neo4j_host=args.neo4j_host,
            neo4j_bolt_port=args.neo4j_bolt_port,
            neo4j_database=args.neo4j_database,
            save_path=args.save_path,
        )
        print(f"\npi queries saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error during pi query generation: {e}")
        raise


if __name__ == "__main__":
    main()
