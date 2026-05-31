"""Degree-stratified random edge deletion.

Unlike `delete_random_edges.py`, which removes edges uniformly at random,
this script bins edges into strata based on the minimum endpoint degree
(`min(deg(u), deg(v))`) and applies a different deletion rate per stratum.
Edges incident to low-degree (long-tail) entities are removed at a higher
rate than edges between high-degree entities, while the overall fraction
of edges removed still matches `--perc`.

Within a stratum, sampling is uniform — every edge with the same
min-endpoint-degree bucket has the same probability of being removed.
This addresses the reviewer's concern that uniform deletion may not
reflect the long-tail nature of real-world KG incompleteness.
"""

import argparse
import random

from neo4j import GraphDatabase
from tqdm import tqdm


def fetch_edges_with_endpoints(session):
    result = session.run(
        "MATCH (u)-[r]->(v) "
        "RETURN id(r) AS rid, id(u) AS uid, id(v) AS vid "
        "ORDER BY rid"
    )
    return [(rec["rid"], rec["uid"], rec["vid"]) for rec in result]


def compute_degrees(session):
    result = session.run(
        "MATCH (n)-[r]-() "
        "RETURN id(n) AS nid, count(r) AS deg"
    )
    return {rec["nid"]: rec["deg"] for rec in result}


def assign_strata(edges, degrees, num_strata):
    """Bin edges by min-endpoint-degree into `num_strata` quantile-based buckets.

    Returns: list of lists — strata[i] is the list of edge ids in stratum i,
    ordered from lowest min-degree (stratum 0) to highest (stratum num_strata - 1).
    """
    edge_scores = [(rid, min(degrees[uid], degrees[vid])) for rid, uid, vid in edges]
    edge_scores.sort(key=lambda x: x[1])

    total = len(edge_scores)
    strata = []
    for i in range(num_strata):
        lo = (i * total) // num_strata
        hi = ((i + 1) * total) // num_strata
        strata.append([rid for rid, _ in edge_scores[lo:hi]])
    return strata, edge_scores


def per_stratum_targets(strata_sizes, overall_perc, bias):
    """Compute how many edges to delete from each stratum.

    Uses a linear gradient over stratum index: stratum 0 (lowest degree) has
    rate `overall * (1 + bias)`, stratum N-1 has rate `overall * (1 - bias)`.
    Rates linearly interpolate, average exactly to `overall_perc` when strata
    are equal-sized (quantile binning guarantees near-equal sizes).

    `bias` ∈ [0, 1]:
      - 0   → uniform across strata (matches `delete_random_edges.py`)
      - 1   → maximally biased (low-degree rate is 2× overall, high-degree is 0)

    Per-stratum rates are clipped to [0, 100] and the total deletion count is
    rebalanced so the overall percentage still matches `overall_perc`.
    """
    n = len(strata_sizes)
    if n == 1:
        return [int(strata_sizes[0] * overall_perc / 100.0)]

    rates = []
    for i in range(n):
        # Linearly interpolate from (1 + bias) at i=0 to (1 - bias) at i=n-1.
        frac = i / (n - 1)
        rate = overall_perc * (1 + bias - 2 * bias * frac)
        rates.append(max(0.0, min(100.0, rate)))

    counts = [int(size * rate / 100.0) for size, rate in zip(strata_sizes, rates)]

    # Rebalance to hit the global target exactly (rounding + clipping drift).
    total_edges = sum(strata_sizes)
    target_total = int(total_edges * overall_perc / 100.0)
    drift = target_total - sum(counts)
    if drift != 0:
        # Distribute residual proportionally to remaining capacity in each stratum
        # (favor strata that aren't already saturated/empty).
        if drift > 0:
            capacities = [size - c for size, c in zip(strata_sizes, counts)]
        else:
            capacities = list(counts)  # how much we could give back
        total_cap = sum(capacities)
        if total_cap > 0:
            for i, cap in enumerate(capacities):
                share = int(round(drift * cap / total_cap))
                counts[i] += share
            # Final tiny correction
            counts[0] += target_total - sum(counts)

    counts = [max(0, min(size, c)) for size, c in zip(strata_sizes, counts)]
    return counts


def delete_stratified_edges(uri, user, password, percent, num_strata, bias,
                            chunk_size=1000, seed=1):
    rng = random.Random(seed)

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            print("Fetching graph topology...")
            edges = fetch_edges_with_endpoints(session)
            degrees = compute_degrees(session)

            total = len(edges)
            print(f"Total relationships: {total}")
            print(f"Total nodes with degree info: {len(degrees)}")

            strata, edge_scores = assign_strata(edges, degrees, num_strata)
            strata_sizes = [len(s) for s in strata]
            counts = per_stratum_targets(strata_sizes, percent, bias)

            # Report stratum boundaries (min-degree range per stratum) for the paper.
            print(f"\nDegree-stratified deletion (strata={num_strata}, bias={bias}, overall={percent}%)")
            print(f"{'stratum':>8} {'size':>10} {'min_deg':>10} {'max_deg':>10} {'rate%':>8} {'to_delete':>12}")
            for i, stratum in enumerate(strata):
                if stratum:
                    # edge_scores is sorted by min-degree; stratum i is slice [lo, hi).
                    lo = (i * total) // num_strata
                    hi = ((i + 1) * total) // num_strata
                    min_deg = edge_scores[lo][1]
                    max_deg = edge_scores[hi - 1][1]
                else:
                    min_deg = max_deg = -1
                rate = (100.0 * counts[i] / strata_sizes[i]) if strata_sizes[i] else 0.0
                print(f"{i:>8d} {strata_sizes[i]:>10d} {min_deg:>10d} {max_deg:>10d} "
                      f"{rate:>8.2f} {counts[i]:>12d}")

            total_to_delete = sum(counts)
            print(f"\nTotal to delete: {total_to_delete} ({100.0 * total_to_delete / total:.2f}% of {total})")

            if total_to_delete == 0:
                print("Nothing to delete. Exiting.")
                return

            # Sample within each stratum independently using the seeded RNG.
            delete_ids = []
            for i, stratum in enumerate(strata):
                if counts[i] >= len(stratum):
                    delete_ids.extend(stratum)
                else:
                    delete_ids.extend(rng.sample(stratum, counts[i]))

            # Delete in chunks.
            num_chunks = (len(delete_ids) + chunk_size - 1) // chunk_size
            for i in tqdm(range(0, len(delete_ids), chunk_size),
                          total=num_chunks,
                          desc="Deleting relationships"):
                chunk = delete_ids[i:i + chunk_size]
                session.run(
                    "MATCH ()-[r]-() WHERE id(r) IN $ids DELETE r",
                    ids=chunk
                )
            print("\nDeletion complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Delete edges from Neo4j using degree-stratified sampling. '
                     'Edges between low-degree (long-tail) entities are removed '
                     'at a higher rate than edges between high-degree entities, '
                     'while preserving the overall deletion percentage.')
    )
    parser.add_argument(
        '--perc', type=float, required=True,
        help='Overall percentage of edges to delete (0-100). The per-stratum rates '
             'are chosen to average to this value.'
    )
    parser.add_argument(
        '--num-strata', type=int, default=5,
        help='Number of degree-based strata (default: 5 quintiles).'
    )
    parser.add_argument(
        '--bias', type=float, default=1.0,
        help='Bias strength in [0, 1]. 0 = uniform across strata; 1 = maximally '
             'skewed (lowest stratum has 2x overall rate, highest has 0). '
             'Default: 1.0.'
    )
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Random seed for reproducible sampling.'
    )
    parser.add_argument("--neo4j-host", type=str, default="localhost")
    parser.add_argument("--neo4j-bolt-port", type=int, default=7687)

    args = parser.parse_args()
    if not 0 <= args.perc <= 100:
        parser.error('--perc must be between 0 and 100')
    if not 0 <= args.bias <= 1:
        parser.error('--bias must be between 0 and 1')
    if args.num_strata < 1:
        parser.error('--num-strata must be >= 1')

    uri = f"bolt://{args.neo4j_host}:{args.neo4j_bolt_port}"
    print("Deleting edges on Neo4j instance:", uri)
    delete_stratified_edges(
        uri, "neo4j", "password123",
        percent=args.perc,
        num_strata=args.num_strata,
        bias=args.bias,
        seed=args.seed,
    )
