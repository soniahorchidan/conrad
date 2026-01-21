import argparse
import random
from neo4j import GraphDatabase
from tqdm import tqdm


def delete_random_edges(uri, user, password, percent, chunk_size=1000, seed=1):
    rng = random.Random(seed)  # isolate RNG so other code cannot disturb it

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            # Deterministic, de-duplicated list of relationship ids
            id_result = session.run(
                "MATCH ()-[r]-() "
                "RETURN DISTINCT id(r) AS rid "
                "ORDER BY rid"
            )
            all_ids = [record["rid"] for record in id_result]

            total = len(all_ids)
            to_delete_count = int(total * (percent / 100.0))
            print(f"Total relationships: {total}")
            print(f"Deleting {to_delete_count} relationships ({percent}% of total) at random...")

            if to_delete_count <= 0:
                print("Nothing to delete. Exiting.")
                return

            # Deterministic sample given fixed all_ids and seed
            delete_ids = rng.sample(all_ids, to_delete_count)

            # Delete in chunks
            num_chunks = (to_delete_count + chunk_size - 1) // chunk_size
            for i in tqdm(range(0, to_delete_count, chunk_size), 
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
        description='Delete a random percentage of edges from the Neo4j database.'
    )
    parser.add_argument(
        '--perc', type=float, required=True,
        help='Percentage of edges to delete (0-100)'
    )

    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password123"

    args = parser.parse_args()
    if not 0 <= args.perc <= 100:
        parser.error('perc must be between 0 and 100')

    delete_random_edges(uri, user, password, args.perc)
