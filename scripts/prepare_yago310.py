#!/usr/bin/env python3
"""
Download and prepare YAGO3-10 dataset for Neo4j import.

This script:
1. Downloads YAGO3-10 dataset from Hugging Face
2. Converts it to Neo4j CSV format (entities and relationships)

Dataset source: https://huggingface.co/datasets/VLyb/YAGO3-10
"""

import os
import sys
import csv
from collections import defaultdict
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library is required.")
    print("Please install it with: pip install datasets")
    sys.exit(1)

# Dataset configuration
YAGO310_DATASET = "VLyb/YAGO3-10"

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "artifacts" / "data"
YAGO310_DIR = DATA_PATH / "yago310"


def load_triples_from_hf():
    """Load triples from Hugging Face dataset."""
    print(f"Downloading YAGO3-10 from Hugging Face: {YAGO310_DATASET}")
    print("This may take a few minutes...")
    
    # Load the dataset
    dataset = load_dataset(YAGO310_DATASET)
    
    # Get train split (we only need train for Neo4j import)
    train_data = dataset["train"]
    
    triples = []
    entities = set()
    relations = set()
    
    print(f"Processing {len(train_data)} training triples...")
    for example in train_data:
        head = str(example["head"]).strip()
        relation = str(example["relation"]).strip()
        tail = str(example["tail"]).strip()
        
        triples.append((head, relation, tail))
        entities.add(head)
        entities.add(tail)
        relations.add(relation)
    
    print(f"Loaded {len(triples)} triples, {len(entities)} entities, {len(relations)} relations")
    return triples, entities, relations


def create_entity_mapping(entities):
    """Create entity ID mapping."""
    entity_to_id = {}
    id_to_entity = {}
    
    for idx, entity in enumerate(sorted(entities)):
        entity_to_id[entity] = idx
        id_to_entity[idx] = entity
    
    return entity_to_id, id_to_entity


def create_relation_mapping(relations):
    """Create relation ID mapping."""
    relation_to_id = {}
    id_to_relation = {}
    
    for idx, relation in enumerate(sorted(relations)):
        relation_to_id[relation] = idx
        id_to_relation[idx] = relation
    
    return relation_to_id, id_to_relation


def write_neo4j_entity_files(entity_to_id, id_to_entity, output_dir: Path):
    """Write Neo4j entity CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write header file
    header_file = output_dir / "neo4j_train_ind_ent_header.csv"
    with open(header_file, "w", newline="") as f:
        f.write(":ID,id:int,:LABEL,label:string\n")
    print(f"Created {header_file}")
    
    # Write entity data file
    entity_file = output_dir / "neo4j_train_ind_ent.csv"
    with open(entity_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for entity_id, entity_name in sorted(id_to_entity.items()):
            # Format: ID, id, Label, label
            writer.writerow([entity_id, entity_id, "Entity", entity_name])
    print(f"Created {entity_file} with {len(entity_to_id)} entities")


def write_neo4j_relation_files(triples, entity_to_id, relation_to_id, output_dir: Path):
    """Write Neo4j relationship CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write header file
    header_file = output_dir / "neo4j_train_ind_rels_header.csv"
    with open(header_file, "w", newline="") as f:
        f.write(":START_ID,:END_ID,type:int,label:string,:TYPE\n")
    print(f"Created {header_file}")
    
    # Write relationship data file
    rel_file = output_dir / "neo4j_train_ind_rels.csv"
    with open(rel_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for head, rel, tail in triples:
            start_id = entity_to_id[head]
            end_id = entity_to_id[tail]
            rel_id = relation_to_id[rel]
            # Format: START_ID, END_ID, type, label, TYPE
            writer.writerow([start_id, end_id, rel_id, rel, "Relation"])
    print(f"Created {rel_file} with {len(triples)} relationships")


def main():
    """Main function to download and prepare YAGO3-10 dataset."""
    print("=" * 80)
    print("YAGO3-10 Dataset Preparation for Neo4j")
    print("=" * 80)
    print(f"Dataset source: https://huggingface.co/datasets/{YAGO310_DATASET}")
    print("=" * 80)
    
    # Create directories
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    YAGO310_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load triples from Hugging Face
    triples, entities, relations = load_triples_from_hf()
    
    # Step 2: Create mappings
    print("\nCreating entity and relation mappings...")
    entity_to_id, id_to_entity = create_entity_mapping(entities)
    relation_to_id, id_to_relation = create_relation_mapping(relations)
    
    print(f"Entity mapping: {len(entity_to_id)} entities")
    print(f"Relation mapping: {len(relation_to_id)} relations")
    
    # Step 3: Write Neo4j CSV files
    print("\nWriting Neo4j CSV files...")
    write_neo4j_entity_files(entity_to_id, id_to_entity, YAGO310_DIR)
    write_neo4j_relation_files(triples, entity_to_id, relation_to_id, YAGO310_DIR)
    
    print("\n" + "=" * 80)
    print("YAGO3-10 dataset preparation complete!")
    print(f"Output directory: {YAGO310_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
