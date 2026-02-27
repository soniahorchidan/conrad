from datasets import load_dataset, concatenate_datasets
import csv
import os
import sys
import pyjson5
import hashlib
import urllib.request


def get_ent_and_rel_mappings(combined_dataset):
    entities = set()
    relations = set()

    print("Getting entity and relation mappings...")
    for triple in combined_dataset:
        head = triple["head"]
        if dataset_name == "fb15k-237":
            # NOTE: stored with the wrong labels in the original dataset
            relation = triple["tail"]
            tail = triple["relation"]
        elif dataset_name == "yago310":
            relation = triple["relation"]
            tail = triple["tail"]

        entities.add(head)
        entities.add(tail)
        relations.add(relation)

    entities = list(entities)
    relations = list(relations)

    # sort entities and relations to ensure consistent mappings
    entities.sort()
    relations.sort()

    ent2ind = {entities[i]: i for i in range(len(entities))}
    rel2ind = {relations[i]: i for i in range(len(relations))}
    return ent2ind, rel2ind


# Remap relations to start from 0 for nell-955
def remap_relations(relations):
    rels = sorted(set(int(r[2]) for r in relations))
    rel2ind = {rel: i for i, rel in enumerate(rels)}
    return [[r[0], r[1], rel2ind[int(r[2])], r[3]] for r in relations]


def get_entities_and_relations_yago310():
    """Process YAGO3-10 dataset (uses entity names directly as labels)."""
    train_dataset = dataset["train"]
    ent2ind, rel2ind = get_ent_and_rel_mappings(train_dataset)
    
    entities = set()
    relations = list()
    seen_rels_hashes = set()

    print("Processing dataset %s..." % dataset_name)

    for triple in train_dataset:
        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]

        # Check if we inserted the triple already. If it has, ignore it.
        t = head + relation + tail
        triple_hash = hashlib.sha256(t.encode()).hexdigest()
        if triple_hash in seen_rels_hashes:
            continue
        else:
            seen_rels_hashes.add(triple_hash)

        # Use entity name as label directly
        entities.add((ent2ind[head], head))
        entities.add((ent2ind[tail], tail))
        relations.append([ent2ind[head], ent2ind[tail], rel2ind[relation], relation])
    
    return entities, relations


def get_entities_and_relations_q2b(q2b_mid2name_path=None):
    # Access the 'train' split of the dataset
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    validation_dataset = dataset["validation"]
    combined_dataset = concatenate_datasets(
        [train_dataset, validation_dataset, test_dataset]
    )

    ent2ind, rel2ind = get_ent_and_rel_mappings(combined_dataset)
    
    # For fb15k-237, use name mapping from JSON file
    # Get the mapping between the original FB15k names and the indices
    if q2b_mid2name_path is None:
        raise ValueError(f"q2b_mid2name_path is required for dataset {dataset_name}")
    with open(q2b_mid2name_path, "r") as file:
        name_hashmap = pyjson5.load(file)

    entities = set()
    relations = list()
    seen_rels_hashes = set()

    print("Processing dataset %s..." % dataset_name)
    unknown_id = 0

    for triple in combined_dataset:
        head = triple["head"]
        # NOTE: fb15k-237 stored with the wrong labels in the original dataset
        relation = triple["tail"]
        tail = triple["relation"]

        # Check if we inserted the triple already. If it has, ignore it.
        t = head + relation + tail
        triple_hash = hashlib.sha256(t.encode()).hexdigest()
        if triple_hash in seen_rels_hashes:
            continue
        else:
            seen_rels_hashes.add(triple_hash)

        if head not in name_hashmap:
            name_hashmap[head] = {"label": f"UNK_{unknown_id}"}
            unknown_id += 1
        label = name_hashmap[head]["label"]
        entities.add((ent2ind[head], label))

        if tail not in name_hashmap:
            name_hashmap[tail] = {"label": f"UNK_{unknown_id}"}
            unknown_id += 1
        label = name_hashmap[tail]["label"]
        entities.add((ent2ind[tail], label))
        relations.append([ent2ind[head], ent2ind[tail], rel2ind[relation], relation])
    return entities, relations


def write_to_files(dataset_name, entities, relations):
    # write to files
    OUT_PATH = "./artifacts/data/%s" % dataset_name
    os.makedirs(OUT_PATH, exist_ok=True)

    # ----------- General CSVs with headers ------------
    OUT_ENTITIES = OUT_PATH + "/train_ind_ent_with_headers.csv"
    OUT_RELATIONS = OUT_PATH + "/train_ind_rels_with_headers.csv"
    print("Writing entities for dataset %s..." % dataset_name)
    with open(OUT_ENTITIES, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "label"])
        writer.writerows(entities)
    print("Writing relations for dataset %s..." % dataset_name)
    with open(OUT_RELATIONS, "w", newline="") as out_rels:
        writer = csv.writer(out_rels)
        writer.writerow(["src", "dst", "type", "label"])
        writer.writerows(relations)

    # ----------- Kuzu ------------
    OUT_ENT = OUT_PATH + "/train_ind_ent.csv"
    OUT_RELS = OUT_PATH + "/train_ind_rels.csv"

    print("Writing entities for dataset %s..." % dataset_name)
    with open(OUT_ENT, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([[item[0], item[1]] for item in list(entities)])

    print("Writing relations for dataset %s..." % dataset_name)
    with open(OUT_RELS, "w", newline="") as out_rels:
        writer = csv.writer(out_rels)
        writer.writerows(relations)

    # ----------- Neo4j ------------
    OUT_RELS_HEADER = OUT_PATH + "/neo4j_train_ind_rels_header.csv"
    OUT_RELS_NEO4J = OUT_PATH + "/neo4j_train_ind_rels.csv"
    OUT_ENT_HEADER = OUT_PATH + "/neo4j_train_ind_ent_header.csv"
    OUT_ENT_NEO4J = OUT_PATH + "/neo4j_train_ind_ent.csv"
    ENT_LABEL = "Entity"
    REL_TYPE = "Relation"

    print("Writing headers for dataset %s..." % dataset_name)
    with open(OUT_RELS_HEADER, "w") as out_rels_header:
        out_rels_header.write(":START_ID,:END_ID,type:int,label:string,:TYPE")

    with open(OUT_ENT_HEADER, "w") as out_ent_header:
        out_ent_header.write(":ID,id:int,:LABEL,label:string")

    print("Writing Neo4j relations for dataset %s..." % dataset_name)
    with open(OUT_RELS_NEO4J, "w", newline="") as out_rels:
        writer = csv.writer(out_rels)
        writer.writerows([item + [REL_TYPE] for item in relations])

    print("Writing Neo4j entities for dataset %s..." % dataset_name)
    with open(OUT_ENT_NEO4J, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(
            [[item[0], item[0], ENT_LABEL, item[1]] for item in list(entities)]
        )

    print("Processing dataset %s...done." % dataset_name)


def download_csv(url, save_path):
    # Check if the file already exists at the save path.
    if os.path.exists(save_path):
        print("File already exists. Skipping download.")
        return
    else:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with urllib.request.urlopen(url) as response, open(save_path, "wb") as out_file:
            data = response.read()  # Read the content of the response
            out_file.write(data)  # Write the content to the file


def load_csv(csv_file_path):
    result = []
    with open(csv_file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')

        for row in reader:
            result.append(row)

    return result


if __name__ == "__main__":
    assert len(sys.argv) > 1
    dataset_name = sys.argv[1]

    # Read original dataset
    print("Loading dataset %s..." % dataset_name)
    if dataset_name == "fb15k-237":
        mid2name_path = f"./datasets/{dataset_name}/entity2wikidata.json"
        dataset = load_dataset("VLyb/FB15k-237")
        entities, relations = get_entities_and_relations_q2b(mid2name_path)
        write_to_files(dataset_name, entities, relations)
    elif dataset_name == "nell-955":
        # Download files to artifacts/data/nell-955/
        dataset_path = "./artifacts/data/nell-955"
        os.makedirs(dataset_path, exist_ok=True)
        
        entities_filepath = f"{dataset_path}/nell955_entities.csv"
        download_csv(
            "https://www.dropbox.com/scl/fi/wp4crv2urmxtpln1m3g6y/nell955_entity_id.csv?rlkey=ihohnuac5ds6era1owouew9fm&st=j65izs1b&dl=1",
            entities_filepath,
        )
        entities = load_csv(entities_filepath)

        relations_filepath = f"{dataset_path}/nell955_relations.csv"
        download_csv(
            "https://www.dropbox.com/scl/fi/db191d66agiqzcwmj7t0a/nell955_relations.csv?rlkey=m4c81eh0lau1akau2o5ew6cev&st=c4gls2r1&dl=1",
            relations_filepath,
        )
        relations = load_csv(relations_filepath)

        relations = remap_relations(relations)
        write_to_files("nell-955", entities, relations)
    elif dataset_name == "yago310":
        # Load YAGO3-10 from Hugging Face (only train split needed for Neo4j import)
        dataset = load_dataset("VLyb/YAGO3-10")
        entities, relations = get_entities_and_relations_yago310()
        write_to_files("yago310", entities, relations)
    else:
        print(
            "Dataset %s not supported. Supported: fb15k-237, nell-955, yago310. Exiting."
            % dataset_name
        )
        exit(1)