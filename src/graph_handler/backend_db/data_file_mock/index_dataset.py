import pickle
import os.path as osp
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import json


def index_dataset(data_path):
    train_ent_file = osp.join(data_path, "raw_data/train-ents.txt")
    valid_ent_file = osp.join(data_path, "raw_data/valid-ents.txt")
    test_ent_file = osp.join(data_path, "raw_data/test-ents.txt")

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}
    entid, relid = 0, 0

    ents = defaultdict(list)

    for p in [train_ent_file, valid_ent_file, test_ent_file]:
        data_type = p.split("-")[0].split("/")[-1]
        with open(p, "r") as f:
            for line in f.readlines():
                e = line.strip()
                if e not in ent2id.keys():
                    ent2id[e] = entid
                    id2ent[entid] = e
                    ents[data_type].append(entid)
                    entid += 1
                else:
                    raise ValueError("Duplicate entity found in {0}".format(p))

    with open(osp.join(data_path, "ent2id.pkl"), "wb") as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(data_path, "id2ent.pkl"), "wb") as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)

    relation_file = osp.join(data_path, "raw_data/relations.txt")
    with open(relation_file, "r") as f:
        for line in f.readlines():
            rel = line.strip()
            rel_reverse = rel + "_reverse"
            if rel not in rel2id.keys():
                rel2id[rel] = relid
                id2rel[relid] = rel
                assert relid % 2 == 0
                relid += 1
            else:
                raise ValueError(
                    "Duplicate relation found in {0}".format(relation_file)
                )
            if rel_reverse not in rel2id.keys():
                rel2id[rel_reverse] = relid
                id2rel[relid] = rel_reverse
                assert relid % 2 == 1
                relid += 1
            else:
                raise ValueError(
                    "Duplicate relation found in {0}".format(relation_file)
                )

    with open(osp.join(data_path, "rel2id.pkl"), "wb") as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(data_path, "id2rel.pkl"), "wb") as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    files = [
        "raw_data/ind-train.txt",
        "raw_data/ind-valid.txt",
        "raw_data/ind-test.txt",
        "raw_data/ind-valid-support.txt",
        "raw_data/ind-test-support.txt",
    ]
    indexified_files = [
        "ind-train_indexified.txt",
        "ind-valid_indexified.txt",
        "ind-test_indexified.txt",
        "ind-valid-support_indexified.txt",
        "ind-test-support_indexified.txt",
    ]

    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(data_path, indexified_p), "w")
        with open(osp.join(data_path, p), "r") as f:
            for line in tqdm(f.readlines()):
                e1, rel, e2 = line.strip().split("\t")
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = rel + "_reverse"

                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write(
                        "\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])])
                        + "\n"
                    )
                    fw.write(
                        "\t".join(
                            [str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]
                        )
                        + "\n"
                    )
        fw.close()

    stats = {
        "numtrainentities": len(ents["train"]),
        "numvalidentities": len(ents["valid"]),
        "numtestentities": len(ents["test"]),
        "numrelations": len(rel2id),
    }
    json.dump(stats, open(osp.join(data_path, "stats.json"), "w"))

    print(
        "num train entity: %d, num valid entity: %d, num test entity: %d"
        % (len(ents["train"]), len(ents["valid"]), len(ents["test"]))
    )
    print("indexing finished!!")


if __name__ == "__main__":
    data_path = "../../../datasets/FB15k-237-ind/"
    index_dataset(data_path)
