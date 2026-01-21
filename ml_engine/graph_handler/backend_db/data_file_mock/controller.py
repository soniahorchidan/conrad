from tqdm import tqdm
import igraph
import torch
import os.path as osp
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from collections import defaultdict
from ..common import AbstractDBController
from .index_dataset import index_dataset
import pickle
import ast
import random


def build_graph(data_path, indexified_files):
    edge_index, edge_type = [], []
    for indexified_p in indexified_files:
        with open(osp.join(data_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split("\t")
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                edge_index.append([e1, e2])
                edge_type.append(rel)
    edge_index = np.array(edge_index).T
    edge_type = np.array(edge_type)
    edge_index = torch.from_numpy(edge_index)
    edge_type = torch.from_numpy(edge_type)
    return edge_index, edge_type


class MockController(AbstractDBController):
    """
    MockController class implements database operations specific to Neo4J.
    """

    def __init__(self, db_path=None):
        super().__init__()
        self.load_data()
        self.set_mode("train")
        self.entityDict = defaultdict(dict)

    def connect_to_db(self, db_path):
        pass

    def get_all_node_ids(self):
        return list(self.id2ent.keys())

    def get_all_relations(self):
        return list(self.id2rel.keys())

    def get_num_relations(self):
        return len(self.id2rel)

    def get_node_attributes(self, nids: list):
        ret = {}
        for nid in nids:
            if nid in self.entityDict:
                ret[nid] = self.entityDict[nid]
        return ret

    def get_whole_graph(self):
        raise NotImplementedError

    def set_property(self, nids: list, new_prop: str, prop_value):
        for nid in nids:
            self.entityDict[nid][new_prop] = prop_value

    def remove_property(self, nids: list, property: str):
        if nids is None:
            for nid in self.entityDict:
                self.entityDict[nid].pop(property, None)
        else:
            for nid in nids:
                self.entityDict[nid].pop(property, None)

    def is_anchor_node(self, nid: int):
        attrs = self.get_node_attributes([nid])[nid]
        if self._prefix + "isAnchor" in attrs:
            return attrs[self._prefix + "isAnchor"]
        return False

    def get_anchor_node_ids(self):
        return [
            nid
            for nid in self.entityDict
            if self.entityDict[nid].get("isAnchor") == "True"
        ]

    def write_isAnchor_property(self, ids: list, value: str):
        self.set_property(ids, "isAnchor", value)

    def get_random_node_ids(self, n: int):
        all_nodes = self.get_all_node_ids()
        sampled_nodes = random.sample(all_nodes, n)
        return sampled_nodes

    def get_one_hop_neighbours(
        self, nid, k=None, flow="both", ret_rels=False, merge_and_uniq=False
    ):
        raise NotImplementedError

    def get_two_hop_neighbours(self, nid, one_hop_ns, maxx):
        raise NotImplementedError

    def get_neighbourhood(self, nids: list, maxx: int, hops: int, flow="both"):
        """
        Retrieves neighbors of a given node from the Neo4J database for a specified number of hops.
        Args:
            nids (list): The IDs of the nodes whose neighbors are to be retrieved.
            maxx (int): The maximum number of neighbors to retrieve at each hop.
            hops (int): The number of hops to retrieve neighbors for.
            flow (str): The direction of the flow of the neighbors. Can be 'both', 'incoming', or 'outgoing'. Not used in this implementation.
        Returns:
            list: A list containing the neighbors for each hop for each node in nids.
        """
        edge_index = self.get_edge_index()
        data = Data(edge_index=edge_index).contiguous()
        neighbor_loader = NeighborLoader(
            data, num_neighbors=[maxx] * hops, shuffle=True
        )
        nids = torch.tensor(nids, dtype=torch.long)
        batch = neighbor_loader(nids).edge_index.tolist()
        return batch

    def get_dists_to_anchors_property(self, nids: list, node_attr: dict = None):
        if node_attr is None:
            node_attr = self.get_node_attributes(nids)
        dists = {}
        for nid, node in node_attr.items():
            if self._prefix + "distsToAnchors" in node:
                dists[nid] = node[self._prefix + "distsToAnchors"]
            else:
                dists[nid] = None
        return dists

    def get_relation_context(self, nids: list, node_attr: dict = None):
        if node_attr is None:
            node_attr = self.get_node_attributes(nids)
        relation_context = {}
        for nid, node in node_attr.items():
            if self._prefix + "relationContext" in node:
                relation_context[nid] = node[self._prefix + "relationContext"]
            else:
                relation_context[nid] = None
        return relation_context

    def get_neighbours_edge_types(self, nid, k, flow="both", unique=True):
        edge_index = self.get_edge_index()
        edge_type = self.get_edge_type()
        # filter out the edges that are connected to the given node
        if flow == "both":
            mask = (edge_index[0] == nid) | (edge_index[1] == nid)
        elif flow == "incoming":
            mask = edge_index[1] == nid
        elif flow == "outgoing":
            mask = edge_index[0] == nid

        edge_index = edge_index[:, mask]
        edge_type = edge_type[mask]
        # get the unique edge types
        edge_types = torch.unique(edge_type).tolist()
        if unique:
            edge_types = list(set(edge_types))
        if k is not None:
            edge_types = edge_types[:k]
        return edge_types

    def write_dists_to_anchors_property(self, dists: dict):
        for nid in tqdm(dists.keys()):
            if len(dists[nid]) > 0:
                self.set_property([nid], "distsToAnchors", dists[nid])

    def calculate_distance_to_anchor_nodes(self, nid: int, anchors: list):
        edge_index = self.get_edge_index()
        graph = igraph.Graph(edges=edge_index.T)
        paths = graph.shortest_paths(source=nid, target=anchors, output="vpath")
        dists = {nid: [(anchor, len(path) - 1) for path, anchor in zip(paths, anchors)]}
        return dists

    def write_relation_context(self, relation_context):
        for nid in tqdm(relation_context.keys()):
            if len(relation_context[nid]) > 0:
                self.set_property([nid], "relationContext", relation_context[nid])

    def multi_hop_query_retrieval(self, nid: int, rids: list):
        raise NotImplementedError

    def sample_queries(self, hops: int, batch_size: int, negative_sample_size: int):
        queries, tails, negative_tails, subsampling_weight = [], [], [], []
        while len(queries) < batch_size:
            start_node_id = self.get_random_node_ids(1)[0]
            involved_nodes = [start_node_id]
            involved_rels = []
            now_node_id = start_node_id
            for _ in range(hops):
                neighbour = self.get_one_hop_neighbours(
                    now_node_id, flow="outgoing", ret_rels=True
                )
                if len(neighbour) == 0:
                    break
                filtered_neighbour = [
                    x for x in neighbour if x[-1] not in involved_nodes
                ]
                if len(filtered_neighbour) == 0:
                    break
                neighbour = random.choice(filtered_neighbour)
                now_node_id = neighbour[-1]
                involved_nodes.append(now_node_id)
                involved_rels.append(neighbour[1])
            if len(involved_nodes) == hops + 1:
                queries.append([start_node_id] + involved_rels)
                tails.append([now_node_id])

        for query in queries:
            db_query = f"""
            MATCH (n){"".join([f'-[r{i}]->(n{i})' for i in range(1, len(query))])}
            WHERE n.id = {query[0]}{''.join([f' AND r{i}.type = {query[i]}' for i in range(1, len(query))])}
            RETURN n{len(query) - 1}.id
            """
            records, _, _ = self.driver.execute_query(db_query, database_="neo4j")
            true_tails = [record.data()[f"n{len(query) - 1}.id"] for record in records]
            negative_tails.append(
                negative_sampling(
                    negative_sample_size, self.get_all_node_ids(), true_tails
                )
            )
            subsampling_weight.append(np.sqrt(1 / (len(true_tails) + 4)))

        return queries, tails, negative_tails, subsampling_weight

    def sample_queries_neg_relation(self, batch_size: int, negative_sample_size: int):
        queries, tails, negative_relations, subsampling_weight = [], [], [], []
        while len(queries) < batch_size:
            start_node_id = self.get_random_node_ids(1)[0]
            hops = 1
            involved_nodes = [start_node_id]
            involved_rels = []
            now_node_id = start_node_id
            for _ in range(hops):
                neighbour = self.get_one_hop_neighbours(
                    now_node_id, flow="outgoing", ret_rels=True
                )
                if len(neighbour) == 0:
                    break
                filtered_neighbour = [
                    x for x in neighbour if x[-1] not in involved_nodes
                ]
                if len(filtered_neighbour) == 0:
                    break
                neighbour = random.choice(filtered_neighbour)
                now_node_id = neighbour[-1]
                involved_nodes.append(now_node_id)
                involved_rels.append(neighbour[1])
            if len(involved_nodes) == hops + 1:
                queries.append([start_node_id] + involved_rels)
                tails.append([now_node_id])

        for i, query in enumerate(queries):
            db_query = f"""
            MATCH (n)-[r]->(n1)
            WHERE n.id = {query[0]} AND n1.id = {tails[i][0]}
            RETURN r.id
            """
            records, _, _ = self.driver.execute_query(db_query, database_="neo4j")
            true_rels = [record.data()["r.id"] for record in records]
            negative_relations.append(
                negative_sampling(
                    negative_sample_size, self.get_all_relations(), true_rels
                )
            )
            subsampling_weight.append(np.sqrt(1 / (len(true_rels) + 4)))

        return queries, tails, negative_relations, subsampling_weight

    def set_mode(self, mode):
        assert mode in ["train", "valid", "test"]
        self.mode = mode

    def get_edge_index(self):
        if self.mode == "train":
            return self.train_edge_index
        elif self.mode == "valid":
            return self.valid_edge_index
        elif self.mode == "test":
            return self.test_edge_index

    def get_edge_type(self):
        if self.mode == "train":
            return self.train_edge_type
        elif self.mode == "valid":
            return self.valid_edge_type
        elif self.mode == "test":
            return self.test_edge_type

    def load_data(self):
        if not osp.exists(osp.join(self.data_path, "ent2id.pkl")):
            index_dataset(self.data_path)
        graph_files_train = ["ind-train_indexified.txt"]
        self.train_edge_index, self.train_edge_type = build_graph(
            self.data_path, graph_files_train
        )
        graph_files_valid = [
            "ind-train_indexified.txt",
            "ind-valid-support_indexified.txt",
        ]
        self.valid_edge_index, self.valid_edge_type = build_graph(
            self.data_path, graph_files_valid
        )
        graph_files_test = [
            "ind-train_indexified.txt",
            "ind-test-support_indexified.txt",
        ]
        self.test_edge_index, self.test_edge_type = build_graph(
            self.data_path, graph_files_test
        )
        self.ent2id = pickle.load(open(osp.join(self.data_path, "ent2id.pkl"), "rb"))
        self.id2ent = pickle.load(open(osp.join(self.data_path, "id2ent.pkl"), "rb"))
        self.rel2id = pickle.load(open(osp.join(self.data_path, "rel2id.pkl"), "rb"))
        self.id2rel = pickle.load(open(osp.join(self.data_path, "id2rel.pkl"), "rb"))

    def sample_sub_graph(self, num_entities: int):
        raise NotImplementedError

    def close(self):
        pass


def negative_sampling(target_negative_sample_size, all_entities, true_tails):
    negative_sample_list = []
    negative_sample_size = 0
    while negative_sample_size < target_negative_sample_size:
        negative_sample = np.random.choice(
            all_entities, target_negative_sample_size * 2, replace=False
        )
        mask = np.in1d(negative_sample, true_tails, assume_unique=True, invert=True)
        negative_sample = negative_sample[mask]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size
    negative_sample = np.concatenate(negative_sample_list)[:target_negative_sample_size]
    negative_sample = negative_sample.tolist()
    return negative_sample
