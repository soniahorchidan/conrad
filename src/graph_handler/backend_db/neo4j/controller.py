import random
import ast
from tqdm import tqdm
import numpy as np
from neo4j import GraphDatabase
from ..common import AbstractDBController
from utils import remove_duplicates

class Neo4JBackendDBController(AbstractDBController):
    """
    Neo4JBackendDBController class implements database operations specific to Neo4J.
    """

    def __init__(self, uri, nodeUID, relUID):
        """
        Initializes the Neo4JBackendDBController with the provided URI.

        Args:
            uri (str): The URI of the Neo4J database.
        """
        super().__init__()
        self.uri = uri
        self.cache_all_node_ids = None
        self.cache_all_relations = None
        self.nodeUID = nodeUID
        self.relUID = relUID

    def connect_to_db(self, uri):
        driver = GraphDatabase.driver(uri, auth=("neo4j", "password123"))
        return driver

    def clear_cache(self):
        self.cache_all_node_ids = None
        self.cache_all_relations = None

    def execute_query(self, query, **kwargs):
        with self.connect_to_db(self.uri) as driver:
            records, _, _ = driver.execute_query(query, **kwargs, database_="neo4j")
        return records

    def get_all_node_ids(self):
        if self.cache_all_node_ids is not None:
            return self.cache_all_node_ids
        records = self.execute_query(f"MATCH (n) WHERE n.{self.nodeUID} IS NOT NULL RETURN n.{self.nodeUID}")
        all_nodes = [record.data()[f"n.{self.nodeUID}"] for record in records]
        self.cache_all_node_ids = all_nodes
        return all_nodes

    def get_all_relations(self):
        if self.cache_all_relations is not None:
            return self.cache_all_relations
        records = self.execute_query(f"MATCH ()-[r]->() RETURN DISTINCT r.{self.relUID}")
        all_relations = [record.data()[f"r.{self.relUID}"] for record in records]
        self.cache_all_relations = all_relations
        return all_relations

    def get_num_relations(self):
        all_relations = self.get_all_relations()
        # check if the relations are continuous
        if max(all_relations) == len(all_relations) - 1:
            return len(all_relations)
        else:
            raise ValueError("Relations are not continuous")

    def get_node_attributes(self, nids: list):
        records = self.execute_query(
            f"MATCH (n) WHERE n.{self.nodeUID} IN $nids RETURN n", nids=nids
        )
        attrs = {}
        for record in records:
            node = record.data()["n"]
            attrs[node[self.nodeUID]] = node
        return attrs
    
    def get_whole_graph(self):
        records = self.execute_query(f"MATCH (n)-[r]->(m) RETURN n.{self.nodeUID}, r.{self.relUID}, m.{self.nodeUID}")
        triples = []
        for record in records:
            record = record.data()
            triples.append((record[f"n.{self.nodeUID}"], record[f"r.{self.relUID}"], record[f"m.{self.nodeUID}"]))
        return triples

    def set_property(self, nids: list, new_prop: str, prop_value):
        new_prop = self._prefix + new_prop
        self.execute_query(
            f"MATCH (n) WHERE n.{self.nodeUID} IN $nids SET n.{new_prop} = $prop_value",
            nids=nids,
            new_prop=new_prop,
            prop_value=str(prop_value),
        )

    def remove_property(self, nids: list, property: str):
        if nids is None:
            self.execute_query(f"MATCH (n) REMOVE n.{self._prefix + property}")
        else:
            self.execute_query(
                f"MATCH (n) WHERE n.{self.nodeUID} IN $nids REMOVE n.{self._prefix + property}",
                nids=nids,
            )

    def is_anchor_node(self, nid: int):
        attrs = self.get_node_attributes([nid])[nid]
        if self._prefix + "isAnchor" in attrs:
            return attrs[self._prefix + "isAnchor"]
        return False

    def get_anchor_node_ids(self):
        property_ = self._prefix + "isAnchor"
        records = self.execute_query(
            f"MATCH (n) WHERE n.{property_} = 'True' RETURN n.{self.nodeUID}"
        )
        anchor_nodes = [record.data()[f"n.{self.nodeUID}"] for record in records]
        return anchor_nodes

    def write_isAnchor_property(self, ids: list, value: str):
        self.set_property(ids, "isAnchor", value)

    def get_random_node_ids(self, n: int):
        records = self.execute_query(
            f"MATCH (n) WHERE n.{self.nodeUID} IS NOT NULL RETURN n.{self.nodeUID} ORDER BY rand() LIMIT {n}"
        )
        sampled_nodes = [record.data()[f"n.{self.nodeUID}"] for record in records]
        return sampled_nodes

    def get_one_hop_neighbours(
        self, nid, k=None, flow="both", ret_rels=False, merge_and_uniq=False
    ):
        if isinstance(nid, int):
            nid_is_list = False
            nid = [nid]
        else:
            nid_is_list = True

        def _get_one_hop_neighbours(query, reverse=False):
            records = self.execute_query(query, nid=nid)
            results_dict = {}
            for record in records:
                n_id = record.data()["nodeId"]
                results_dict[n_id] = []
                n_ids = record.data()["neighbors"]
                if ret_rels:
                    relations = record.data()["relations"]
                    for r, n in zip(relations, n_ids):
                        if reverse:
                            results_dict[n_id].append([n, r, n_id])
                        else:
                            results_dict[n_id].append([n_id, r, n])
                else:
                    for n in n_ids:
                        if reverse:
                            results_dict[n_id].append([n, n_id])
                        else:
                            results_dict[n_id].append([n_id, n])

            results = [results_dict[n] if n in results_dict else [] for n in nid]
            return results

        if ret_rels:
            query_return = f"RETURN n.{self.nodeUID} AS nodeId, collect(relation.{self.relUID}) AS relations, collect(neighbor.{self.nodeUID}) AS neighbors"
        else:
            query_return = f"RETURN n.{self.nodeUID} AS nodeId, collect(neighbor.{self.nodeUID}) AS neighbors"
        query_match = "WITH $nid AS ids UNWIND ids AS nodeId "
        if flow == "incoming":
            query_match += f"MATCH (n)<-[relation]-(neighbor) WHERE n.{self.nodeUID} = nodeId"
            results = _get_one_hop_neighbours(
                f"{query_match} {query_return}", reverse=True
            )
        elif flow == "outgoing":
            query_match += f"MATCH (n)-[relation]->(neighbor) WHERE n.{self.nodeUID} = nodeId"
            results = _get_one_hop_neighbours(f"{query_match} {query_return}")
        elif flow == "both":
            query_match1 = (
                query_match + f"MATCH (n)<-[relation]-(neighbor) WHERE n.{self.nodeUID} = nodeId"
            )
            res1 = _get_one_hop_neighbours(
                f"{query_match1} {query_return}", reverse=True
            )
            query_match2 = (
                query_match + f"MATCH (n)-[relation]->(neighbor) WHERE n.{self.nodeUID} = nodeId"
            )
            res2 = _get_one_hop_neighbours(f"{query_match2} {query_return}")
            results = res1 + res2
        if k is not None:
            for i in range(len(results)):
                if len(results[i]) > k:
                    results[i] = random.sample(results[i], k)

        if not nid_is_list:
            results = results[0]
        elif merge_and_uniq:
            results = list(set([tuple(x) for re in results for x in re]))
            results = [list(x) for x in results]
        return results

    def get_two_hop_neighbours(self, nid, one_hop_ns, maxx):
        out = list()
        for neighbour in one_hop_ns:
            two_hop_neighbours = self.get_one_hop_neighbours(neighbour, maxx)
            out.extend(two_hop_neighbours)
        return out

    def get_neighbourhood(self, nids: list, maxx: int):
        neighbourhood = []

        for nid in nids:
            one_hop_n = self.get_one_hop_neighbours(nid, maxx)
            one_hop_neighbours = [x[0] for x in one_hop_n]

            two_hop_n = self.get_two_hop_neighbours(nid, one_hop_neighbours, maxx)

            neighbourhood.extend(one_hop_n)
            neighbourhood.extend(two_hop_n)

            neighbourhood = remove_duplicates(neighbourhood)

        return neighbourhood

    def get_dists_to_anchors_property(self, nids: list, node_attr: dict = None):
        if node_attr is None:
            node_attr = self.get_node_attributes(nids)
        ret = {}
        for nid in nids:
            if self._prefix + "distsToAnchors" in node_attr[nid]:
                ret[nid] = ast.literal_eval(
                    node_attr[nid][self._prefix + "distsToAnchors"]
                )
            else:
                ret[nid] = None
        return ret

    def get_relation_context(self, nids: list, node_attr: dict = None):
        if node_attr is None:
            node_attr = self.get_node_attributes(nids)
        ret = {}
        for nid in nids:
            if self._prefix + "relationContext" in node_attr[nid]:
                ret[nid] = ast.literal_eval(
                    node_attr[nid][self._prefix + "relationContext"]
                )
            else:
                ret[nid] = None
        return ret

    def get_neighbours_edge_types(self, nid, k, flow="both", unique=True):
        if flow == "both":
            query = f"MATCH ()-[relation]-(n) WHERE n.{self.nodeUID} = $nid RETURN relation.{self.relUID} AS edge_type"
        elif flow == "incoming":
            query = f"MATCH ()-[incoming]->(n) WHERE n.{self.nodeUID} = $nid RETURN incoming.{self.relUID} AS edge_type"
        elif flow == "outgoing":
            query = f"MATCH (n)-[outcoming]->() WHERE n.{self.nodeUID} = $nid RETURN outcoming.{self.relUID} AS edge_type"

        records = self.execute_query(query, nid=nid)

        edge_types = []

        for r in records:
            edge_type = r.data()["edge_type"]
            edge_types.append(int(edge_type))

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
        records = self.execute_query(
            f"MATCH (anchorNode), (n) WHERE n.{self.nodeUID} = $nid AND anchorNode.{self.nodeUID} IN $anchorIds "
            f"RETURN anchorNode.{self.nodeUID} AS anchorNodeId, "
            "CASE WHEN anchorNode = n THEN 0 "
            "ELSE length(shortestPath((anchorNode)-[*]-(n))) END AS distance",
            nid=nid,
            anchorIds=anchors,
        )

        dists = {
            nid: (
                record.data()["anchorNodeId"],
                (
                    record.data()["distance"]
                    if record.data()["distance"] is not None
                    else "[DISCONNECTED]"
                ),
            )
            for record in records
        }
        return dists

    def write_relation_context(self, relation_context):
        for nid in tqdm(relation_context.keys()):
            if len(relation_context[nid]) > 0:
                self.set_property([nid], "relationContext", relation_context[nid])

    def multi_hop_query_retrieval(self, nid: int, rids: list):
        # Construct the dynamic Cypher query
        match_clauses = []
        for i, rid in enumerate(rids):
            match_clauses.append(f"-[r{i}:Relation {{type: $r{i}}}]->(n{i+1}:Entity)")

        match_query = f"MATCH (n0:Entity {{id: $nid}}) {''.join(match_clauses)} RETURN DISTINCT n{len(rids)};"

        # Construct the parameters for the query
        query_parameters = {f"r{i}": rid for i, rid in enumerate(rids)}
        query_parameters["nid"] = nid

        # Execute the query
        records = self.execute_query(match_query, **query_parameters)

        # Extract the results
        query_results = [r.data()[f"n{len(rids)}"]["id"] for r in records]
        return query_results

    def sample_queries(self, hops: int, batch_size: int, negative_sample_size: int):
        queries, tails, negative_tails, subsampling_weight = [], [], [], []
        while len(queries) < batch_size:
            start_node_ids = self.get_random_node_ids(batch_size * 2)
            involved_nodes_prev = {}
            for x in start_node_ids:
                involved_nodes_prev[x] = (x,)
            nodes_to_expand = start_node_ids
            for i in range(hops):
                next_nodes = []
                node_neighbours = self.get_one_hop_neighbours(
                    nodes_to_expand, flow="outgoing", ret_rels=True
                )
                for neighbours, node in zip(node_neighbours, nodes_to_expand):
                    node_prev_list = list(involved_nodes_prev[node])
                    neighbours_filtered = [
                        x for x in neighbours if x[-1] not in involved_nodes_prev
                    ]
                    neighbours_filtered = random.sample(
                        neighbours_filtered, k=min(20, len(neighbours_filtered))
                    )
                    for neighbour in neighbours_filtered:
                        next_nodes.append(neighbour[-1])
                        involved_nodes_prev[neighbour[-1]] = tuple(
                            node_prev_list + [neighbour[1]]
                        )
                nodes_to_expand = random.sample(
                    next_nodes, min(batch_size * 2, len(next_nodes))
                )
            k = min(batch_size, len(nodes_to_expand))
            # nodes_to_expand = random.choices(nodes_to_expand, k=k)
            nodes_to_expand = random.sample(nodes_to_expand, k)
            for node in nodes_to_expand:
                queries.append(list(involved_nodes_prev[node]))
                tails.append([node])

        for tail in tails:
            all_nodes = self.get_all_node_ids()
            while True:
                negative_tail = random.sample(all_nodes, negative_sample_size)
                if tail[0] not in negative_tail:
                    break
            negative_tails.append(negative_tail)
            subsampling_weight.append(np.sqrt(1 / (1 + 4)))

        return queries, tails, negative_tails, subsampling_weight

    def sample_queries_neg_relation(self, batch_size: int, negative_sample_size: int):
        queries, tails, negative_relations, subsampling_weight = [], [], [], []
        start_node_ids = self.get_random_node_ids(batch_size * 2)
        start_node_ids_p = 0
        while len(queries) < batch_size:
            start_node_id = start_node_ids[start_node_ids_p]
            start_node_ids_p += 1
            if start_node_ids_p >= len(start_node_ids):
                start_node_ids = self.get_random_node_ids(batch_size * 2)
                start_node_ids_p = 0
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
            WHERE n.{self.nodeUID} = {query[0]} AND n1.{self.nodeUID} = {tails[i][0]}
            RETURN r.{self.nodeUID}
            """
            records = self.execute_query(db_query)
            true_rels = [record.data()[f"r.{self.nodeUID}"] for record in records]
            negative_relations.append(
                negative_sampling(
                    negative_sample_size, self.get_all_relations(), true_rels
                )
            )
            subsampling_weight.append(np.sqrt(1 / (len(true_rels) + 4)))

        return queries, tails, negative_relations, subsampling_weight

    def sample_sub_graph(self, num_entities: int):
        query = f"""
            MATCH (n) WITH n ORDER BY rand() LIMIT {num_entities}
            WITH collect(n) as sampledNodes
            UNWIND sampledNodes as n
            MATCH (n)-[r]-(m) WHERE m IN sampledNodes
            RETURN DISTINCT n.{self.nodeUID}, r.{self.relUID}, m.{self.nodeUID}
            """
        records = self.execute_query(query)

        triples = []
        for record in records:
            record = record.data()
            triples.append((record[f"n.{self.nodeUID}"], record[f"r.{self.relUID}"], record[f"m.{self.nodeUID}"]))
        return triples

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
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