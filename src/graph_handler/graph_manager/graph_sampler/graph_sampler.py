import pickle
import os
import copy
import os.path as osp
import numpy as np
import time
import logging
from .query_generator import QueryGenerator
from ...backend_db import AbstractDBController


class GraphSampler(object):
    def __init__(
        self,
        work_dir: str,
        max_num_entity: int,
        db_controller: AbstractDBController,
        min_hops: int,
        max_hops: int,
        max_num_ans: int,
        gen_num: dict,
        size_ratio: float,
        mode: str,
        keep_history: bool,
        query_generator_log_ratio: float,
        non_overlap_dataset: set,
    ):
        self.work_dir = work_dir
        self.max_num_entity = max_num_entity
        self.db_controller = db_controller
        self.query_generator = QueryGenerator(
            min_hops,
            max_hops,
            max_num_ans,
            gen_num,
            size_ratio,
            mode,
            query_generator_log_ratio,
            non_overlap_dataset,
        )
        self.mode = mode
        self.keep_history = keep_history  # This is not used yet
        self.next_sample_round = 0
        self.data_meta_infos = {}

    def sample_graph_and_queries(self, data_meta_info, data_meta_info_round):
        round_num = self.next_sample_round
        self.next_sample_round += 1
        save_path_prefix = osp.join(self.work_dir, f"{self.mode}_round_{round_num}")
        os.makedirs(save_path_prefix, exist_ok=True)
        graph_save_path = osp.join(save_path_prefix, "graph.pkl")
        relabel_mappings, [graph_save_path, relabel_mappings_save_path] = (
            self.sample_graph(round_num, save_path_prefix)
        )
        data_meta_info["triples"] = graph_save_path
        data_meta_info["relabel_mappings"] = relabel_mappings_save_path

        logging.info("Start generating queries, round %d" % round_num)
        query_files_generated = self.query_generator.generate_queries(
            graph_save_path, save_path_prefix, relabel_mappings["num2idx"]
        )
        data_meta_info["queries"] = query_files_generated

        unique_entities = list(relabel_mappings["idx2num"].keys())
        node_attr = self.db_controller.get_node_attributes(unique_entities)
        ancs_and_dists = self.db_controller.get_dists_to_anchors_property(
            unique_entities, node_attr
        )
        relation_context = self.db_controller.get_relation_context(
            unique_entities, node_attr
        )
        # relabel the entities for ancs_and_dists and relation_context
        ancs_and_dists = {
            relabel_mappings["idx2num"][k]: v for k, v in ancs_and_dists.items()
        }
        relation_context = {
            relabel_mappings["idx2num"][k]: v for k, v in relation_context.items()
        }
        ancs_and_dists_save_path = osp.join(save_path_prefix, "ancs_and_dists.pkl")
        relation_context_save_path = osp.join(save_path_prefix, "relation_context.pkl")
        pickle.dump(ancs_and_dists, open(ancs_and_dists_save_path, "wb"))
        pickle.dump(relation_context, open(relation_context_save_path, "wb"))
        data_meta_info["ancs_and_dists"] = ancs_and_dists_save_path
        data_meta_info["relation_context"] = relation_context_save_path

        self.data_meta_infos[round_num] = copy.deepcopy(data_meta_info)
        data_meta_info_round.value = round_num

    def sample_graph(self, round_num, save_path_prefix):
        logging.info("Start sampling graph, round %d" % round_num)
        tic = time.time()

        triples = self.db_controller.sample_sub_graph(self.max_num_entity)

        unique_entities = set()
        for triple in triples:
            unique_entities.add(triple[0])
            unique_entities.add(triple[2])
        unique_entities = list(unique_entities)

        logging.info("Graph sampling results for round %d" % round_num)
        logging.info(f"Number of specified entities: {self.max_num_entity}")
        logging.info(f"Number of unique nodes: {len(unique_entities)}")
        logging.info(f"Number of triples: {len(triples)}")
        logging.info(f"Time taken to process records: {time.time() - tic:.2f} seconds")

        idx2num = {idx: i for i, idx in enumerate(unique_entities)}
        num2idx = {i: idx for i, idx in enumerate(unique_entities)}
        relabel_mappings = {"idx2num": idx2num, "num2idx": num2idx}

        for i in range(len(triples)):
            triples[i] = list(triples[i])
            triples[i][0] = idx2num[triples[i][0]]
            triples[i][2] = idx2num[triples[i][2]]

        logging.info("Finish relabeling nodes")

        # save the triples using pickle
        tic = time.time()
        triples_np = np.array(triples)
        graph_save_path = osp.join(save_path_prefix, "graph.pkl")
        pickle.dump(triples_np, open(graph_save_path, "wb"))
        logging.info(f"Succesfully saved the sampled triples to {graph_save_path}")
        relabel_mappings_save_path = osp.join(save_path_prefix, "relabel_mappings.pkl")
        pickle.dump(relabel_mappings, open(relabel_mappings_save_path, "wb"))
        logging.info(
            f"Succesfully saved the unique entities to {relabel_mappings_save_path}"
        )
        return relabel_mappings, [graph_save_path, relabel_mappings_save_path]
