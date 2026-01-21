import pickle
import os.path as osp
import numpy as np
from collections import defaultdict
import random
from copy import deepcopy
import logging
from torch_geometric.data import Data
import tqdm


class QueryGenerator(object):
    def __init__(
        self,
        min_hops,
        max_hops,
        max_num_ans,
        gen_num,
        size_ratio,
        mode,
        query_generator_log_ratio,
        non_overlap_dataset,
    ):
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.max_num_ans = max_num_ans
        self.gen_num = gen_num
        self.size_ratio = size_ratio
        self.mode = mode
        self.log_ratio = query_generator_log_ratio
        self.non_overlap_dataset = non_overlap_dataset

    def generate_queries(self, graph_data, save_path_prefix, num2idx):
        self.ent_in, self.ent_out, counter = self.construct_graph(graph_data)

        files_generated = {}
        for num_hop in range(self.min_hops, self.max_hops + 1):
            logging.info("Start generating queries with %d hops" % num_hop)
            save_path = osp.join(save_path_prefix, f"{num_hop}_hops")
            if num_hop == 1:
                files = self.gen_links(save_path, num2idx)
            else:
                # Use the gen_num from the parameter, not hardcoded calculation
                if num_hop in self.gen_num:
                    gen_num = int(self.gen_num[num_hop])
                else:
                    # TODO(sonia): simplify this. 
                    gen_num = int(counter * 0.0001)  # fallback to old calculation
                    logging.warning(f"No gen_num specified for {num_hop} hops, using fallback: {gen_num}")
                
                # Apply size_ratio if not in train mode
                if self.mode != "train":
                    gen_num = max(int(gen_num * self.size_ratio), 100)
                    logging.info(
                        f"Sample {gen_num} {num_hop}-hop queries for {self.mode} (size_ratio={self.size_ratio})"
                    )
                else:
                    logging.info(f"Sample {gen_num} {num_hop}-hop queries for {self.mode}")
                
                files = self.gen_multi_hops(num_hop, gen_num, save_path, num2idx)
            files_generated[num_hop] = files
            logging.info("Finish generating queries with %d hops" % num_hop)
        return files_generated

    def fill_query(self, query_structure, answer):
        assert type(query_structure[-1]) == list
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            found = False
            for j in range(40):
                # Check if the answer entity has any incoming edges
                if answer not in self.ent_in or len(self.ent_in[answer]) == 0:
                    return True  # Skip this query - entity has no incoming edges
                
                r_tmp = random.sample(list(self.ent_in[answer].keys()), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            
            # Check if the relation has any incoming entities
            if r not in self.ent_in[answer] or len(self.ent_in[answer][r]) == 0:
                return True  # Skip this query - relation has no incoming entities
            
            answer = random.sample(list(self.ent_in[answer][r]), 1)[0]
        if query_structure[0] == "e":
            query_structure[0] = answer
        else:
            raise NotImplementedError

    def achieve_answer(self, query, check_intermediate_size=False, max_intermediate_size=None):
        """
        Execute a query and return the answer set.
        
        Args:
            query: The query to execute
            check_intermediate_size: If True, check intermediate hop sizes against max_intermediate_size
            max_intermediate_size: Maximum allowed size for intermediate hops (including final answer)
        
        Returns:
            Set of answer entities, or None if any intermediate hop exceeds max_intermediate_size
        """
        assert type(query[-1]) == list
        ent_set = set([query[0]])
        for i in range(len(query[-1])):
            ent_set_traverse = set()
            for ent in ent_set:
                ent_set_traverse = ent_set_traverse.union(
                    self.ent_out[ent][query[-1][i]]
                )
            ent_set = ent_set_traverse
            
            # Check intermediate hop size if requested
            if check_intermediate_size and max_intermediate_size is not None:
                if len(ent_set) > max_intermediate_size:
                    # Return None to signal that this query exceeds the hop size limit
                    return None
        
        return ent_set

    def construct_graph(self, graph_data):
        ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(
            lambda: defaultdict(set)
        )
        counter = 0
        if isinstance(graph_data, str):
            data = pickle.load(open(graph_data, "rb"))
            for e1, rel, e2 in data:
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)
                counter += 1
        elif isinstance(graph_data, Data):
            for e1, rel, e2 in zip(
                graph_data.edge_index[0], graph_data.edge_type, graph_data.edge_index[1]
            ):
                ent_out[e1.item()][rel.item()].add(e2.item())
                ent_in[e2.item()][rel.item()].add(e1.item())
                counter += 1

        return ent_in, ent_out, counter

    def gen_multi_hops(self, num_hop, gen_num, save_path, num2idx):
        query_structure = ["e", ["r"] * num_hop]
        num_sampled, num_try, num_repeat = 0, 0, 0
        num_more_answer, num_broken, num_empty, num_overlap = 0, 0, 0, 0
        num_hop_size_exceeded = 0  # Track queries that exceed intermediate hop size limit
        ans_num = []
        queries, answers = defaultdict(set), defaultdict(set)
        
        # Filter entities that have sufficient connectivity for multi-hop queries
        valid_entities = []
        for entity in self.ent_in.keys():
            if len(self.ent_in[entity]) > 0:  # Entity must have at least one incoming relation
                # Check if entity can support the required number of hops
                can_support_hops = True
                temp_entity = entity
                for hop in range(num_hop):
                    if temp_entity not in self.ent_in or len(self.ent_in[temp_entity]) == 0:
                        can_support_hops = False
                        break
                    # Sample a relation and check if it has incoming entities
                    available_relations = list(self.ent_in[temp_entity].keys())
                    if not available_relations:
                        can_support_hops = False
                        break
                    temp_rel = available_relations[0]  # Just check the first relation
                    if temp_rel not in self.ent_in[temp_entity] or len(self.ent_in[temp_entity][temp_rel]) == 0:
                        can_support_hops = False
                        break
                    # Move to a predecessor entity
                    temp_entity = list(self.ent_in[temp_entity][temp_rel])[0]
                
                if can_support_hops:
                    valid_entities.append(entity)
        
        logging.info(f"Filtered {len(valid_entities)} valid entities out of {len(self.ent_in)} total entities for {num_hop}-hop query generation")
        
        if len(valid_entities) == 0:
            logging.error(f"No valid entities found for {num_hop}-hop query generation!")
            return None, None
        
        # Retry mechanism: continue generating until target number is reached
        max_attempts = gen_num * 10  # Allow up to 10x the target number of attempts
        consecutive_failures = 0
        max_consecutive_failures = 1000  # Stop if we have too many consecutive failures
        
        sample_progress = tqdm.tqdm(
            total=gen_num, desc=f"Generating {num_hop}-hop queries, num_tried: {num_try}"
        )
        
        while num_sampled < gen_num and num_try < max_attempts and consecutive_failures < max_consecutive_failures:
            num_try += 1
            sample_progress.set_description(
                f"Generating {num_hop}-hop queries: {num_sampled}/{gen_num} (tried: {num_try}, failures: {consecutive_failures})"
            )
            empty_query_structure = deepcopy(query_structure)
            answer = random.sample(valid_entities, 1)[0]
            broken_flag = self.fill_query(empty_query_structure, answer)
            if broken_flag:
                num_broken += 1
                consecutive_failures += 1
                continue
            query = empty_query_structure
            if num2idx is not None:
                anchor = num2idx[query[0]]
            else:
                anchor = query[0]
            if list2tuple([anchor, query[1]]) in self.non_overlap_dataset:
                num_overlap += 1
                consecutive_failures += 1
                continue
            # Check intermediate hop sizes during answer computation
            answer_set = self.achieve_answer(query, check_intermediate_size=True, max_intermediate_size=self.max_num_ans)
            if answer_set is None:
                # Query exceeded intermediate hop size limit
                num_hop_size_exceeded += 1
                consecutive_failures += 1
                continue
            if len(answer_set) > self.max_num_ans:
                num_more_answer += 1
                consecutive_failures += 1
                continue
            if len(answer_set) == 0:
                num_empty += 1
                consecutive_failures += 1
                continue
            if list2tuple(query) in queries[list2tuple(query_structure)]:
                num_repeat += 1
                consecutive_failures += 1
                continue
            
            # Successfully generated a valid query
            queries[list2tuple(query_structure)].add(list2tuple(query))
            answers[list2tuple(query)] = answer_set
            num_sampled += 1
            ans_num.append(len(answers[list2tuple(query)]))
            consecutive_failures = 0  # Reset consecutive failures counter
            if num_sampled % int(gen_num * self.log_ratio) == 0:
                logging.info(
                    "Number sampled: {} / {}, Number try: {}, Number repeat: {}, Number more answer: {}, Number broken: {}, Number empty: {}, Number hop size exceeded: {}".format(
                        num_sampled,
                        gen_num,
                        num_try,
                        num_repeat,
                        num_more_answer,
                        num_broken,
                        num_empty,
                        num_hop_size_exceeded,
                    )
                )
                if self.mode == "train":
                    logging.info(
                        "Number repeat in overlapped dataset: {}".format(num_overlap)
                    )
            sample_progress.update(1)
        sample_progress.close()

        # Check if we reached the target
        if num_sampled < gen_num:
            if consecutive_failures >= max_consecutive_failures:
                logging.warning(
                    f"Stopped due to {consecutive_failures} consecutive failures. "
                    f"Generated {num_sampled}/{gen_num} queries."
                )
            elif num_try >= max_attempts:
                logging.warning(
                    f"Stopped due to reaching maximum attempts ({max_attempts}). "
                    f"Generated {num_sampled}/{gen_num} queries."
                )
        else:
            logging.info(f"Successfully generated {num_sampled}/{gen_num} queries!")
            
        logging.info(
            "Number sampled: {}, Number try: {}, Number repeat: {}, Number more answer: {}, Number broken: {}, Number empty: {}, Number hop size exceeded: {}, Number repeat in overlapped dataset: {}".format(
                num_sampled,
                num_try,
                num_repeat,
                num_more_answer,
                num_broken,
                num_empty,
                num_hop_size_exceeded,
                num_overlap,
            )
        )
        logging.info(
            "Answers max: {}, min: {}, mean: {}, std: {}".format(
                np.max(ans_num), np.min(ans_num), np.mean(ans_num), np.std(ans_num)
            )
        )

        query_path = save_path + "_queries.pkl"
        answer_path = save_path + "_answers.pkl"
        with open(query_path, "wb") as f:
            pickle.dump(queries, f)
        with open(answer_path, "wb") as f:
            pickle.dump(answers, f)
        logging.info(
            "Number of queries with %d hops: %d"
            % (num_hop, len(queries[list2tuple(query_structure)]))
        )
        return query_path, answer_path

    def gen_links(self, save_path, num2idx):
        queries, answers = defaultdict(set), defaultdict(set)
        num_more_answer, num_overlap = 0, 0
        for ent in self.ent_out:
            for rel in self.ent_out[ent]:
                if num2idx is not None:
                    anchor = num2idx[ent]
                else:
                    anchor = ent
                if (anchor, (rel,)) in self.non_overlap_dataset:
                    num_overlap += 1
                    continue
                if len(self.ent_out[ent][rel]) <= self.max_num_ans:
                    queries[("e", ("r",))].add((ent, (rel,)))
                    answers[(ent, (rel,))] = self.ent_out[ent][rel]
                else:
                    num_more_answer += 1

        if self.mode == "val" or self.mode == "calib":
            queries_1hop = list(queries[("e", ("r",))])
            queries_1hop = random.sample(
                queries_1hop, max(int(len(queries_1hop) * self.size_ratio), 100)
            )
            queries[("e", ("r",))] = set(queries_1hop)
            answers = {k: v for k, v in answers.items() if k in queries_1hop}
            logging.info(f"Sampled {len(queries_1hop)} 1-hop queries for {self.mode}")
        query_path = save_path + "_queries.pkl"
        answer_path = save_path + "_answers.pkl"
        with open(query_path, "wb") as f:
            pickle.dump(queries, f)
        with open(answer_path, "wb") as f:
            pickle.dump(answers, f)
        logging.info("Number of queries with 1 hop: %d" % len(queries[("e", ("r",))]))
        logging.info(
            "Number of queries with 1 hop more than max_num_ans: %d" % num_more_answer
        )
        if self.mode == "train":
            logging.info(
                "Number of queries repeat in overlapped dataset: %d" % num_overlap
            )
        return query_path, answer_path


class RPQueryGenerator(object):
    def __init__(self):
        pass

    def construct_pairs(self, graph_data):
        ent_pairs = defaultdict(set)
        if isinstance(graph_data, str):
            data = pickle.load(open(graph_data, "rb"))
            for e1, rel, e2 in data:
                ent_pairs[(e1, e2)].add(rel)
        elif isinstance(graph_data, Data):
            for e1, rel, e2 in zip(
                graph_data.edge_index[0], graph_data.edge_type, graph_data.edge_index[1]
            ):
                ent_pairs[(e1.item(), e2.item())].add(rel.item())
        logging.info("Number of entity pairs: %d" % len(ent_pairs))
        return ent_pairs

    def generate_queries(self, graph_data, save_path_prefix):
        logging.info("Start generating relation prediction queries")

        ent_pairs = self.construct_pairs(graph_data)
        save_path = osp.join(save_path_prefix, f"rp_ent_pairs.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(ent_pairs, f)
        file_generated = save_path
        logging.info("Finish generating relation prediction queries")
        return file_generated


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)
