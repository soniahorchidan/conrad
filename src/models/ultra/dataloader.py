import os
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from graph_handler import AbstractDBController, QueryGenerator
import pickle
from torch_geometric.data import Data
from utils import get_graph


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def collate_list(x):
    return torch.concat([torch.from_numpy(np.array(_)) for _ in x], dim=0)


class ValDataset(Dataset):
    def __init__(
        self,
        min_hops: int,
        max_hops: int,
        queries: list,
        answers: dict,
        graph_data: Data,
    ):
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.hop_range = list(range(min_hops, max_hops + 1))
        self.queries = queries
        self.answers = answers
        self.graph_data = graph_data
        self.num_nodes = graph_data.num_nodes

    def __len__(self):
        return len(self.queries)

    def get_sample(self, idx: int):
        query = self.queries[idx]
        answers = list(self.answers[query])
        query = [[query[0]] + list(query[1])]

        return query, answers

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return sample

    def collate_fn(self, data):
        query = collate_list([_[0] for _ in data]).squeeze(-1)
        answers = [_[1] for _ in data]
        return query, answers, self.graph_data


class CalibDataset(Dataset):
    def __init__(
        self,
        min_hops: int,
        max_hops: int,
        queries: list,
        answers: dict,
        graph_data: Data,
    ):
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.hop_range = list(range(min_hops, max_hops + 1))
        self.queries = queries
        self.answers = answers
        self.graph_data = graph_data
        self.num_nodes = graph_data.num_nodes

    def __len__(self):
        return len(self.queries)

    def get_sample(self, idx: int):
        query = self.queries[idx]
        answers = list(self.answers[query])
        query = [[query[0]] + list(query[1])]

        return query, answers

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return sample

    def collate_fn(self, data):
        query = collate_list([_[0] for _ in data]).squeeze(-1)
        answers = [_[1] for _ in data]

        return query, answers, self.graph_data


class DataIterator(object):
    def __init__(
        self,
        args,
        save_path,
        db_controller: AbstractDBController,
        device,
        mode="train",
        non_overlap_dataset=set(),
    ):
        self.step = 0
        self.mode = mode
        self.device = device
        self.save_path = save_path
        self.non_overlap_dataset = non_overlap_dataset

        dataset, batch_size = self.get_dataset(args, db_controller, mode)

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )
        self.iterator = self.one_shot_iterator(self.data_loader)

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_loader)

    def reset(self):
        assert self.mode != "train"
        self.step = 0
        self.iterator = self.one_shot_iterator(self.data_loader)

    def get_non_overlap_dataset(self):
        return self.non_overlap_dataset

    def one_shot_iterator(self, dataloader):
        while True:
            for data in dataloader:
                yield data
            if self.mode != "train":
                break

    def get_dataset(self, args, db_controller, mode):
        if mode == "train":
            raise NotImplementedError("Training mode is not supported in inference-only mode.")
        elif mode == "val":
            Dataset = ValDataset
        elif mode == "calib":
            Dataset = CalibDataset
        else:
            raise NotImplementedError

        batch_size = args.batch_size

        graph_data = self.get_graph_data(db_controller, self.device)
        queries, answers, min_hops, max_hops = self.get_queries(
            args, graph_data, batch_size, mode, self.non_overlap_dataset, self.save_path
        )

        dataset_params = {
            "min_hops": min_hops,
            "max_hops": max_hops,
            "queries": queries,
            "answers": answers,
            "graph_data": graph_data,
        }
        dataset = Dataset(**dataset_params)
        return dataset, batch_size

    @staticmethod
    def get_graph_data(db_controller, device):
        graph_data = get_graph(
            db_controller, device, augment_inverse_edges=True, relation_graph=True
        )
        return graph_data

    @staticmethod
    def get_queries(args, graph_data, batch_size, mode, non_overlap_dataset, save_path):
        # generating queries
        gen_num = (
            {
                int(k): float(v)
                for k, v in zip(
                    range(args.train_min_hops + 1, args.train_max_hops + 1),
                    args.gen_num.split(","),
                )
            }
            if hasattr(args, "gen_num")
            else {}
        )

        # print("DEBUG:: USING ARGS=", args)

        # if hasattr(args, "calib_min_hops"):
        #     min_hops = args.calib_min_hops
        # else:
        #     min_hops = args.train_min_hops if hasattr(args, "train_min_hops") else 1

        # if hasattr(args, "calib_max_hops"):
        #     max_hops = args.calib_max_hops
        # else:
        #     max_hops = args.train_max_hops if hasattr(args, "train_max_hops") else 1

        min_hops = 3
        max_hops = 3
        print("DEBUG:: min_hops, max_hops:", min_hops, max_hops)

        if mode == "val":
            size_ratio = args.gen_val_num
        elif mode == "calib":
            # Using hardcoded size ratio for calibration mode
            size_ratio = 0.001
            logging.info(f"Using hardcoded size ratio calib mode to {size_ratio}!!")

        else:
            size_ratio = 1
        logging.info(f"Generating queries, Mode is {mode}, Size ratio is {size_ratio}")

        query_generator = QueryGenerator(
            min_hops=min_hops,
            max_hops=max_hops,
            max_num_ans=args.max_num_ans,
            gen_num=gen_num,
            size_ratio=size_ratio,
            mode=mode,
            query_generator_log_ratio=args.query_generator_log_ratio,
            non_overlap_dataset=non_overlap_dataset,
        )
        save_path_prefix = os.path.join(save_path, mode)
        os.makedirs(save_path_prefix, exist_ok=True)
        query_files = query_generator.generate_queries(
            graph_data, save_path_prefix, None
        )

        queries = []
        answers = {}
        for hop in range(min_hops, max_hops + 1):
            queries_hop = pickle.load(open(query_files[hop][0], "rb"))
            queries_hop = list(queries_hop[list(queries_hop.keys())[0]])
            queries_hop = queries_hop[len(queries_hop) % batch_size :]
            queries.extend(queries_hop)
            answers_hop = pickle.load(open(query_files[hop][1], "rb"))
            answers.update(answers_hop)
        return queries, answers, min_hops, max_hops
