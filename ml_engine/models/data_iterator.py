import torch
from torch.utils.data import Dataset, DataLoader
from graph_handler import GraphManager
import numpy as np
from abc import ABC, abstractmethod


class DataIteratorABC(ABC):
    def __init__(
        self,
        args,
        save_path,
        db_controller,
        device,
        mode="train",
        non_overlap_dataset=set(),
    ):
        self.step = 0
        self.mode = mode
        self.db_mode = args.db_mode
        self.non_overlap_dataset = non_overlap_dataset

        dataset, batch_size = self.get_dataset(args, db_controller, mode, device)

        if self.db_mode == "sample":
            self.graph_manager = GraphManager(
                args,
                save_path,
                db_controller,
                device,
                mode,
                dataset,
                non_overlap_dataset,
            )
            if self.mode == "train":
                self.graph_manager.load_generated_data()
            else:
                self.non_overlap_dataset = self.graph_manager.load_generated_data()

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )
        self.iterator = self.one_shot_iterator(self.data_loader)

    def __del__(self):
        if self.db_mode == "sample":
            self.graph_manager.finished.set()
            self.graph_manager.multi_process_manager.shutdown()

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

    @abstractmethod
    def get_dataset(self, args, db_controller, mode):
        pass

    def one_shot_iterator(self, dataloader):
        while True:
            for data in dataloader:
                if self.db_mode == "sample":
                    data = self.sample_graph_data_postprocess(data)
                yield data
            if self.db_mode == "sample":
                self.graph_manager.update_round()
            if self.mode != "train":
                break

    @abstractmethod
    def sample_graph_data_postprocess(self, data):
        pass
