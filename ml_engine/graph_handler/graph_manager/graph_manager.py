from collections import defaultdict
import logging
from torch.utils.data import Dataset
import multiprocessing
import multiprocess
from . import GraphSampler


class GraphManager(object):
    def __init__(
        self,
        args,
        save_path,
        db_controller,
        device,
        mode,
        dataset: Dataset,
        non_overlap_dataset,
    ):
        self.mode = mode
        self.times_per_round = args.times_per_round
        self.device = device
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.next_data_ready_event = multiprocessing.Event()
        self.next_data_loaded_event = multiprocessing.Event()
        self.finished = multiprocessing.Event()
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
        train_min_hops = args.train_min_hops if hasattr(args, "train_min_hops") else 1
        train_max_hops = args.train_max_hops if hasattr(args, "train_max_hops") else 1
        if mode == "val":
            size_ratio = args.gen_val_num
        elif mode == "calib":
            assert hasattr(args, "conformal_prediction")
            size_ratio = args.conformal_prediction.calib_size
        else:
            size_ratio = 1
        logging.info(f"Mode is {mode}")
        logging.info(f"Size ratio is {size_ratio}")
        self.graph_sampler = GraphSampler(
            save_path,
            args.max_num_entity,
            db_controller,
            train_min_hops,
            train_max_hops,
            args.max_num_ans,
            gen_num,
            size_ratio,
            mode,
            args.keep_history,
            args.query_generator_log_ratio,
            non_overlap_dataset,
        )
        self.graph_round_to_load = 0

        if self.mode == "train":
            self.times_per_rounds = defaultdict(int)

        # automatically prepare data for the next round in another process
        self.multi_process_manager = multiprocess.Manager()
        self.data_meta_info = self.multi_process_manager.dict()
        self.data_meta_info_round = self.multi_process_manager.Value("i", 0)
        self.data_preparation_process = multiprocess.Process(
            target=self.prepare_data,
            args=(self.data_meta_info, self.data_meta_info_round),
        )
        self.data_preparation_process.start()

    def update_round(self):
        if self.mode == "train":
            self.times_per_rounds[self.graph_round_to_load] += 1
            if self.times_per_rounds[self.graph_round_to_load] == self.times_per_round:
                self.load_generated_data()

    def prepare_data(self, data_meta_info, data_meta_info_round):
        while True:
            self.next_data_ready_event.clear()
            try:
                self.graph_sampler.sample_graph_and_queries(
                    data_meta_info, data_meta_info_round
                )
            except BrokenPipeError as e:
                logging.error(f"BrokenPipeError in the data preparation process: {e}")
            self.next_data_ready_event.set()
            if self.mode != "train":
                return
            while not self.next_data_loaded_event.wait(8):
                logging.debug("Waiting for next data to load")
                if self.finished.is_set():
                    return
            self.next_data_loaded_event.clear()

    def load_generated_data(self):
        logging.info("Waiting for the next data to be ready...")
        self.next_data_ready_event.wait()

        logging.info(f"Loading the next data round {self.graph_round_to_load}...")
        assert self.data_meta_info_round.value == self.graph_round_to_load

        if self.mode == "train":
            self.edge_index = self.dataset.load_generated_data(
                self.data_meta_info, self.mode
            )
        else:
            self.edge_index, overlap_set = self.dataset.load_generated_data(
                self.data_meta_info, self.mode
            )
        self.edge_index = self.edge_index.to(self.device)

        logging.info(f"Round {self.graph_round_to_load} data loaded.")
        self.graph_round_to_load += 1
        self.next_data_loaded_event.set()

        if self.mode != "train":
            return overlap_set
