import os
import pickle
import torch
import logging
from torch.utils.data import Dataset, DataLoader


class FileBasedDataIterator:
    """
    Data iterator that loads calibration data from saved pickle files.
    Compatible with the calibration data format created by calibration_sampler.py
    """
    
    def __init__(self, calibration_data_path, batch_size, device, hop_level=3, graph_data=None):
        self.calibration_data_path = calibration_data_path
        self.batch_size = batch_size
        self.device = device
        self.hop_level = hop_level  # Which hop level to load (1, 2, or 3)
        
        # Load the saved calibration data with cap
        self.queries = self._load_queries()
        self.answers = self._load_answers(self.queries)
        
        # Load or use provided graph_data
        if graph_data is not None:
            # Use provided graph_data (from database)
            self.graph_data = graph_data
            logging.info("Using provided graph_data from database")
        else:
            # Try to load from file
            self.graph_data = self._load_graph_data()
        
        # Validate graph_data is available
        if self.graph_data is None:
            raise ValueError(
                f"graph_data is required but not available. "
                f"Either provide graph_data parameter or ensure graph_data.pkl exists at {calibration_data_path}"
            )
        
        # Load intermediate hop data if available
        self.intermediate_hop_data = self._load_intermediate_hop_data()
        
        # Create dataset and dataloader
        self.dataset = CalibrationDataset(self.queries, self.answers, self.graph_data, self.intermediate_hop_data)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        self.iterator = iter(self.dataloader)
    
    def _load_queries(self):
        """Load queries from pickle file"""
        queries_path = os.path.join(self.calibration_data_path, "queries.pkl")
        with open(queries_path, "rb") as f:
            queries = pickle.load(f)
            return queries
            # # TODO: Remove this query cap - this was only for testing
            # CAP = 100
            # if len(queries) > CAP:
            #     logging.warning(f"QUERY CAP ACTIVE: Limiting queries from {len(queries)} to {CAP} for testing purposes. Remove cap before production use!")
            # return queries[:CAP]
    
    def _load_answers(self, queries):
        """Load answers from all hop levels and organize by hop"""
        answers_by_hop = {}
        intermediate_dir = os.path.join(self.calibration_data_path, "intermediate_hops")
        
        # Load answers for all available hop levels
        for hop in range(1, 4):  # Check for hop_1, hop_2, hop_3
            hop_dir = os.path.join(intermediate_dir, f"hop_{hop}")
            if os.path.exists(hop_dir):
                answers_path = os.path.join(hop_dir, "answers.pkl")
                try:
                    with open(answers_path, "rb") as f:
                        hop_answers = pickle.load(f)
                        # Only keep answers for the capped queries
                        capped_hop_answers = {query: hop_answers.get(query, []) for query in queries}
                        answers_by_hop[hop] = capped_hop_answers
                        logging.info(f"Loaded answers for hop {hop}: {len(capped_hop_answers)} queries")
                except Exception as e:
                    logging.warning(f"Failed to load answers for hop {hop}: {e}")
        
        return answers_by_hop
    
    def _load_graph_data(self):
        """
        Load graph data from pickle file if it exists.
        
        graph_data is a PyTorch Geometric Data object representing the full knowledge graph.
        It's needed by ULTRA for inference but is expensive to build from Neo4j (~minutes).
        
        The graph is the SAME for all query types (3p, 2u, etc.), so we:
        1. First check the shared location: calibration_data/graph_data.pkl
        2. Fall back to query-specific location for backward compatibility
        3. If neither exists, will be provided from database by caller
        """
        # Try shared location first (preferred)
        parent_dir = os.path.dirname(self.calibration_data_path)
        shared_graph_path = os.path.join(parent_dir, "graph_data.pkl")
        if os.path.exists(shared_graph_path):
            with open(shared_graph_path, "rb") as f:
                logging.info(f"Loading shared cached graph_data from {shared_graph_path}")
                return pickle.load(f)
        
        # Fall back to query-specific location (backward compatibility)
        local_graph_path = os.path.join(self.calibration_data_path, "graph_data.pkl")
        if os.path.exists(local_graph_path):
            with open(local_graph_path, "rb") as f:
                logging.info(f"Loading cached graph_data from {local_graph_path}")
                return pickle.load(f)
        
        logging.info(f"graph_data.pkl not cached. Will use graph_data from database.")
        return None
    
    def _load_intermediate_hop_data(self):
        """Load intermediate hop data for all available hop levels"""
        intermediate_data = {}
        intermediate_dir = os.path.join(self.calibration_data_path, "intermediate_hops")
        
        if not os.path.exists(intermediate_dir):
            logging.warning("No intermediate_hops directory found, skipping intermediate hop data")
            return intermediate_data
        
        # Load data for all available hop levels
        for hop in range(1, 4):  # Check for hop_1, hop_2, hop_3
            hop_dir = os.path.join(intermediate_dir, f"hop_{hop}")
            if os.path.exists(hop_dir):
                try:
                    # Load queries
                    queries_path = os.path.join(hop_dir, "queries.pkl")
                    with open(queries_path, "rb") as f:
                        queries = pickle.load(f)
                    
                    # Load answers
                    answers_path = os.path.join(hop_dir, "answers.pkl")
                    with open(answers_path, "rb") as f:
                        answers = pickle.load(f)
                    
                    intermediate_data[hop] = {
                        "queries": queries,
                        "answers": answers
                    }
                    logging.info(f"Loaded intermediate hop {hop} data: {len(queries)} queries")
                except Exception as e:
                    logging.warning(f"Failed to load intermediate hop {hop} data: {e}")
        
        return intermediate_data
    
    def _collate_fn(self, batch):
        """Collate function for batching data"""
        queries = torch.stack([item[0] for item in batch])
        answers_by_hop = [item[1] for item in batch]  # List of {hop: answers} dicts
        graph_data = batch[0][2]  # All items have the same graph_data
        intermediate_hop_data = batch[0][3]  # All items have the same intermediate_hop_data
        
        return queries, answers_by_hop, graph_data, intermediate_hop_data
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator for next epoch
            self.iterator = iter(self.dataloader)
            raise StopIteration
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_answers_for_hop(self, query, hop_level):
        """Get answers for a specific query and hop level"""
        if hop_level in self.answers:
            answer = self.answers[hop_level].get(query, [])
            # Convert from set to list if needed
            if isinstance(answer, set):
                return list(answer)
            return answer
        return []
    
    def get_all_hop_answers(self, query):
        """Get answers for a specific query across all hop levels"""
        all_answers = {}
        for hop_level, hop_answers in self.answers.items():
            answer = hop_answers.get(query, [])
            if isinstance(answer, set):
                answer = list(answer)
            all_answers[hop_level] = answer
        return all_answers
    
    def get_available_hop_levels(self):
        """Get list of available hop levels"""
        return list(self.answers.keys())
    
    def get_hop_statistics(self):
        """Get statistics for all available hop levels"""
        stats = {}
        for hop_level, hop_answers in self.answers.items():
            answer_counts = [len(answers) for answers in hop_answers.values()]
            if answer_counts:
                stats[hop_level] = {
                    "num_queries": len(hop_answers),
                    "total_answers": sum(answer_counts),
                    "avg_answers_per_query": sum(answer_counts) / len(answer_counts),
                    "min_answers": min(answer_counts),
                    "max_answers": max(answer_counts)
                }
        return stats
    
    def load_hop_level(self, hop_level):
        """Load data for a specific hop level (1, 2, or 3)"""
        if hop_level in self.answers:
            return {
                "queries": self.queries,
                "answers": self.answers[hop_level]
            }
        else:
            logging.warning(f"Hop level {hop_level} not available. Available levels: {self.get_available_hop_levels()}")
            return None


class CalibrationDataset(Dataset):
    """Dataset class for calibration data"""
    
    def __init__(self, queries, answers, graph_data, intermediate_hop_data=None):
        self.queries = queries
        self.answers = answers
        self.graph_data = graph_data
        self.intermediate_hop_data = intermediate_hop_data or {}
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        
        # Get answers for all hop levels
        answers_by_hop = {}
        for hop_level, hop_answers in self.answers.items():
            answer = hop_answers.get(query, [])
            # Convert answer from set to list if needed (QueryGenerator stores answers as sets)
            if isinstance(answer, set):
                answer = list(answer)
            answers_by_hop[hop_level] = answer
        
        # Convert query to tensor format expected by the model
        # Handle different query formats:
        # - 3p: (entity, (relation1, relation2, relation3))
        # - 2u: ((anchor1, rel1), (anchor2, rel2), "2u")
        # - 2ip: ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
        
        if isinstance(query, tuple):
            # Check if it's a 2u query: ((anchor1, rel1), (anchor2, rel2), "2u")
            if len(query) == 3 and isinstance(query[2], str) and query[2] == "2u":
                (anchor1, rel1), (anchor2, rel2), _ = query
                query_tensor = torch.tensor([anchor1, rel1, anchor2, rel2], dtype=torch.long)
            # Check if it's a 2ip query: ((anchor1, rel1), (anchor2, rel2), rel3, "2ip")
            elif len(query) == 4 and isinstance(query[3], str) and query[3] == "2ip":
                (anchor1, rel1), (anchor2, rel2), rel3, _ = query
                query_tensor = torch.tensor([anchor1, rel1, anchor2, rel2, rel3], dtype=torch.long)
            # 3p query: (entity, (relation1, relation2, relation3))
            elif len(query) == 2:
                entity, relations = query
                if isinstance(relations, tuple):
                    # Flatten to [entity, rel1, rel2, rel3] format
                    query_list = [entity] + list(relations)
                    query_tensor = torch.tensor(query_list, dtype=torch.long)
                else:
                    # Single relation case
                    query_tensor = torch.tensor([entity, relations], dtype=torch.long)
            else:
                # Fallback for other formats
                query_tensor = torch.tensor(query, dtype=torch.long)
        else:
            # Fallback for non-tuple formats
            query_tensor = torch.tensor(query, dtype=torch.long)
        
        return query_tensor, answers_by_hop, self.graph_data, self.intermediate_hop_data
