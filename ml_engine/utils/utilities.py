import time
import random
import numpy as np
import torch
import logging
import os
import json
import requests
import hashlib
from argparse import Namespace
from typing import List
import dropbox
import urllib.request
from dateutil import parser
from torch_scatter import scatter_add
from torch_geometric.data import Data


def count_frequency(queries, answers, start=4):
    count = {}
    for query in queries:
        count[query] = start + len(answers[query])
    return count


def remove_duplicates(list_of_lists):
    """
    Remove duplicates from a list of lists.

    Parameters:
        list_of_lists (list): A list containing sublists.

    Returns:
        list: A new list with duplicates removed.
    """
    list_of_tuples = [tuple(x) for x in list_of_lists]
    result = list(set(list_of_tuples))
    result = [list(x) for x in result]
    return result


def calc_params_num(model):
    num_params = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            num_params += np.prod(param.size())
    return num_params


def log_args(args: Namespace, name: str = "Arguments"):
    logging.info(name + ":")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def args2sequence(args: dict) -> List[str]:
    lst = []
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                lst.append(f"--{k}")
        elif isinstance(v, list):
            lst.append(f"--{k}")
            for item in v:
                lst.append(f"{item}")
        else:
            lst.append(f"--{k}={v}")
    return lst


def merge_args(parse_args, json_args_name, json_keys, command_line_args=None):
    """
    Read the arguments from the JSON file and merge them with the command line arguments.
    Command line arguments have higher priority.
    """

    args = parse_args(command_line_args)
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the arguments from the JSON file
    if args.__dict__[json_args_name] is not None:
        with open(args.__dict__[json_args_name], "r") as f:
            json_dict = json.load(f)

        args_list = []
        for key in json_keys:
            args_list += args2sequence(json_dict[key])

        args = vars(args)
        json_args = parse_args(args_list)
        json_args = vars(json_args)

        # command line arguments override the JSON file arguments
        for key in args:
            if args[key] is not None:
                json_args[key] = args[key]

        # convert the paths to absolute paths if they are relative
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for key in json_args:
            if key.endswith("_path") and json_args[key] is not None:
                json_args[key] = os.path.abspath(
                    os.path.join(str(root_dir), json_args[key])
                )
        args = Namespace(**json_args)
    return args


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_component(save_path: str, model_name: str, component_name: str, component):
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
    torch.save(component, os.path.join(save_path, model_name, component_name + ".pth"))


def calculate_dropbox_content_hash(file_path):
    """Calculate the Dropbox content hash of a local file."""
    BLOCK_SIZE = 4 * 1024 * 1024  # 4MB
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(BLOCK_SIZE):
            chunk_hash = hashlib.sha256(chunk).digest()
            hasher.update(chunk_hash)

    return hasher.hexdigest()


def download_pretrained_model(url, save_path, save_name):
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, save_name)
    with urllib.request.urlopen(url) as response, open(save_name, "wb") as out_file:
        data = response.read()  # Read the content of the response
        out_file.write(data)  # Write the content to the file

    return save_name


def download_ckpt(
    dataset, url, save_path, model_name, app_key, app_secret, refresh_token
):
    os.makedirs(save_path, exist_ok=True)

    shared_link = dropbox.files.SharedLink(url=url)
    dbx = dropbox.Dropbox(
        app_key=app_key, app_secret=app_secret, oauth2_refresh_token=refresh_token
    )
    dbx.check_and_refresh_access_token()

    # Get the list of snapshots
    snapshots = []
    logging.info(f"Getting snapshots list from {url}")
    result = dbx.files_list_folder(path=f"/{dataset}", shared_link=shared_link)
    for entry in result.entries:
        # only retrieve folders
        if isinstance(entry, dropbox.files.FolderMetadata):
            snapshots.append(entry.name)

    # Sort the snapshots by date
    snapshots_map = {snapshot: parser.parse(snapshot) for snapshot in snapshots}
    snapshots = sorted(snapshots_map, key=snapshots_map.get, reverse=True)

    # iterate through snapshots to find the latest snapshot for each model
    ckpt_entry = None
    ckpt_date_str = None
    for snapshot in snapshots:
        result = dbx.files_list_folder(path=f"/{dataset}/{snapshot}", shared_link=shared_link)
        for entry in result.entries:
            if not isinstance(entry, dropbox.files.FileMetadata):
                continue
            if entry.name != f"{model_name}.zip":
                continue
            ckpt_entry = entry
            ckpt_date_str = snapshot
            break
        if ckpt_entry is not None:
            break
    assert ckpt_entry is not None, f"No snapshot found in {url}"

    # Download the latest snapshot
    file_name = f"{ckpt_date_str}_{entry.name}"
    file_path = os.path.join(save_path, file_name)
    downloaded = False
    if os.path.exists(file_path):
        logging.info(f"{file_name} already exists in {save_path}")
        if calculate_dropbox_content_hash(file_path) == ckpt_entry.content_hash:
            logging.info(f"Content hash of {file_name} is correct.")
            downloaded = True
        else:
            logging.info(f"Content hash of {file_name} is incorrect.")
    if not downloaded:
        logging.info(f"Downloading {file_name} from Dropbox...")
        dbx.files_download_to_file(file_path, entry.path_lower)

    # Extract the snapshot
    os.system(f"unzip -q -o {file_path} -d {save_path}")
    if os.path.exists(os.path.join(save_path, "__MACOSX")):
        os.system(f"rm -rf {os.path.join(save_path, '__MACOSX')}")
    snapshot_path = file_path.replace(".zip", "")
    # assume the unzipped folder is the model name
    os.system(f"rm -rf {snapshot_path}")
    os.system(f"mv {os.path.join(save_path, model_name)} {snapshot_path}")
    return snapshot_path


def set_remote_urls(args, kwargs, key):
    if hasattr(args, "model_to_infer"):
        attr_name = getattr(args, "model_to_infer")
        if hasattr(args, attr_name):
            attr = getattr(args, attr_name)
            if key in attr:
                kwargs["remote_url"] = attr[key]

    return kwargs


def set_logger(log_path, file_name, print_on_screen):
    """
    Write logs to checkpoint and console
    """

    log_file = os.path.join(log_path, file_name)

    logging.basicConfig(
        format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs
    """
    for metric in metrics:
        logging.info("%s %s at step %d: %f" % (mode, metric, step, metrics[metric]))


def override_config(args):  #! may update here
    """
    Override model and data configuration
    """

    with open(os.path.join(args.init_checkpoint, "config.json"), "r") as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict["data_path"]
    args.model = argparse_dict["model"]
    args.hidden_dim = argparse_dict["hidden_dim"]


def override_all_config(args, exemption=[]):  #! may update here
    """
    Override model and data configuration
    """

    with open(os.path.join(args.init_checkpoint, "config.json"), "r") as fjson:
        argparse_dict = json.load(fjson)

    for key in argparse_dict:
        if key in exemption:
            continue
        args.__dict__[key] = argparse_dict[key]


def np_save(save_path, file_name, arr):
    """
    Save numpy array
    """
    np.save(os.path.join(save_path, file_name), arr)


def calc_metrics(scores: torch.Tensor, answers: list):
    """
    Calculate metrics for a pair of scores and answers.
    Higher scores are better.
    """
    _, indices = torch.sort(scores, descending=True)
    answers_tensor = torch.tensor(answers, device=scores.device)
    rankings = indices.unsqueeze(1) == answers_tensor
    rankings = rankings.nonzero(as_tuple=False)[:, 0] + 1

    hits1 = torch.mean((rankings <= 1).to(torch.float)).item()
    hits3 = torch.mean((rankings <= 3).to(torch.float)).item()
    hits10 = torch.mean((rankings <= 10).to(torch.float)).item()
    mrr = torch.mean(1.0 / rankings.to(torch.float)).item()

    metrics = {"MRR": mrr, "HITS@1": hits1, "HITS@3": hits3, "HITS@10": hits10}
    return metrics


def calc_metrics_batch(scores: torch.Tensor, answers: list):
    """
    Calculate metrics for a batch of scores and answers.
    Higher scores are better.
    """
    batch_size = scores.shape[0]
    indices = torch.argsort(scores, dim=1, descending=True)

    all_rankings = []
    for i in range(batch_size):
        answer_set = torch.tensor(answers[i], device=scores.device)
        rankings = (indices[i].unsqueeze(0) == answer_set.unsqueeze(1)).nonzero(
            as_tuple=False
        )[:, 1] + 1
        all_rankings.append(rankings)

    rankings = torch.cat(all_rankings)

    hits1 = torch.mean((rankings <= 1).to(torch.float)).item()
    hits3 = torch.mean((rankings <= 3).to(torch.float)).item()
    hits10 = torch.mean((rankings <= 10).to(torch.float)).item()
    mrr = torch.mean(1.0 / rankings.to(torch.float)).item()

    metrics = {"MRR": mrr, "HITS@1": hits1, "HITS@3": hits3, "HITS@10": hits10}
    return metrics


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class AverageMeterDict(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = {}
        self.sum = {}
        self.cnt = {}

    def update(self, val_dict, n=1):
        for key in val_dict:
            if key not in self.avg:
                self.avg[key] = 0
                self.sum[key] = 0
                self.cnt[key] = 0
            self.sum[key] += val_dict[key] * n
            self.cnt[key] += n
            self.avg[key] = self.sum[key] / self.cnt[key]


def get_embeddings(vectordb_controller, ids: list, device):
    """
    Retrieve embeddings for a given query from a vector database.

    Args:
        vectordb_controller (AbstractVectorDBController): A controller object for the vector database.
        ids (list[int]): A list of integers representing the query for which embeddings are to be retrieved.
        device (torch.device): The device on which the embeddings should be stored.

    Returns:
        list[float]: A list of float values representing the embeddings retrieved from the vector database.

    Raises:
        TypeError: If the provided query is of an incompatible type or if there is an issue with the database read operation.
    """

    try:
        results = vectordb_controller.read(ids)
        results = torch.tensor(results, dtype=torch.float32).to(
            device
        )  # mps (the macOS version of cuda) does not support float64
        return results
    except Exception as e:
        logging.error(
            f"Error occurred while reading embeddings from the vector database: {e}",
            exc_info=True,
        )
    return []


def get_graph(db_controller, device, augment_inverse_edges=True, relation_graph=True):

    graph_triplets = db_controller.get_whole_graph()

    # if augment_inverse_edges:
    #     logging.info("Reverse edges are added to the graph")
    #     num_relations = db_controller.get_num_relations()
    #     graph_triplets += [(t[2], t[1] + num_relations, t[0]) for t in graph_triplets]
    #     num_relations *= 2
    # else:
    num_relations = db_controller.get_num_relations()

    logging.info(f"Number of edges in the graph: {len(graph_triplets)}")
    logging.info(f"Number of relations in the graph: {num_relations}")

    train_edges = torch.tensor(
        [[t[0], t[2]] for t in graph_triplets], dtype=torch.long, device=device
    ).t()
    train_edge_types = torch.tensor(
        [t[1] for t in graph_triplets], dtype=torch.long, device=device
    )

    num_nodes = len(db_controller.get_all_node_ids())

    # The 'inverse_rel_plus_one' property is needed for traversal dropout to determine the way of deriving inverse edges
    # In BetaE datasets, inv_rel = direct_rel + 1, but in inducitve datasets it is inv_rel = direct_rel + num_relations
    graph_data = Data(
        edge_index=train_edges,
        edge_type=train_edge_types,
        num_nodes=num_nodes,
        num_relations=num_relations,
        inverse_rel_plus_one=True,
    )

    if relation_graph:
        graph_data = build_relation_graph(graph_data)

    return graph_data


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0]).to(device)

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]],
        (num_rels, num_nodes),
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0]).to(device)
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T,
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]],
        (num_rels, num_nodes),
    )
    Et = torch.sparse_coo_tensor(
        Et.T, torch.ones(Et.shape[0], device=device), (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat(
        [
            Ahh.indices().T,
            torch.zeros(
                Ahh.indices().T.shape[0], 1, dtype=torch.long, device=device
            ).fill_(0),
        ],
        dim=1,
    )  # head to head
    tt_edges = torch.cat(
        [
            Att.indices().T,
            torch.zeros(
                Att.indices().T.shape[0], 1, dtype=torch.long, device=device
            ).fill_(1),
        ],
        dim=1,
    )  # tail to tail
    ht_edges = torch.cat(
        [
            Aht.indices().T,
            torch.zeros(
                Aht.indices().T.shape[0], 1, dtype=torch.long, device=device
            ).fill_(2),
        ],
        dim=1,
    )  # head to tail
    th_edges = torch.cat(
        [
            Ath.indices().T,
            torch.zeros(
                Ath.indices().T.shape[0], 1, dtype=torch.long, device=device
            ).fill_(3),
        ],
        dim=1,
    )  # tail to head

    rel_graph = Data(
        edge_index=torch.cat(
            [
                hh_edges[:, [0, 1]].T,
                tt_edges[:, [0, 1]].T,
                ht_edges[:, [0, 1]].T,
                th_edges[:, [0, 1]].T,
            ],
            dim=1,
        ),
        edge_type=torch.cat(
            [hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0
        ),
        num_nodes=num_rels,
        num_relations=4,
    )

    graph.relation_graph = rel_graph
    return graph
