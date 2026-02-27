import time
import random
import numpy as np
import torch
import logging
import os
import json
from argparse import Namespace
from typing import List
from torch_scatter import scatter_add
from torch_geometric.data import Data


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
    Parse command-line arguments, auto-detect device, and convert relative paths to absolute.
    Note: json_args_name and json_keys parameters are kept for API compatibility but config files are no longer used.
    """

    args = parse_args(command_line_args)
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert relative paths to absolute paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    args_dict = vars(args)
    for key in args_dict:
        if key.endswith("_path") and args_dict[key] is not None and not os.path.isabs(args_dict[key]):
            args_dict[key] = os.path.abspath(os.path.join(str(root_dir), args_dict[key]))
    args = Namespace(**args_dict)
    
    return args


def save_component(save_path: str, model_name: str, component_name: str, component):
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
    torch.save(component, os.path.join(save_path, model_name, component_name + ".pth"))




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


def get_graph(db_controller, device, augment_inverse_edges=True, relation_graph=True):

    graph_triplets = db_controller.get_whole_graph()

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
