#!/usr/bin/env python3
"""
Evaluate 1-hop link-prediction accuracy of dataset-specific ULTRA models vs UltraQuery (zero-shot).

Compares:
  - ultra_fb15k237 vs ultraquery on FB15k-237
  - ultra_nell955   vs ultraquery on NELL-955

Metrics: MRR, Hits@1, Hits@3, Hits@10 for one-hop (head, relation) -> tail.

Usage:
  Self-contained (starts Neo4j, imports dataset, runs eval, optional stop):
    bash src/run_eval_ultra_onehop.sh --dataset fb15k-237
    bash src/run_eval_ultra_onehop.sh --dataset fb15k-237 nell-955 -o artifacts/eval_onehop_results.csv

  Standalone (Neo4j must already be running with the target dataset loaded):
    python src/eval_ultra_onehop.py --dataset fb15k-237
    python src/eval_ultra_onehop.py --dataset fb15k-237 --max-test 5000 --device cuda -o results.csv
    python src/eval_ultra_onehop.py --dataset fb15k-237 --test-triples path/to/test.txt
"""

import os
import sys
import argparse
import logging
import random
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass

import torch
from tqdm import tqdm

# Repo layout: script lives in src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from inference import ModelFactory
from graph_handler import Neo4JBackendDBController
from utils import get_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Snapshot names under artifacts/snapshots/
DATASET_TO_MODEL_DIR = {
    "fb15k-237": "ultra_fb15k237",
    "nell-955": "ultra_nell955",
}
ULTRAQUERY_DIR = "ultraquery"


@dataclass
class OneHopMetrics:
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    # Multilabel-style: precision = |pred∩relevant|/|pred|, recall = |pred∩relevant|/|relevant| (macro = mean over queries)
    precision_at_1: float
    precision_at_3: float
    precision_at_10: float
    recall_at_1: float
    recall_at_3: float
    recall_at_10: float
    # Macro F1 = 2*P*R/(P+R) using macro P and R
    macro_f1_at_1: float
    macro_f1_at_3: float
    macro_f1_at_10: float
    # Micro: pool TP/pred and TP/relevant across all queries
    micro_precision_at_1: float
    micro_precision_at_3: float
    micro_precision_at_10: float
    micro_recall_at_1: float
    micro_recall_at_3: float
    micro_recall_at_10: float
    micro_f1_at_1: float
    micro_f1_at_3: float
    micro_f1_at_10: float
    num_queries: int

    def __str__(self) -> str:
        return (
            f"MRR={self.mrr:.4f}  Hits@1/3/10={self.hits_at_1:.4f}/{self.hits_at_3:.4f}/{self.hits_at_10:.4f}  "
            f"Macro P@1/3/10={self.precision_at_1:.4f}/{self.precision_at_3:.4f}/{self.precision_at_10:.4f}  "
            f"Macro R@1/3/10={self.recall_at_1:.4f}/{self.recall_at_3:.4f}/{self.recall_at_10:.4f}  "
            f"Macro F1@1/3/10={self.macro_f1_at_1:.4f}/{self.macro_f1_at_3:.4f}/{self.macro_f1_at_10:.4f}  "
            f"Micro P/R/F1@10={self.micro_precision_at_10:.4f}/{self.micro_recall_at_10:.4f}/{self.micro_f1_at_10:.4f}  (n={self.num_queries})"
        )


def compute_ranks_and_hits(
    scores: torch.Tensor,
    true_tails: torch.Tensor,
    k_list: List[int] = [1, 3, 10],
) -> Tuple[float, Dict[int, float], List[float]]:
    """
    scores: (batch_size, num_entities), true_tails: (batch_size,)
    Returns: mean_reciprocal_rank, {k: hits@k}, list of reciprocal ranks.
    """
    batch_size = scores.shape[0]
    true_tails = true_tails.to(scores.device).long()
    _, indices = torch.sort(scores, dim=1, descending=True)
    ranks = torch.zeros(batch_size, device=scores.device, dtype=torch.float)
    for i in range(batch_size):
        pos = (indices[i] == true_tails[i]).nonzero(as_tuple=True)[0]
        ranks[i] = (pos.item() + 1) if len(pos) > 0 else 0.0
    rr = torch.where(ranks > 0, 1.0 / ranks, torch.zeros_like(ranks))
    mrr = rr.sum().item() / batch_size
    hits = {}
    for k in k_list:
        hits[k] = (ranks <= k).float().sum().item() / batch_size
    return mrr, hits, rr.cpu().tolist()


def compute_multilabel_precision_recall(
    scores: torch.Tensor,
    relevant_per_query: List[Set[int]],
    k_list: List[int] = [1, 3, 10],
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float]]]:
    """
    Multilabel-style precision and recall: per query, precision = |pred∩relevant|/|pred|,
    recall = |pred∩relevant|/|relevant|. Returns macro (P, R, F1) and micro (P, R) per k.

    relevant_per_query: list of sets of true tail entity ids (one set per query).
    scores: (batch_size, num_entities).
    Returns:
      macro: dict k -> (macro_precision, macro_recall, macro_f1)
      micro: dict k -> (micro_precision, micro_recall)
    """
    batch_size = scores.shape[0]
    _, indices = torch.sort(scores, dim=1, descending=True)
    macro = {k: (0.0, 0.0, 0.0) for k in k_list}
    micro = {k: (0.0, 0.0) for k in k_list}

    for k in k_list:
        sum_precision_i = 0.0
        sum_recall_i = 0.0
        n_with_relevant = 0
        total_tp = 0
        total_pred = 0
        total_relevant = 0
        for i in range(batch_size):
            pred_top_k = set(indices[i, :k].cpu().tolist())
            rel = relevant_per_query[i]
            if not rel:
                continue
            n_with_relevant += 1
            tp = len(pred_top_k & rel)
            total_tp += tp
            total_pred += len(pred_top_k)
            total_relevant += len(rel)
            sum_precision_i += (tp / len(pred_top_k)) if pred_top_k else 0.0
            sum_recall_i += (tp / len(rel)) if rel else 0.0
        n = n_with_relevant
        if n == 0:
            continue
        macro_p = sum_precision_i / n
        macro_r = sum_recall_i / n
        macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)) if (macro_p + macro_r) > 0 else 0.0
        macro[k] = (macro_p, macro_r, macro_f1)
        micro_p = (total_tp / total_pred) if total_pred else 0.0
        micro_r = (total_tp / total_relevant) if total_relevant else 0.0
        micro[k] = (micro_p, micro_r)
    sums_out = {k: (0, 0, 0) for k in k_list}
    for k in k_list:
        tp = pred = rel = 0
        for i in range(batch_size):
            if not relevant_per_query[i]:
                continue
            pred_k = set(indices[i, :k].cpu().tolist())
            tp += len(pred_k & relevant_per_query[i])
            pred += len(pred_k)
            rel += len(relevant_per_query[i])
        sums_out[k] = (tp, pred, rel)
    return macro, micro, sums_out


def sample_one_hop_triples(
    graph_triplets: List[Tuple[int, int, int]],
    max_test: int,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """Sample (head, relation, tail) triples for 1-hop evaluation. Each is one (h, r, t)."""
    rng = random.Random(seed)
    triples = list(graph_triplets)
    rng.shuffle(triples)
    return triples[:max_test]


def load_triples_from_file(path: str) -> List[Tuple[int, int, int]]:
    """Load (head, relation, tail) triples from file. One triple per line: 'head_id relation_id tail_id'."""
    triples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
    return triples


def load_ultra_model(
    load_path: str,
    device: str,
    neo4j_host: str,
    neo4j_bolt_port: int,
    node_unique_id: str,
    relation_unique_id: str,
    use_ultraquery_defaults: bool = False,
):
    """Load ULTRA model from snapshot. Uses ModelFactory with model_to_infer=ULTRA."""
    inf_args = argparse.Namespace(
        load_path=load_path,
        device=device,
        neo4j_host=neo4j_host,
        neo4j_bolt_port=neo4j_bolt_port,
        node_unique_id=node_unique_id,
        relation_unique_id=relation_unique_id,
        model_to_infer="ULTRA",
        use_multi_gpu=True,
    )
    if use_ultraquery_defaults:
        inf_args.msp_threshold = 0.0
    factory = ModelFactory(inf_args)
    factory.prepare_model()
    return factory.model


def evaluate_one_hop(
    model: torch.nn.Module,
    graph_data: Any,
    triples: List[Tuple[int, int, int]],
    device: str,
    batch_size: int = 64,
) -> OneHopMetrics:
    """Run 1-hop evaluation: for each (h, r, t), score tail entities and compute MRR, Hits@k, and multilabel P/R/F1."""
    model.eval()
    k_list = [1, 3, 10]
    all_rr: List[float] = []
    total_hits = {k: 0 for k in k_list}
    n = 0
    # Accumulators for multilabel macro/micro across batches
    sum_macro_p = {k: 0.0 for k in k_list}
    sum_macro_r = {k: 0.0 for k in k_list}
    n_with_relevant = {k: 0 for k in k_list}
    total_tp = {k: 0 for k in k_list}
    total_pred = {k: 0 for k in k_list}
    total_rel = {k: 0 for k in k_list}

    for start in tqdm(range(0, len(triples), batch_size), desc="1-hop eval", leave=False):
        batch_triples = triples[start : start + batch_size]
        heads = torch.tensor([t[0] for t in batch_triples], dtype=torch.long, device=device)
        relations = torch.tensor([t[1] for t in batch_triples], dtype=torch.long, device=device)
        tails = torch.tensor([t[2] for t in batch_triples], dtype=torch.long, device=device)
        query = torch.stack([heads, relations], dim=1)
        with torch.no_grad():
            scores = model.forward(graph_data, query)
        if isinstance(scores, tuple):
            scores = scores[0]
        mrr, hits, rr_list = compute_ranks_and_hits(scores, tails, k_list)
        all_rr.extend(rr_list)
        for k in k_list:
            total_hits[k] += hits[k] * len(batch_triples)
        n += len(batch_triples)
        # Multilabel: one relevant tail per query (set per row)
        relevant_per_query = [{t.item()} for t in tails.cpu()]
        macro_batch, micro_batch, sums_batch = compute_multilabel_precision_recall(scores, relevant_per_query, k_list)
        for k in k_list:
            mp, mr, mf1 = macro_batch[k]
            n_q = len(batch_triples)  # all have relevant
            sum_macro_p[k] += mp * n_q
            sum_macro_r[k] += mr * n_q
            n_with_relevant[k] += n_q
            b_tp, b_pred, b_rel = sums_batch[k]
            total_tp[k] += b_tp
            total_pred[k] += b_pred
            total_rel[k] += b_rel

    if n == 0:
        return OneHopMetrics(
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0,
        )
    mrr = sum(all_rr) / n
    h1 = total_hits[1] / n
    h3 = total_hits[3] / n
    h10 = total_hits[10] / n

    def _macro_p_r_f1(k: int):
        nn = n_with_relevant[k]
        if nn == 0:
            return 0.0, 0.0, 0.0
        mp = sum_macro_p[k] / nn
        mr = sum_macro_r[k] / nn
        mf1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
        return mp, mr, mf1

    def _micro_p_r_f1(k: int):
        if total_pred[k] == 0 or total_rel[k] == 0:
            return 0.0, 0.0, 0.0
        mic_p = total_tp[k] / total_pred[k]
        mic_r = total_tp[k] / total_rel[k]
        mic_f1 = (2 * mic_p * mic_r / (mic_p + mic_r)) if (mic_p + mic_r) > 0 else 0.0
        return mic_p, mic_r, mic_f1

    p1, r1, f1_1 = _macro_p_r_f1(1)
    p3, r3, f1_3 = _macro_p_r_f1(3)
    p10, r10, f1_10 = _macro_p_r_f1(10)
    mic_p1, mic_r1, mic_f1_1 = _micro_p_r_f1(1)
    mic_p3, mic_r3, mic_f1_3 = _micro_p_r_f1(3)
    mic_p10, mic_r10, mic_f1_10 = _micro_p_r_f1(10)

    return OneHopMetrics(
        mrr=mrr,
        hits_at_1=h1,
        hits_at_3=h3,
        hits_at_10=h10,
        precision_at_1=p1,
        precision_at_3=p3,
        precision_at_10=p10,
        recall_at_1=r1,
        recall_at_3=r3,
        recall_at_10=r10,
        macro_f1_at_1=f1_1,
        macro_f1_at_3=f1_3,
        macro_f1_at_10=f1_10,
        micro_precision_at_1=mic_p1,
        micro_precision_at_3=mic_p3,
        micro_precision_at_10=mic_p10,
        micro_recall_at_1=mic_r1,
        micro_recall_at_3=mic_r3,
        micro_recall_at_10=mic_r10,
        micro_f1_at_1=mic_f1_1,
        micro_f1_at_3=mic_f1_3,
        micro_f1_at_10=mic_f1_10,
        num_queries=n,
    )


def run_dataset(
    dataset: str,
    repo_root: str,
    neo4j_host: str,
    neo4j_bolt_port: int,
    node_unique_id: str,
    relation_unique_id: str,
    device: str,
    max_test: int,
    seed: int,
    test_triples_path: Optional[str] = None,
) -> Dict[str, OneHopMetrics]:
    """
    For one dataset: load graph from Neo4j, sample 1-hop test triples,
    evaluate dataset-specific ULTRA and UltraQuery, return metrics for both.
    """
    uri = f"neo4j://{neo4j_host}:{neo4j_bolt_port}"
    db = Neo4JBackendDBController(uri, node_unique_id, relation_unique_id)
    try:
        graph_triplets = db.get_whole_graph()
        num_relations = db.get_num_relations()
        num_nodes = len(db.get_all_node_ids())
        graph_data = get_graph(db, device, augment_inverse_edges=True, relation_graph=True)
    finally:
        db.close()

    logger.info(f"Dataset {dataset}: {num_nodes} nodes, {num_relations} relations, {len(graph_triplets)} edges")
    if test_triples_path and os.path.isfile(test_triples_path):
        triples = load_triples_from_file(test_triples_path)
        logger.info(f"Loaded {len(triples)} 1-hop test triples from {test_triples_path}")
    else:
        triples = sample_one_hop_triples(graph_triplets, max_test, seed)
        logger.info(f"Using {len(triples)} 1-hop test triples (sampled from graph)")

    results: Dict[str, OneHopMetrics] = {}

    # 1) Dataset-specific model
    model_dir = DATASET_TO_MODEL_DIR.get(dataset)
    if not model_dir:
        logger.warning(f"Unknown dataset {dataset}, skipping dataset-specific model")
    else:
        load_path = os.path.join(repo_root, "artifacts", "snapshots", model_dir)
        if not os.path.isdir(load_path):
            logger.warning(f"Snapshot not found: {load_path}, skipping")
        else:
            logger.info(f"Loading dataset-specific model: {load_path}")
            model = load_ultra_model(
                load_path, device, neo4j_host, neo4j_bolt_port,
                node_unique_id, relation_unique_id, use_ultraquery_defaults=False,
            )
            results[model_dir] = evaluate_one_hop(model, graph_data, triples, device)
            logger.info(f"  {model_dir}: {results[model_dir]}")
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # 2) UltraQuery (zero-shot on this dataset)
    uq_path = os.path.join(repo_root, "artifacts", "snapshots", ULTRAQUERY_DIR)
    if not os.path.isdir(uq_path):
        logger.warning(f"UltraQuery snapshot not found: {uq_path}, skipping")
    else:
        logger.info(f"Loading UltraQuery (zero-shot): {uq_path}")
        model = load_ultra_model(
            uq_path, device, neo4j_host, neo4j_bolt_port,
            node_unique_id, relation_unique_id, use_ultraquery_defaults=True,
        )
        results[ULTRAQUERY_DIR] = evaluate_one_hop(model, graph_data, triples, device)
        logger.info(f"  {ULTRAQUERY_DIR}: {results[ULTRAQUERY_DIR]}")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate 1-hop accuracy: dataset-specific ULTRA vs UltraQuery (zero-shot)"
    )
    ap.add_argument("--dataset", nargs="+", default=["fb15k-237", "nell-955"],
                    help="Datasets to evaluate (default: fb15k-237 nell-955)")
    ap.add_argument("--neo4j_host", default="localhost", help="Neo4j host")
    ap.add_argument("--neo4j_bolt_port", type=int, default=7687, help="Neo4j bolt port")
    ap.add_argument("--node_unique_id", default="id", help="Node unique id field in Neo4j")
    ap.add_argument("--relation_unique_id", default="type", help="Relation unique id field in Neo4j")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    ap.add_argument("--max-test", type=int, default=10000,
                    help="Max 1-hop test triples per dataset (sampled from graph)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling test triples")
    ap.add_argument("--test-triples", type=str, default=None,
                    help="Path to test triples file (one 'head_id relation_id tail_id' per line). Use with a single --dataset.")
    ap.add_argument("--output", "-o", type=str, default=None,
                    help="Path to save results CSV (e.g. artifacts/eval_onehop_results.csv). If not set, results are only printed.")
    args = ap.parse_args()

    repo_root = REPO_ROOT
    all_results: Dict[str, Dict[str, OneHopMetrics]] = {}
    test_path = args.test_triples if (args.test_triples and len(args.dataset) == 1) else None

    for dataset in args.dataset:
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset}")
        logger.info("=" * 60)
        try:
            all_results[dataset] = run_dataset(
                dataset=dataset,
                repo_root=repo_root,
                neo4j_host=args.neo4j_host,
                neo4j_bolt_port=args.neo4j_bolt_port,
                node_unique_id=args.node_unique_id,
                relation_unique_id=args.relation_unique_id,
                device=args.device,
                max_test=args.max_test,
                seed=args.seed,
                test_triples_path=test_path,
            )
        except Exception as e:
            logger.exception(f"Failed for dataset {dataset}: {e}")
            all_results[dataset] = {}

    # Summary table
    print("\n" + "=" * 80)
    print("1-HOP ACCURACY: Dataset-specific ULTRA vs UltraQuery (zero-shot)")
    print("=" * 80)
    for dataset in args.dataset:
        res = all_results.get(dataset, {})
        if not res:
            print(f"\n{dataset}: no results")
            continue
        print(f"\n{dataset} (n={next(iter(res.values())).num_queries} test triples)")
        for model_name, m in res.items():
            print(f"  {model_name:25s}  {m}")
    print()

    # Save to CSV if requested
    if args.output:
        import csv
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset", "model", "mrr", "hits_at_1", "hits_at_3", "hits_at_10",
                "precision_at_1", "precision_at_3", "precision_at_10",
                "recall_at_1", "recall_at_3", "recall_at_10",
                "macro_f1_at_1", "macro_f1_at_3", "macro_f1_at_10",
                "micro_precision_at_1", "micro_precision_at_3", "micro_precision_at_10",
                "micro_recall_at_1", "micro_recall_at_3", "micro_recall_at_10",
                "micro_f1_at_1", "micro_f1_at_3", "micro_f1_at_10", "num_queries",
            ])
            for dataset in args.dataset:
                for model_name, m in all_results.get(dataset, {}).items():
                    w.writerow([
                        dataset, model_name,
                        f"{m.mrr:.4f}", f"{m.hits_at_1:.4f}", f"{m.hits_at_3:.4f}", f"{m.hits_at_10:.4f}",
                        f"{m.precision_at_1:.4f}", f"{m.precision_at_3:.4f}", f"{m.precision_at_10:.4f}",
                        f"{m.recall_at_1:.4f}", f"{m.recall_at_3:.4f}", f"{m.recall_at_10:.4f}",
                        f"{m.macro_f1_at_1:.4f}", f"{m.macro_f1_at_3:.4f}", f"{m.macro_f1_at_10:.4f}",
                        f"{m.micro_precision_at_1:.4f}", f"{m.micro_precision_at_3:.4f}", f"{m.micro_precision_at_10:.4f}",
                        f"{m.micro_recall_at_1:.4f}", f"{m.micro_recall_at_3:.4f}", f"{m.micro_recall_at_10:.4f}",
                        f"{m.micro_f1_at_1:.4f}", f"{m.micro_f1_at_3:.4f}", f"{m.micro_f1_at_10:.4f}",
                        m.num_queries,
                    ])
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
