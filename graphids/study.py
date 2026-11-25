"""Evaluate capture-disjoint graph classifiers on observed flow windows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn

from graphids.data import load
from graphids.model import Detector, neighborhood


NEURAL = {
    "gcn": ("gcn", "true", False, True),
    "sage": ("sage", "true", False, True),
    "gat": ("gat", "true", False, True),
    "temporal_sage": ("sage", "true", True, True),
    "sage_self": ("sage", "self", False, True),
    "sage_rewired": ("sage", "rewired", False, True),
    "sage_flow_only": ("sage", "true", False, False),
}


def pack(windows, train):
    x = torch.cat([w.x for w in train])
    flow = torch.cat([w.flow for w in train])
    centers = x.mean(0), flow.mean(0)
    widths = x.std(0).clamp(min=.01), flow.std(0).clamp(min=.01)
    for w in windows:
        w.x = ((w.x - centers[0]) / widths[0]).clamp(-12, 12)
        w.flow = ((w.flow - centers[1]) / widths[1]).clamp(-12, 12)
        history = ((w.history[:, :, :-1] - centers[0]) / widths[0]).clamp(-12, 12)
        w.history = torch.cat([history * w.history[:, :, -1:], w.history[:, :, -1:]], dim=2)


def metrics(y, scores, threshold):
    mask = y >= 0
    y, scores = y[mask], scores[mask]
    positive = y == 1
    negative = y == 0
    prediction = scores > threshold if threshold is not None else None
    return dict(n=len(y), positives=int(positive.sum()), negatives=int(negative.sum()),
                ap=float(average_precision_score(y, scores)) if positive.any() else None,
                auc=float(roc_auc_score(y, scores)) if positive.any() and negative.any() else None,
                recall=float(prediction[positive].mean()) if positive.any() and prediction is not None else None,
                fpr=float(prediction[negative].mean()) if negative.any() and prediction is not None else None)


def quantile(y, scores, alpha):
    normal = scores[y == 0]
    return float(np.quantile(normal, 1 - alpha, method="higher")) if len(normal) else None


def tabular(kind, train, val, test, seed):
    def matrix(w):
        if kind == "flow_tabular":
            return w.flow.numpy()
        return torch.cat([w.flow, w.x[w.edges[0]], w.x[w.edges[1]]], dim=1).numpy()

    started = perf_counter()
    xs = np.concatenate([matrix(w)[w.y.numpy() >= 0] for w in train])
    ys = np.concatenate([w.y.numpy()[w.y.numpy() >= 0] for w in train])
    classifier = HistGradientBoostingClassifier(max_iter=115, max_leaf_nodes=24,
                                                class_weight="balanced", random_state=seed,
                                                early_stopping=False).fit(xs, ys)
    train_seconds = perf_counter() - started

    def predict(w):
        s = classifier.predict_proba(matrix(w))[:, 1]
        nodes = np.zeros(len(w.nodes))
        for idx in w.edges.numpy():
            np.maximum.at(nodes, idx, s)
        return s, nodes

    start = perf_counter()
    pred_val, pred_test = list(map(predict, val)), list(map(predict, test))
    return pred_val, pred_test, train_seconds, perf_counter() - start, None


def neural(variant, train, val, test, seed, epochs):
    method, topology, temporal, auxiliary = NEURAL[variant]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    graphs = {id(w): neighborhood(w.edges, len(w.nodes), topology, generator)
              for w in [*train, *val, *test]}
    reference = train[0]
    started = perf_counter()
    model = Detector(reference.x.shape[1], reference.flow.shape[1], 32,
                     method, temporal, auxiliary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    y_train = torch.cat([w.y for w in train])
    y_host = torch.cat([w.host_y for w in train])
    flow_weight = (y_train == 0).sum() / (y_train == 1).sum()
    host_weight = (y_host == 0).sum() / (y_host == 1).sum()
    flow_loss = nn.BCEWithLogitsLoss(pos_weight=flow_weight.clamp(.1, 10))
    host_loss = nn.BCEWithLogitsLoss(pos_weight=host_weight.clamp(.1, 10))

    def logits(w):
        return model(w.x, w.edges, w.flow, w.history, graphs[id(w)])

    @torch.inference_mode()
    def predict(w):
        a, b = logits(w)
        score = a.sigmoid().numpy()
        if auxiliary:
            return score, b.sigmoid().numpy()
        hosts = np.zeros(len(w.nodes))
        for index in w.edges.numpy():
            np.maximum.at(hosts, index, score)
        return score, hosts

    yv = np.concatenate([w.y.numpy() for w in val])
    labeled = yv >= 0
    best = -1
    state = None
    for _ in range(epochs):
        model.train()
        for i in rng.permutation(len(train)):
            w = train[i]
            mask = w.y >= 0
            if not mask.any():
                continue
            optimizer.zero_grad()
            s, h = logits(w)
            loss = flow_loss(s[mask], w.y[mask].float())
            if auxiliary:
                mask_node = w.host_y >= 0
                loss = loss + .2 * host_loss(h[mask_node], w.host_y[mask_node].float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        scores = np.concatenate([predict(w)[0] for w in val])
        ap = average_precision_score(yv[labeled], scores[labeled])
        if ap > best:
            best = ap
            state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    train_seconds = perf_counter() - started
    start = perf_counter()
    valid, predicted = list(map(predict, val)), list(map(predict, test))
    prediction_seconds = perf_counter() - start
    return valid, predicted, train_seconds, prediction_seconds, model


def shuffled_endpoints(w, max_flows=200):
    n = min(len(w.y), max_flows)
    # Use a capped, deterministic display subset. No addresses leave the loader.
    ix = np.linspace(0, len(w.y) - 1, n, dtype=int)
    es = w.edges.numpy()[:, ix]
    ids, pairs = np.unique(es, return_inverse=True)
    return ix, ids, pairs.reshape(2, -1)


def evaluate(kind, val, test, predictions, unseen, alpha):
    v, t, fit_time, pred_time, model = predictions
    all_y = np.concatenate([w.y.numpy() for w in test])
    all_h = np.concatenate([w.host_y.numpy() for w in test])
    all_v = np.concatenate([w.y.numpy() for w in val])
    all_vh = np.concatenate([w.host_y.numpy() for w in val])
    sc = np.concatenate([p[0] for p in t])
    ns = np.concatenate([p[1] for p in t])
    cutoff = quantile(all_v, np.concatenate([p[0] for p in v]), alpha)
    cut_host = quantile(all_vh, np.concatenate([p[1] for p in v]), alpha)
    rows = {}
    for subgroup in sorted({w.capture for w in test}):
        ids = [i for i, w in enumerate(test) if w.capture == subgroup]
        flow = np.concatenate([test[i].y.numpy() for i in ids])
        pred = np.concatenate([t[i][0] for i in ids])
        rows[str(subgroup)] = metrics(flow, pred, cutoff)
    order = np.array([len(w.y) for w in test])
    size_pivot = int(np.median(order))
    size_groups = {}
    for name, flags in {"small": order <= size_pivot, "large": order > size_pivot}.items():
        chosen = np.repeat(flags, order)
        size_groups[name] = metrics(all_y[chosen], sc[chosen], cutoff)
    incident = metrics(all_h, ns, cut_host)
    return dict(flow=metrics(all_y, sc, cutoff), host=incident,
                unseen=metrics(all_y[unseen], sc[unseen], cutoff), capture=rows,
                sizes=size_groups, size_cutoff=size_pivot, fit_seconds=fit_time,
                inference_seconds=pred_time, threshold=cutoff,
                unknown_context_alert_rate=float((sc[all_y < 0] > cutoff).mean()),
                model=kind), (t, model)


def demo_artifact(test, vectors, network, seed):
    choices = [i for i, w in enumerate(test) if (w.y == 1).any() and len(w.y) >= 120]
    if not choices:
        return []
    samples = []
    for i in np.array_split(choices, min(6, len(choices))):
        if not len(i):
            continue
        idx = int(i[len(i)//2])
        w = test[idx]
        chosen, anon, endpoints = shuffled_endpoints(w)
        probs = vectors[idx][0]
        shadow = neighborhood(w.edges, len(w.nodes), "self", torch.Generator().manual_seed(seed))
        with torch.inference_mode():
            self_scores = network(w.x, w.edges, w.flow, w.history, shadow)[0].sigmoid().numpy()
        samples.append(dict(capture=w.capture, segment=w.segment, n_nodes=len(anon),
                            n_full_nodes=len(w.nodes), n_full_flows=len(w.y),
                            edges=[dict(source=int(a), dest=int(b), score=float(probs[j]),
                                        topology_delta=float(probs[j]-self_scores[j]),
                                        label=int(w.y[j]))
                                   for j, a, b in zip(chosen, endpoints[0], endpoints[1])]))
    return samples


def bounds(rows):
    groups = defaultdict(list)
    rng = np.random.default_rng(822)
    for row in rows:
        groups[row["model"]].append(row)
    out = {}
    for name, results in groups.items():
        out[name] = {}
        for scope in ("flow", "host", "unseen"):
            summary = {}
            for field in ("ap", "auc", "recall", "fpr"):
                vals = [r[scope][field] for r in results if r[scope][field] is not None]
                if vals:
                    lo, hi = np.quantile(rng.choice(vals, (3000, len(vals))).mean(1), [.025, .975])
                    summary[field] = dict(mean=float(np.mean(vals)), low=float(lo), high=float(hi))
            out[name][scope] = summary
    return out


def run(directory, manifest, config):
    torch.set_num_threads(1)
    splits = [set(config[k]) for k in ("train", "validation", "test")]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        raise ValueError("Capture splits must be disjoint")
    windows = load(directory, manifest, config)
    train, valid, test = ([w for w in windows if w.capture in group] for group in splits)
    for part in (train, valid, test):
        if not part or not {0, 1}.issubset(set(torch.cat([w.y for w in part]).tolist())):
            raise ValueError("Each split must contain labeled positive and negative flows")
    known = {name for w in train for name in w.nodes}
    unseen = []
    for w in test:
        new = np.fromiter((name not in known for name in w.nodes), count=len(w.nodes), dtype=bool)
        unseen.append(new[w.edges[0]] | new[w.edges[1]])
    unseen = np.concatenate(unseen)
    pack(windows, train)
    rows = []
    demo = []
    for seed in config["seeds"]:
        for kind in ("flow_tabular", "context_tabular", *NEURAL):
            if kind in NEURAL:
                pred = neural(kind, train, valid, test, seed, config["epochs"])
            else:
                pred = tabular(kind, train, valid, test, seed)
            result, (vectors, model) = evaluate(kind, valid, test, pred, unseen, config["threshold_fpr"])
            rows.append(dict(seed=seed, **result))
            if kind == "temporal_sage" and not demo:
                demo = demo_artifact(test, vectors, model, seed)
            print(f"Finished {kind}, seed {seed}", flush=True)
    context = dict(windows=len(windows), flows=sum(len(w.y) for w in windows),
                   splits={key: {"windows": len(part), "flows": sum(len(w.y) for w in part),
                                 "normal": sum(int((w.y == 0).sum()) for w in part),
                                 "botnet": sum(int((w.y == 1).sum()) for w in part)}
                           for key, part in zip(("train", "validation", "test"), (train, valid, test))})
    return dict(config=config, dataset_counts=context, runs=rows, summary=bounds(rows), dashboard=demo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/ctu13-ranges.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/study.json"))
    args = parser.parse_args()
    output = run(args.data, args.manifest, json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False))
    temp.replace(args.output)


if __name__ == "__main__":
    main()
