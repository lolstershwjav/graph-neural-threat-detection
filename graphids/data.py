"""Construct observation-time graph windows from labeled CTU-13 flow summaries."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FLOW_COLUMNS = ["StartTime", "Dur", "Proto", "SrcAddr", "DstAddr", "Dport",
                "TotPkts", "TotBytes", "SrcBytes", "Label"]
PROTO = ("tcp", "udp", "icmp", "arp")


@dataclass
class Window:
    capture: int
    segment: int
    time: int
    nodes: np.ndarray
    x: torch.Tensor
    edges: torch.Tensor
    flow: torch.Tensor
    history: torch.Tensor
    y: torch.Tensor
    host_y: torch.Tensor


def numeric_features(frame):
    number = frame[["Dur", "TotPkts", "TotBytes", "SrcBytes"]].apply(pd.to_numeric, errors="coerce")
    values = number.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy()
    log_columns = np.log1p(values)
    ratio = np.divide(values[:, 3], np.maximum(1, values[:, 2])).clip(0, 1)
    proto = frame.Proto.str.lower().to_numpy()
    port = pd.to_numeric(frame.Dport, errors="coerce").fillna(-1).to_numpy()
    return np.column_stack([log_columns, ratio, *(proto == p for p in PROTO),
                            port < 0, (port >= 0) & (port < 1024),
                            (port >= 1024) & (port < 49152), port >= 49152]).astype(np.float32)


def aggregate_nodes(edges, flow, n_nodes):
    source, dest = edges
    result = []
    for indices in (source, dest):
        counts = np.bincount(indices, minlength=n_nodes)
        summed = np.zeros((n_nodes, flow.shape[1]), dtype=np.float32)
        np.add.at(summed, indices, flow)
        result.extend([np.log1p(counts[:, None]), summed / np.maximum(1, counts[:, None])])
    return np.column_stack(result).astype(np.float32)


def labels_to_array(labels):
    text = labels.fillna("").str.lower()
    positive = text.str.contains("botnet", regex=False) | text.str.contains("c&c", regex=False)
    negative = text.str.contains("normal", regex=False) & ~positive
    return np.where(positive, 1, np.where(negative, 0, -1)).astype(np.int64)


def windows_for(frame, capture, segment, config):
    time = pd.to_datetime(frame.StartTime, errors="coerce", format="mixed", utc=True)
    length = pd.to_numeric(frame.Dur, errors="coerce").clip(0, 86400)
    end = time + pd.to_timedelta(length, unit="s")
    is_valid = end.notna() & frame.SrcAddr.notna() & frame.DstAddr.notna()
    frame = frame.loc[is_valid].copy()
    frame["end"] = end[is_valid]
    frame = frame.sort_values("end", kind="stable").drop_duplicates(subset=FLOW_COLUMNS[:-1])
    frame["window"] = frame.end.astype("int64") // (10**9 * config["window_seconds"])
    past = {}
    for tick, group in frame.groupby("window", sort=True):
        cap = config["edges_per_window"]
        if len(group) > cap:
            group = group.iloc[np.linspace(0, len(group) - 1, cap, dtype=int)]
        labels = labels_to_array(group.Label)
        features = numeric_features(group)
        names = np.concatenate([group.SrcAddr.astype(str), group.DstAddr.astype(str)])
        nodes, mapped = np.unique(names, return_inverse=True)
        edges = mapped.reshape(2, -1)
        signals = aggregate_nodes(edges, features, len(nodes))
        history = np.zeros((len(nodes), config["history_windows"], signals.shape[1] + 1),
                           dtype=np.float32)
        allowed = config["history_windows"]
        for idx, node in enumerate(nodes):
            for previous_tick, previous in past.get(node, ()):
                offset = int(tick - previous_tick)
                if 0 < offset <= allowed:
                    history[idx, allowed - offset, :-1] = previous
                    history[idx, allowed - offset, -1] = 1
            past[node] = [(t, v) for t, v in past.get(node, []) if tick - t < allowed]
            past[node].append((tick, signals[idx]))
        host_label = np.full(len(nodes), -1)
        host_label[edges[:, labels == 0]] = 0
        host_label[edges[:, labels == 1]] = 1
        yield Window(capture, segment, int(tick), nodes, torch.from_numpy(signals),
                     torch.from_numpy(edges), torch.from_numpy(features), torch.from_numpy(history),
                     torch.from_numpy(labels), torch.from_numpy(host_label))


def load(directory: Path, manifest: Path, config):
    windows = []
    for item in json.loads(manifest.read_text()):
        for segment, part in enumerate(item["ranges"]):
            path = directory / part["file"]
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != part["sha256"]:
                raise ValueError(f"Flow segment hash mismatch for {path.name}")
            frame = pd.read_csv(path, usecols=FLOW_COLUMNS, dtype={"Dport": "string"}, low_memory=False)
            # Cross-segment identities are used only to assess an unseen-host slice.
            windows.extend(windows_for(frame, item["capture"], segment, config))
    return windows
