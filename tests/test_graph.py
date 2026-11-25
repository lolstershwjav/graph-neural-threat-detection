"""Check input isolation, graph invariance and the full benchmark path."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from graphids.data import labels_to_array, windows_for
from graphids.model import Detector, neighborhood
from graphids.study import run


def frame():
    time = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    for batch in range(5):
        for i in range(20):
            rows.append(dict(StartTime=str(time + pd.Timedelta(seconds=batch * 90 + i)),
                             Dur=1., Proto="udp" if i % 3 else "tcp", SrcAddr=f"s{i % 4}",
                             DstAddr=f"d{i % 5}", Dport=53 if i % 5 else 443,
                             TotPkts=i+1, TotBytes=10*i, SrcBytes=4*i,
                             Label=["Botnet", "Normal", "Background"][i % 3]))
    return pd.DataFrame(rows)


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = dict(window_seconds=90, edges_per_window=50, history_windows=3)

    def test_background_is_unverified_context(self):
        y = labels_to_array(pd.Series(["Background", "Normal", "From-Botnet", "To-Botnet", "C&C"]))
        np.testing.assert_array_equal(y, [-1, 0, 1, 1, 1])

    def test_features_are_label_independent(self):
        source = frame()
        a = list(windows_for(source, 1, 0, self.cfg))
        source["Label"] = "Normal"
        b = list(windows_for(source, 1, 0, self.cfg))
        for left, right in zip(a, b, strict=True):
            torch.testing.assert_close(left.flow, right.flow)
            torch.testing.assert_close(left.x, right.x)
            torch.testing.assert_close(left.history, right.history)
            self.assertFalse(torch.equal(left.y, right.y))

    def test_history_uses_only_earlier_closed_windows(self):
        source = frame()
        a = list(windows_for(source, 1, 0, self.cfg))
        self.assertEqual(len(a), 5)
        self.assertFalse(a[0].history.any())
        torch.testing.assert_close(a[1].history[:, -1, :-1], a[0].x)
        altered = source.copy()
        altered.iloc[-20:, altered.columns.get_loc("TotBytes")] = 99999
        b = list(windows_for(altered, 1, 0, self.cfg))
        torch.testing.assert_close(a[4].history, b[4].history)
        self.assertFalse(torch.equal(a[4].x, b[4].x))

    def test_message_passing_is_node_permutation_equivariant(self):
        w = next(windows_for(frame(), 1, 0, self.cfg))
        n = len(w.x)
        ids = torch.arange(n - 1, -1, -1)
        inverse = torch.argsort(ids)
        for method in ("gcn", "sage", "gat"):
            with self.subTest(method=method):
                m = Detector(w.x.shape[1], w.flow.shape[1], 16, method, temporal=True).eval()
                g = neighborhood(w.edges, n, "true", torch.Generator())
                first = m(w.x, w.edges, w.flow, w.history, g)
                second = m(w.x[ids], inverse[w.edges], w.flow, w.history[ids], inverse[g])
                torch.testing.assert_close(first[0], second[0])
                torch.testing.assert_close(first[1], second[1][inverse])
                (first[0].mean() + first[1].mean()).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_capture_validation_and_end_to_end(self):
        config = {**self.cfg, "train":[1], "validation":[2], "test":[3],
                  "seeds":[4], "epochs":1, "threshold_fpr":.05}
        with self.assertRaises(ValueError):
            run(Path("."), Path("."), {**config, "test":[1]})
        windows = [w for i in (1, 2, 3) for w in windows_for(frame(), i, 0, self.cfg)]
        with patch("graphids.study.load", return_value=windows):
            output = run(Path("."), Path("."), config)
        self.assertEqual(len(output["summary"]), 9)
        self.assertFalse(output["dashboard"])
        json.dumps(output, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
