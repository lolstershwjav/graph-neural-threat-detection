"""Build an offline viewer from anonymized evaluation evidence."""

import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/study.json.gz"))
    parser.add_argument("--output", type=Path, default=Path("results/flow-graph.html"))
    args = parser.parse_args()
    opener = gzip.open if args.source.suffix == ".gz" else open
    with opener(args.source, "rt", encoding="utf8") as file:
        cases = json.load(file)["dashboard"]
    if not cases:
        raise ValueError("No eligible graph windows in the evidence")
    template = Path("dashboard/template.html").read_text()
    encoded = json.dumps(cases, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    args.output.write_text(template.replace("__GRAPH_DATA__", encoded), encoding="utf8")


if __name__ == "__main__":
    main()
