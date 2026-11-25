"""Fetch bounded, disjoint ranges of labeled CTU-13 flow CSVs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://mcfp.felk.cvut.cz/publicDatasets/"
FILES = {42: "20110810", 43: "20110811", 44: "20110812", 45: "20110815",
         46: "20110815-2", 47: "20110816", 48: "20110816-2", 49: "20110816-3",
         50: "20110817", 51: "20110818", 52: "20110818-2", 53: "20110819",
         54: "20110815-3"}
RANGE_SIZE = 1_750_000
N_SEGMENTS = 4


def fetch_range(url, start, end):
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=35) as response:
        if response.status != 206:
            raise ValueError(f"Expected HTTP partial response, received {response.status}")
        size = int(response.headers.get("Content-Range", "").split("/")[-1])
        data = response.read(end - start + 1)
    return size, data


def acquire(capture: int, root: Path) -> dict:
    name = f"capture{FILES[capture]}.binetflow"
    url = f"{BASE}CTU-Malware-Capture-Botnet-{capture}/detailed-bidirectional-flow-labels/{name}"
    # Request one bounded prefix to discover the full size and the CSV schema.
    total, prefix = fetch_range(url, 0, RANGE_SIZE - 1)
    header = prefix.split(b"\n", 1)[0].rstrip(b"\r") + b"\n"
    if not header.startswith(b"StartTime,Dur,Proto,"):
        raise ValueError(f"Unexpected CSV header for capture {capture}")
    width = min(RANGE_SIZE, total // N_SEGMENTS)
    offsets = [i * (total - width) // (N_SEGMENTS - 1) for i in range(N_SEGMENTS)]
    manifest = []
    for segment, start in enumerate(offsets):
        end = start + width - 1
        observed, data = (total, prefix[:width]) if not start else fetch_range(url, start, end)
        if observed != total or len(data) != width:
            raise ValueError(f"Flow archive changed during read for capture {capture}")
        # Drop arbitrary partial rows at segment boundaries.
        body = data.split(b"\n", 1)[1].rsplit(b"\n", 1)[0] + b"\n"
        target = root / f"capture-{capture}-{segment}.csv"
        temporary = target.with_suffix(".tmp")
        output = header + body
        temporary.write_bytes(output)
        temporary.replace(target)
        manifest.append(dict(file=target.name, start=start, end=end,
                             sha256=hashlib.sha256(output).hexdigest()))
    return dict(capture=capture, url=url, source_bytes=total, ranges=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/ctu13-ranges.json"))
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    expected = {} if args.refresh_manifest else {
        item["capture"]: item for item in json.loads(args.manifest.read_text())
    }

    def process(capture):
        prior = expected.get(capture)
        if prior and all(
            (args.data / item["file"]).is_file()
            and hashlib.sha256((args.data / item["file"]).read_bytes()).hexdigest() == item["sha256"]
            for item in prior["ranges"]
        ):
            return prior
        actual = acquire(capture, args.data)
        if prior and actual != prior:
            raise ValueError(f"Flow data differs from pinned manifest for capture {capture}")
        return actual

    with ThreadPoolExecutor(max_workers=5) as pool:
        captures = list(pool.map(process, FILES))
    if args.refresh_manifest:
        args.manifest.write_text(json.dumps(captures, indent=2) + "\n")


if __name__ == "__main__":
    main()
