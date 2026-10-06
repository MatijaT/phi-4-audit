"""Download the Panzer-Schnetz Periods file from arXiv and build both datasets.
Usage: python scripts/build_data.py [--periods PATH]"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phi4audit import census, periods  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--periods", default="data/raw/Periods")
ap.add_argument("--out", default="data/built")
a = ap.parse_args()

periods.fetch(a.periods)
os.makedirs(a.out, exist_ok=True)


def write(rows, name):
    path = os.path.join(a.out, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows, {sum(r['weight_drop'] for r in rows)} with c2 = 0)")


t0 = time.time()
write(census.census(a.periods, max_loop=9), "census_L3-9.csv")
print(f"  {time.time() - t0:.0f}s")
write(census.irreducible(a.periods, loops=range(6, 11)), "irreducible_L6-10.csv")
print(f"  {time.time() - t0:.0f}s")
