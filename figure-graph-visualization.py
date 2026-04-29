# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import trnlib
from orbitkit.utils import module_logger

log = module_logger("trn")


def main(
    n: int,
    *,
    tfinal: float = 1.0,
    adjacency: str = "set1",
    outfile: pathlib.Path | None = None,
    force: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path(f"trn_graph_{adjacency}_{n:03d}")

    if not force and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'", outfile)
        return 1

    import networkx as nx

    # rng = np.random.default_rng()
    # A, indices, gaps = trnlib.make_adjacency_set_random(adjacency, n, rng=rng)
    A, indices, gaps = trnlib.make_adjacency_set(adjacency, n)
    G = nx.from_numpy_array(A)
    # log.info("Matrix:\n%s", trnlib.stringify_adjacency(A, fmt="box"))

    i = 0
    for index, m in enumerate(indices):
        for k in range(i, i + m):
            G.nodes[k]["group"] = f"Group {index}"

        i += m + gaps[index]

    for k in G.nodes():
        if "group" not in G.nodes[k]:
            G.nodes[k]["group"] = ""

    from orbitkit.visualization import figure

    with figure(outfile, overwrite=force) as fig:
        _ax = fig.gca()

        import matplotlib.pyplot as mp

        colors = mp.rcParams["axes.prop_cycle"].by_key()["color"]
        palette = {f"Group {i}": colors[i] for i in range(indices.size)}
        palette = {**palette, "": "k"}

        import nxviz as nv  # ty: ignore[unresolved-import]

        nv.circos(
            G,
            node_color_by="group",
            node_palette=palette,
            # edge_enc_kwargs={"alpha_scale": 0.25},
        )

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Name of the output file",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=100,
        help="Number of excitatory and inhibitory neurons",
    )
    parser.add_argument(
        "--adjacency",
        choices=("set1", "set2", "set3", "set4"),
        default="set1",
        help="Name of one of the hardcoded adjacency matrices",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.size,
            adjacency=args.adjacency,
            outfile=args.outfile,
            force=args.force,
        )
    )
