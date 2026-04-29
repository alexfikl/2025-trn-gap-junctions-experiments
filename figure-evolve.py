# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from dataclasses import replace

import numpy as np

import trnlib
from orbitkit.utils import module_logger

log = module_logger("trn")

# {{{ evolve


def main(
    n: int,
    *,
    tfinal: float = 1.0,
    adjacency: str = "set1",
    outfile: pathlib.Path | None = None,
    force: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path(f"trn_evolve_{n:04d}_{adjacency}")

    A, ms, _ = trnlib.make_adjacency_set(adjacency, n)
    log.info("Gap junction clusters: %s", ms)
    model = trnlib.Model.defaults(n, A=A, g_el=0.0)

    rng = np.random.default_rng(seed=42)
    x0 = np.hstack([
        rng.uniform(-60.0, 0.0, n),
        rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n),
    ])

    suffix = f"evolve_{n:04d}_{adjacency}"
    with trnlib.jitcode_module(suffix) as module_location:
        tspan = (0.0, tfinal)

        for g_syn in [0.2, 0.3, 0.4, 0.65]:
            log.info("Running for g_syn = %g", g_syn)

            model = replace(model, param=replace(model.param, g_syn=g_syn))
            result = trnlib.solve_ivp(model, tspan, x0, module_location=module_location)

            from orbitkit.visualization import figure

            filename = outfile.with_stem(f"{outfile.stem}_{g_syn:.2f}_solution")
            with figure(
                filename, figsize=(10, 6), normalize=True, overwrite=force
            ) as fig:
                ax = fig.gca()

                V = result.y[:n, :].T
                ax.plot(result.t, V)
                ax.axhline(model.V_threshold, color="k", ls="--")

                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Voltage (mV)")
                ax.set_xlim([result.t[0], tfinal])
                ax.set_ylim([-80.0, 0.0])
                ax.set_title("")

    return 0


# }}}


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
        "--tfinal",
        type=float,
        default=5.0,
        help="Final time for the simulation [0, tfinal].",
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
            tfinal=args.tfinal,
            adjacency=args.adjacency,
            outfile=args.outfile,
            force=args.force,
        )
    )
