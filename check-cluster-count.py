# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import time
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
        outfile = pathlib.Path("trn")

    A, _, _ = trnlib.make_adjacency_set(adjacency, n)
    # log.info("Matrix:\n%s", trnlib.stringify_adjacency(A))
    model = trnlib.Model.defaults(n, A=A)

    rng = np.random.default_rng(seed=None)
    x0 = np.hstack([
        rng.uniform(-60.0, 0.0, n),
        rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n),
    ])
    assert x0.shape == (3 * n,)

    for g_syn, g_el in [(0.3, 0.06)]:
        log.info("Running for g_syn = %g g_el = %g", g_syn, g_el)
        model = replace(model, param=replace(model.param, g_syn=g_syn, g_el=g_el))

        suffix = f"cluster_{g_syn:.4f}_{g_el:.4f}".replace(".", "_")
        with trnlib.jitcode_module(suffix) as module_location:
            tspan = (0.0, tfinal)
            result = trnlib.solve_ivp(model, tspan, x0, module_location=module_location)

        from orbitkit.clusters import find_clusters_from_timeseries

        t_start = time.time()
        clusters = find_clusters_from_timeseries(result.y[:n])
        log.info("Found %d clusters (time %.3fs)", len(clusters), time.time() - t_start)

        # check that the clusters are actually synced
        window = result.t.size // 6
        V = result.y[:n, -window:]
        if np.max(V) > model.V_threshold:
            log.info("Found a cluster over threshold!")

        for c in clusters:
            y = V[c, :]

            # fmt: off
            error = (
                np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
                / np.max(np.linalg.norm(y, axis=1, ord=np.inf))
            )
            # fmt: on
            log.info("Cluster: %s", c)
            log.info("Errors:  %.8e", np.linalg.norm(error))

        from orbitkit.synchrony import kuramoto_order_parameter, pfeuty_chi

        log.info(
            "Synchrony: Pfeuty %.8e Kuramoto %.8e",
            pfeuty_chi(V),
            kuramoto_order_parameter(V),
        )

        from orbitkit.clusters import make_mse_weight_matrix

        mat = make_mse_weight_matrix(V, alpha=1.0)

        from orbitkit.visualization import figure

        filename = outfile.with_stem(f"{outfile.stem}_{suffix}_weights")
        with figure(filename, normalize=True, overwrite=force) as fig:
            ax = fig.gca()

            im = ax.imshow(mat)
            ax.grid(visible=False, which="both")
            fig.colorbar(im, ax=ax, shrink=0.75)

        filename = outfile.with_stem(f"{outfile.stem}_{suffix}_solution")
        with figure(filename, normalize=True, overwrite=force, figsize=(10, 5)) as fig:
            ax = fig.gca()

            V = result.y[:n, :]
            ax.plot(result.t, V.T)

            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Voltage (mV)")
            ax.set_xlim(result.t[0], result.t[-1])
            ax.set_ylim([-80.0, 0.0])

        filename = outfile.with_stem(f"{outfile.stem}_{suffix}_unique")
        with figure(filename, normalize=True, overwrite=force, figsize=(10, 5)) as fig:
            ax = fig.gca()

            for i, indices in enumerate(clusters):
                ax.plot(
                    result.t[-window:],
                    V[indices[0], -window:],
                    label=f"Cluster {i + 1}",
                )

            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Voltage (mV)")
            ax.set_xlim(result.t[-window], result.t[-1])
            ax.set_ylim([-80.0, 0.0])
            ax.legend(
                loc="upper left",
                fontsize=16,
                columnspacing=1.0,
                ncol=len(clusters),
            )

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
        default=3.0,
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
