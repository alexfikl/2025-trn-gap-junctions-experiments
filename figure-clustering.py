# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import multiprocessing
import pathlib
from collections.abc import Callable
from dataclasses import asdict, replace

import numpy as np

import trnlib
from orbitkit.utils import module_logger

log = module_logger("trn")


def main(
    n: int,
    *,
    nrounds: int = 128,
    tfinal: float = 5.0,
    params: str = "figure3",
    adjacency: str = "set1",
    max_workers: int | None = None,
    outfile: pathlib.Path | None = None,
    force: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path(f"trn_{params}_{n:04d}_{adjacency}.npz")

    outfile = outfile.with_suffix(".npz")
    if not force and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'", outfile)
        return 1

    if max_workers is None:
        # NOTE: don't want to oversaturate the system by default
        max_workers = multiprocessing.cpu_count() // 2

    A, _, _ = trnlib.make_adjacency_set(adjacency, n)
    model = trnlib.Model.defaults(n, A=A)

    from concurrent.futures import ProcessPoolExecutor
    from itertools import product

    result: dict[tuple[int, int, int], trnlib.Cluster] = {}

    if params == "figure3":
        g_syns = np.array([0.2, 0.22, 0.25, 0.35, 0.4, 0.44, 0.65])
        g_els = np.array([0.01, 0.05])
    elif params == "figure5":
        g_syns = np.array([0.4, 0.45])
        g_els = np.array([0.0, 0.009, 0.018, 0.028, 0.05])
    else:
        raise ValueError(f"Unknown parameter set: '{params}'")

    with trnlib.jitcode_module(params) as module_location:
        from functools import partial

        indices = list(product(range(g_syns.size), range(g_els.size), range(nrounds)))
        worker = partial(
            trnlib.find_clusters_for_model,
            model,
            g_syns=g_syns,
            g_els=g_els,
            tspan=(0.0, tfinal),
            module_location=module_location,
            # NOTE: do not be tempted to add an rng here, since it will be the
            # same for all processes!
            rng=None,
        )

        # NOTE: run once to compile the code before launching processes
        _ = worker((0, 0, 0))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, j, cluster in executor.map(worker, indices):
                key = (i, j, cluster.clusterid)

                value = result.get(key)
                if value is None:
                    value = cluster
                else:
                    value = replace(
                        value,
                        count=value.count + cluster.count,
                        tcount=value.tcount + cluster.tcount,
                    )
                result[key] = value

    np.savez(
        outfile,
        tspan=(0, tfinal),
        nrounds=nrounds,
        g_syns=np.array(g_syns),
        g_els=np.array(g_els),
        model=asdict(model),  # ty: ignore[invalid-argument-type]
        cluster_stats={k: asdict(v) for k, v in result.items()},  # ty: ignore[invalid-argument-type]
    )

    return visualize(outfile, force=force)


def _plot_solution_and_heatmap(
    cinfo: trnlib.Cluster,
    model: trnlib.Model,
    filename: pathlib.Path,
    *,
    predicate: Callable[[float], bool],
    module_location: pathlib.Path | None = None,
    overwrite: bool = False,
) -> bool:
    from orbitkit.visualization import figure

    # {{{ simulate solution at parameters

    k = cinfo.clusterid
    window = 0 if k == 0 else cinfo.window

    # solve system
    tspan = (0.0, 1.0) if k == 0 else cinfo.tspan
    result = trnlib.solve_ivp(
        model,
        tspan,
        cinfo.x0,
        module_location=module_location,
        verbose=False,
    )
    V = result.y[: model.n, :]

    # determine cluster error
    error = trnlib.get_cluster_error(cinfo, V[:, -window:])

    if k != 0 and predicate(error):
        return False

    log.info("Plotting solution with %d clusters: error %.8e", k, error)

    # }}}

    # {{{ plot heatmap

    if k != 0:
        # NOTE: only plot the last second
        window = int(np.argmax(result.t >= (tspan[1] - 1.0)))
        window = result.t.size - window

    outfile = filename.with_stem(f"{filename.name}_heatmap")
    with figure(outfile, figsize=(10, 5), normalize=True, overwrite=overwrite) as fig:
        ax = fig.gca()

        istart = 0
        Vhat = np.empty_like(V[:, -window:])
        for idx in cinfo.cluster_indices:
            size = idx.size
            Vhat[istart : istart + size, :] = V[idx, -window:]
            istart += size

        n = np.arange(model.n)
        t = result.t[-window:]
        n, t = np.meshgrid(t, n)
        im = ax.contourf(n, t, Vhat, cmap="jet")

        counts = "/".join(str(i.size) for i in cinfo.cluster_indices)
        ax.set_title(f"{k} Clusters ({counts})")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$n$")
        ax.set_xlim([result.t[-window], tspan[-1]])

        fig.colorbar(im, ax=ax)

    # }}}

    # {{{ plot line

    outfile = filename.with_stem(f"{filename.name}_solution")
    with figure(outfile, figsize=(10, 5), normalize=True, overwrite=overwrite) as fig:
        ax = fig.gca()

        ax.plot(result.t[-window:], V[:, -window:].T)
        ax.axhline(model.V_threshold, color="k", ls="--")

        counts = "/".join(str(i.size) for i in cinfo.cluster_indices)
        ax.text(
            0.95,
            0.9,
            f"{k} Clusters ({counts})"
            if k > 1
            else ("1 Cluster" if k == 1 else "Fixed Point"),
            transform=ax.transAxes,
            fontsize=32,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "square",
                "facecolor": "white",
                "linewidth": 1,
                "pad": 0.5,
            },
        )

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlim([result.t[-window], tspan[-1]])
        ax.set_ylim([-80.0, 0.0])
        ax.grid(visible=False, which="both")

    # }}}

    return True


def visualize(filename: pathlib.Path, *, force: bool = False) -> int:
    if not filename.exists():
        log.info("File does not exist: '%s'.", filename)
        return 1

    from orbitkit.utils import slugify
    from orbitkit.visualization import figure

    data = np.load(filename, allow_pickle=True)

    g_syns = data["g_syns"]
    g_els = data["g_els"]

    if "result_cluster" in data:
        results = data["result_cluster"][()]
    else:
        results = data["cluster_stats"][()]
    model = trnlib.from_dict(trnlib.Model, data["model"][()])

    # {{{ gather histogram

    max_k = 0
    clusters = np.zeros((g_syns.size, g_els.size, 5), dtype=np.int32)
    for (i, j, k), info in results.items():
        c = trnlib.Cluster(**info)
        clusters[i, j, min(k, 4)] += c.count
        max_k = max(max_k, k)

    # }}}

    # {{{ plot

    from itertools import product

    filename = filename.with_suffix("")
    with trnlib.jitcode_module(filename.stem) as module_location:
        for i, j in product(range(g_syns.size), range(g_els.size)):
            log.info("Plotting solutions for gsyn %.3f gel %.3f", g_syns[i], g_els[j])

            # NOTE: we need to slugify here, otherwise some of the `with_stem`
            # below will be confused by the extra dots in the filenames
            basename = slugify(f"{filename.stem}_{g_syns[i]:.4f}_{g_els[j]:.4f}")

            outfile = filename.with_stem(f"{basename}_histogram")
            with figure(
                outfile,
                figsize=(10, 10),
                normalize=True,
                overwrite=force,
                pad_inches=0.125,
            ) as fig:
                ax = fig.gca()

                trnlib.visualize_cluster_stat(
                    ax,
                    clusters[i, j, :] / np.sum(clusters[i, j, :]),
                    g_syn_label=f"{g_syns[i]:.2f}",
                    g_el_label=f"{g_els[j]:.2f}",
                )

            # plot a nicely converged solution
            chimera_k = 0
            for k in range(15, -1, -1):
                if (i, j, k) not in results:
                    continue

                cinfo = trnlib.Cluster(**results[i, j, k])
                model = replace(
                    model, param=replace(model.param, g_syn=g_syns[i], g_el=g_els[j])
                )

                flag = _plot_solution_and_heatmap(
                    cinfo,
                    model,
                    filename.with_stem(basename),
                    predicate=lambda e: e > 1.0e-2,
                    module_location=module_location,
                    overwrite=force,
                )

                if flag:
                    break

                chimera_k = max(chimera_k, k)

            # # NOTE: we reached the fixed point, there's no chimeras here!
            # if chimera_k == 0:
            #     chimera_k = -1

            # # plot a potential chimera solution
            # for k in range(chimera_k, -1, -1):
            #     if (i, j, k) not in results:
            #         continue

            #     cinfo = trnlib.Cluster(**results[i, j, k])
            #     model = replace(
            #         model, param=replace(model.param, g_syn=g_syns[i], g_el=g_els[j])
            #     )

            #     flag = _plot_solution_and_heatmap(
            #         cinfo,
            #         model,
            #         filename.with_stem(f"{basename}_chimera_{k:02d}"),
            #         predicate=lambda e: e < 1.0e-2,
            #         module_location=module_location,
            #         overwrite=force,
            #     )

            #     if flag:
            #         break

    # }}}

    return 0


if __name__ == "__main__":
    import argparse

    # NOTE: don't want to oversaturate the system by default
    max_workers = multiprocessing.cpu_count() // 2

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", type=pathlib.Path)
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
        "--params",
        choices=("figure3", "figure5"),
        default="figure3",
        help="(g_syn, g_el) parameter ranges to simulate",
    )
    parser.add_argument(
        "--adjacency",
        choices=("set1", "set2", "set3", "set4"),
        default="set1",
        help="Name of one of the hardcoded adjacency matrices",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max_workers,
        help="Maximum number of parallel runs (<= 1 will run sequentially)",
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

    if args.filenames:
        ret = 0
        for filename in args.filenames:
            ret += visualize(filename, force=args.force)

        raise SystemExit(ret)
    else:
        raise SystemExit(
            main(
                args.size,
                tfinal=args.tfinal,
                params=args.params,
                adjacency=args.adjacency,
                max_workers=args.jobs,
                outfile=args.outfile,
                force=args.force,
            )
        )
