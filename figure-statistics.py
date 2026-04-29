# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import multiprocessing
import pathlib
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
    adjacency: str = "set1",
    max_workers: int | None = None,
    outfile: pathlib.Path | None = None,
    force: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path(f"trn_stat_{n:04d}_{adjacency}.npz")

    if not outfile.suffix:
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

    g_syns = np.linspace(0.2, 0.7, 50)
    g_els = np.linspace(0.0, 0.15, 50)

    result: dict[tuple[int, int, int], trnlib.Cluster] = {}

    chi_n = np.zeros((g_syns.size, g_els.size), dtype=np.int32)
    chi_mean = np.zeros((g_syns.size, g_els.size))
    chi_vars = np.zeros((g_syns.size, g_els.size))

    with trnlib.jitcode_module("stat") as module_location:
        from functools import partial

        indices = product(range(g_syns.size), range(g_els.size), range(nrounds))
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

                # compute running mean and variance of the Pfeuty chi value
                chi_n[i, j] += 1
                delta = cluster.pfeuty_chi - chi_mean[i, j]
                chi_mean[i, j] += delta / chi_n[i, j]
                delta2 = cluster.pfeuty_chi - chi_mean[i, j]
                chi_vars[i, j] += delta * delta2

            chi_vars /= chi_n - 1

    np.savez(
        outfile,
        tspan=(0, tfinal),
        nrounds=nrounds,
        g_syns=np.array(g_syns),
        g_els=np.array(g_els),
        model=asdict(model),  # ty: ignore[invalid-argument-type]
        cluster_stats={k: asdict(v) for k, v in result.items()},  # ty: ignore[invalid-argument-type]
        chi_mean=chi_mean,
        chi_std=np.sqrt(chi_vars),
    )

    return visualize(outfile, force=force)


def visualize(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    force: bool = False,
) -> int:
    if filename.suffix:
        filename = filename.with_suffix(".npz")

    if not filename.exists():
        log.info("File does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_suffix("")

    from orbitkit.visualization import figure

    data = np.load(filename, allow_pickle=True)

    nrounds = data["nrounds"]
    g_syns = data["g_syns"]
    g_els = data["g_els"]

    results = data["cluster_stats"][()]

    # {{{ gather stats

    stats = np.zeros((g_syns.size, g_els.size))
    thresh_stats = np.zeros((g_syns.size, g_els.size))
    thresh_count = np.zeros((g_syns.size, g_els.size), dtype=np.int32)
    stats_per_cluster = [np.zeros((g_syns.size, g_els.size)) for _ in range(5)]

    max_k = 0
    for (i, j, k), info in results.items():
        if "window" not in info:
            info["window"] = 0

        max_k = max(max_k, k)
        cluster = trnlib.Cluster(**info)
        assert cluster.clusterid == k

        stats_per_cluster[min(k, 4)][i, j] += cluster.count
        stats[i, j] += k * cluster.count
        thresh_stats[i, j] += k * cluster.tcount
        thresh_count[i, j] += cluster.tcount

    stats_per_cluster = [s / nrounds for s in stats_per_cluster]
    stats /= nrounds

    # NOTE: these will have some "divide by zero", but they'll just become transparent
    thresh_stats /= thresh_count

    # }}}

    # # {{{ plot stats

    for i, s in enumerate(stats_per_cluster):
        filename = outfile.with_stem(f"{outfile.stem}_cluster_{i:02d}")
        with figure(filename, normalize=True, overwrite=force) as fig:
            trnlib.visualize_fraction(
                fig,
                g_syns,
                g_els,
                s.T,
                title="Fraction of Runs" if i % 2 == 0 else None,
            )

    # # }}}

    # # {{{ plot global average

    filename = outfile.with_stem(f"{outfile.stem}_average")
    with figure(filename, normalize=True, overwrite=force) as fig:
        trnlib.visualize_cluster_average(fig, g_syns, g_els, stats.T)

    filename = outfile.with_stem(f"{outfile.stem}_average_threshold")
    with figure(filename, normalize=True, overwrite=force) as fig:
        trnlib.visualize_cluster_average(
            fig,
            g_syns,
            g_els,
            thresh_stats.T,
            alpha=thresh_count.T / nrounds,
            fptick="0",
        )

    # # }}}

    # {{{ plot Pfeuty chi

    chi_avg = data["chi_mean"]
    chi_std = data["chi_std"]

    filename = outfile.with_stem(f"{outfile.stem}_chi_avg")
    with figure(filename, normalize=True, overwrite=force) as fig:
        trnlib.visualize_pfeuty_chi(
            fig,
            g_syns,
            g_els,
            chi_avg.T,
            cmap="managua_white",
        )

    filename = outfile.with_stem(f"{outfile.stem}_chi_std")
    with figure(filename, normalize=True, overwrite=force) as fig:
        trnlib.visualize_pfeuty_chi(
            fig,
            g_syns,
            g_els,
            chi_std.T,
            vmax=0.5,
            cmap="white_managua",
        )

    # }}}

    # {{{ plot bifurcation thing

    sec_g_els = [0.0, 0.009, 0.015, 0.03, 0.06]

    sec_g_syn = 0.4
    filename = outfile.with_stem(f"{outfile.stem}_g_syn_{sec_g_syn:.2f}")

    with figure(
        filename, nrows=len(sec_g_els), normalize=True, overwrite=force, pad_inches=0.1
    ) as fig:
        ax = fig.axes

        for k, sec_g_el in enumerate(sec_g_els):
            i = np.argmin(np.abs(sec_g_syn - g_syns))
            j = np.argmin(np.abs(sec_g_el - g_els))

            trnlib.visualize_cluster_stat(
                ax[k],
                np.array([s[i, j] for s in stats_per_cluster]),
                g_syn_label=f"{sec_g_syn:.2f}",
                g_el_label=f"{sec_g_el:.2f}",
                pad_title=0.0,
                xlabel=k == len(sec_g_els) - 1,
            )

    sec_g_syn = 0.45
    filename = outfile.with_stem(f"{outfile.stem}_g_syn_{sec_g_syn:.2f}")

    with figure(
        filename, nrows=len(sec_g_els), normalize=True, overwrite=force, pad_inches=0.1
    ) as fig:
        ax = fig.axes

        for k, sec_g_el in enumerate(sec_g_els):
            i = np.argmin(np.abs(sec_g_syn - g_syns))
            j = np.argmin(np.abs(sec_g_el - g_els))

            trnlib.visualize_cluster_stat(
                ax[k],
                np.array([s[i, j] for s in stats_per_cluster]),
                g_syn_label=f"{sec_g_syn:.2f}",
                g_el_label=f"{sec_g_el:.2f}",
                pad_title=0.0,
                xlabel=k == len(sec_g_els) - 1,
            )

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
        if len(args.filenames) == 1:
            ret = visualize(args.filenames[0], outfile=args.outfile, force=args.force)
        else:
            for filename in args.filenames:
                ret += visualize(filename, force=args.force)

        raise SystemExit(ret)
    else:
        raise SystemExit(
            main(
                args.size,
                tfinal=args.tfinal,
                adjacency=args.adjacency,
                max_workers=args.jobs,
                outfile=args.outfile,
                force=args.force,
            )
        )
