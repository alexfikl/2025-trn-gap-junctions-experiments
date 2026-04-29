# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger("trn")

# {{{ plotting


def set_plotting_defaults() -> None:
    """Sets default plotting settings from ``scienceplots``.

    The defaults can be overwritten with a local ``default.mplstyle`` file.
    """
    # NOTE: since v1.1.0 an import is required to import the styles
    import scienceplots  # noqa: F401

    dirname = pathlib.Path(__file__).parent
    plt.style.use(["science", "ieee"])

    mplstyle = dirname / "default.mplstyle"
    if mplstyle.exists():
        plt.style.use(dirname / "default.mplstyle")

    import os

    ext = os.environ.get("TRNLIB_FIGURE_FORMAT", "jpg")
    plt.rcParams["savefig.format"] = ext


set_plotting_defaults()


def add_title_box(ax: plt.Axes, title: str, pad: float = 2.0) -> None:
    """Add a little gray box around the title of the provided axis *ax*."""
    from matplotlib.patches import Rectangle

    height = 0.08
    rect = Rectangle(
        (0.0, 1.0),
        1.0,
        height,
        transform=ax.transAxes,
        facecolor="lightgray",
        edgecolor="k",
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.set_title(title, fontsize=40, y=(1.0 + 0.25 * height), va="center")

    # hack to make this fit
    fig = ax.get_figure()
    if fig is None:
        return

    try:
        engine = fig.get_layout_engine()
        if engine is None:
            return

        engine.set(h_pad=pad, w_pad=pad)  # ty: ignore[unknown-argument]
    except AttributeError:
        # NOTE: this was deprecated in matplotlib v3.6
        fig.set_constrained_layout_pads(h_pad=pad, w_pad=pad)  # ty: ignore[unresolved-attribute]


def make_custom_colormap(name: str, *, ncolors: int = 5) -> Any:
    if name == "prop_cycle":
        from matplotlib.colors import LinearSegmentedColormap

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return LinearSegmentedColormap.from_list("prop_cycle", colors[:ncolors], N=256)
    elif name == "managua_white":
        n = 256
        base_cmap = plt.get_cmap("managua", n)
        colors = base_cmap(np.linspace(0, 1, n))

        n_blend = 64
        white = np.ones(3, dtype=np.int32)
        for i in range(1, n_blend + 1):
            t = i / n_blend
            colors[-i, :3] = t * colors[-i, :3] + (1 - t) * white

        from matplotlib.colors import ListedColormap

        return ListedColormap(colors)
    elif name == "white_managua":
        n = 256
        base_cmap = plt.get_cmap("managua", n)
        colors = base_cmap(np.linspace(0, 1, n))

        n_blend = 64
        white = np.ones(3, dtype=np.int32)
        for i in range(n_blend):
            t = i / (n_blend - 1)
            colors[i, :3] = t * colors[i, :3] + (1 - t) * white

        from matplotlib.colors import ListedColormap

        return ListedColormap(colors)
    else:
        return name


def visualize_cluster_stat(
    ax: plt.Axes,
    z: Array,
    *,
    g_syn_label: str = "",
    g_el_label: str = "",
    cmap: str = "turbo",
    pad_title: float = 0.5,
    xlabel: str | bool = False,
) -> None:
    if cmap == "matlab":
        bar_colors = ["#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30"]
    elif cmap == "turbo":
        from matplotlib.colors import to_hex

        norm = plt.Normalize(vmin=0, vmax=4)
        cm = plt.get_cmap("turbo")
        bar_colors = [to_hex(cm(norm(i))) for i in range(5)]
    else:
        raise ValueError(f"Unknown color map: '{cmap}'")

    bar_labels = ["$FP$", "$1$", "$2$", "$3$", "$>3$"]

    ax.bar(
        bar_labels,
        z,
        label=bar_labels,
        color=bar_colors,
        edgecolor="k",
    )
    add_title_box(
        ax,
        rf"$g_{{\text{{syn}}}} = {g_syn_label}$, "
        rf"$g_{{\text{{el}}}} = {g_el_label}$",
        pad=pad_title,
    )

    ax.set_ylim((0.0, 1.0))
    if xlabel:
        ax.set_xlabel(xlabel if isinstance(xlabel, str) else "Clusters", fontsize=40)
    ax.set_ylabel("Fraction of Runs", fontsize=40)
    ax.grid(visible=False, which="both")
    ax.legend(loc="upper right", fontsize=32, framealpha=0.5)


def visualize_imshow_grid(
    fig: plt.Figure,
    x: Array,
    y: Array,
    z: Array,
    *,
    xlabel: str = r"$g_{\text{syn}}$ (mS/cm${}^2$)",
    ylabel: str = r"$g_{\text{el}}$ (mS/cm${}^2$)",
    title: str | None = None,
    cmap: str = "jet",
    alpha: float | Array | None = None,
    vmax: float = 1.0,
    shrink: float = 0.7,
    grid: bool = True,
    line_color: str = "w",
) -> Any:
    # make uniform grid for display
    xs = np.linspace(x[0], x[-1], x.size)
    ys = np.linspace(y[0], y[-1], y.size)

    # define extent
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)

    ax = fig.gca()
    im = ax.imshow(
        z,
        extent=extent,
        interpolation="none",
        aspect="auto",
        cmap=make_custom_colormap(cmap),
        alpha=alpha,
        vmin=0.0,
        vmax=vmax,
        origin="lower",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    indices = np.linspace(0, x.size - 1, 10, endpoint=True, dtype=np.int32)
    ax.set_xticks(xs[indices], [f"{xi:.2f}" for xi in x[indices]])
    indices = np.linspace(0, y.size - 1, 10, endpoint=True, dtype=np.int32)
    ax.set_yticks(ys[indices], [f"{yi:.2f}" for yi in y[indices]])
    ax.tick_params(which="minor", length=0)

    ax.set_box_aspect(1)
    ax.grid(visible=False, which="both")
    ax.tick_params(axis="x", rotation=45)

    if grid:
        for j in range(xs.size - 1):
            ax.axvline(xs[j] + 0.5 * dx, color=line_color, lw=1)

        for j in range(ys.size - 1):
            ax.axhline(ys[j] + 0.5 * dy, color=line_color, lw=1)

    return im


def visualize_fraction(
    fig: plt.Figure,
    x: Array,
    y: Array,
    z: Array,
    *,
    title: str | None = None,
    cmap: str = "binary",
) -> None:
    """Visualize data using ``imshow``.

    :arg x: an array of size ``(n,)`` describing one axis
    :arg y: an array of size ``(n,)`` describing the other axis.
    :arg z: an array of size ``(n, n)`` with values in ``[0, 1]``.
    :arg title: colorbar title. If no title is given, the colorbar is not added
        to the figure.
    """

    ax = fig.gca()
    im = visualize_imshow_grid(fig, x, y, z, cmap=cmap, vmax=1.0, grid=False)

    if title:
        cbar = fig.colorbar(im, ax=ax, shrink=0.73, pad=0.01)
        cbar.set_label(title)
        cbar.ax.yaxis.label.set_rotation(-90)
        cbar.ax.yaxis.labelpad = 30


def visualize_pfeuty_chi(
    fig: plt.Figure,
    x: Array,
    y: Array,
    z: Array,
    *,
    vmax: float = 1.0,
    cmap: str = "managua",
) -> None:
    """Visualize data using ``imshow``.

    :arg x: an array of size ``(n,)`` describing one axis
    :arg y: an array of size ``(n,)`` describing the other axis.
    :arg z: an array of size ``(n, n)`` with values in ``[0, 1]``.
    :arg title: colorbar title. If no title is given, the colorbar is not added
        to the figure.
    """

    ax = fig.gca()
    im = visualize_imshow_grid(
        fig,
        x,
        y,
        z,
        cmap=cmap,
        vmax=vmax,
        line_color="k",
        grid=False,
    )
    fig.colorbar(im, ax=ax, shrink=0.79, pad=0.01)


def visualize_cluster_average(
    fig: plt.Figure,
    x: Array,
    y: Array,
    z: Array,
    *,
    title: str | None = None,
    fptick: str = "FP",
    alpha: float | Array | None = None,
    cmap: str = "turbo",
) -> None:
    ax = fig.gca()

    vmax = 4
    im = visualize_imshow_grid(
        fig,
        x,
        y,
        z,
        cmap=cmap,
        alpha=alpha,
        vmax=vmax,
        line_color="k",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.79, pad=0.01, ticks=np.arange(vmax + 1))
    cbar.ax.set_yticklabels([fptick, "1", "2", "3", "$>3$"])


# }}}


# {{{ model

T = TypeVar("T")


def from_dict(cls: type[T], d: dict[str, Any]) -> T:
    from dataclasses import fields, is_dataclass

    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    from typing import get_type_hints

    kwargs = {}
    type_hints = get_type_hints(cls)

    for f in fields(cls):
        if not f.init:
            continue

        fname = f.name
        ftype = type_hints.get(f.name, f.type)
        value = d.get(fname)
        if is_dataclass(ftype) and isinstance(value, dict):
            value = from_dict(ftype, value)

        kwargs[f.name] = value

    return cls(**kwargs)


@dataclass(frozen=True)
class GatingFunction:
    r"""Sigmoid gating function.

    .. math::

        S(V; \theta, \sigma) = \frac{1}{1 + \exp\left(-(V - \theta) / \sigma\right)}.
    """

    theta: float
    sigma: float

    def exp(self, V: Array, xp: Any = np) -> Array:
        return cast("Array", np.vectorize(xp.exp)(-(V - self.theta) / self.sigma))

    def __call__(self, V: Array, xp: Any = np) -> Array:
        return 1.0 / (1.0 + self.exp(V, xp=xp))


@dataclass(frozen=True)
class Parameters:
    """Dimensional parameters for the Golomb-Rinzel model."""

    C: float
    """Membrane capacitance (F/cm^2)."""
    g_Ca: float
    r"""Maximum conductance of the :math:`\mathrm{Ca}^{2+}` ionic current (mS/cm^2)."""
    g_L: float
    """Maximum conductance of the leak current (mS/cm^2)."""
    g_syn: float
    """Maximum synaptic conductance (mS/cm^2)."""
    g_el: float
    """Maximum gap junction conductance (mS/cm^2)."""
    V_Ca: float
    r"""Reversal potential for the :math:`\mathrm{Ca}^{2+}` ion channel (mV)."""
    V_L: float
    r"""Reversal potential for the leak current (mV)."""
    V_syn: float
    """Reversal potential for the synaptic current (mV)."""
    kr: float
    """Channel closing rate (1/s)."""
    kf: float
    """Channel opening rate (1/s)."""
    phi: float
    """Scaling factor for the kinetics of :math:`h` (1/s)."""


@dataclass(frozen=True)
class Model:
    """Complete neural network model based on the Golomb-Rinzel neurons."""

    param: Parameters
    """Physical parameters for the model."""

    m_inf: GatingFunction
    """Voltage gating function for :math:`m`."""
    h_inf: GatingFunction
    """Voltage gating function for :math:`h`."""
    s_inf: GatingFunction
    """Voltage gating function for :math:`s`."""
    k_inf: GatingFunction
    """Gating function parameters for :math:`k_h`."""

    A: Array
    """Binary adjacency matrix for gap junction connections."""
    M: Array = field(init=False)
    """Total number of gap junctions made on each neuron."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "M", np.sum(self.A, axis=1))

    @property
    def n(self) -> int:
        return self.M.size

    @property
    def V_threshold(self) -> float:  # noqa: N802
        return -40.0

    def kh(self, V: Array, xp: Any = np) -> Array:
        phi = self.param.phi
        return phi * self.k_inf.exp(V, xp=xp) / self.h_inf(V, xp=xp)

    @classmethod
    def defaults(
        cls,
        n: int,
        *,
        g_syn: float = 0.2,
        g_el: float = 0.0,
        A: Array | None = None,
    ) -> Model:
        if A is None:
            # NOTE: by default, we have an all-to-all gap junction connection matrix
            A = np.ones((n, n))
            np.fill_diagonal(A, 0.0)

        param = Parameters(
            C=1.0e-3,
            g_Ca=0.5,
            g_L=0.05,
            g_syn=0.2,
            g_el=0.0,
            V_Ca=120.0,
            V_L=-60,
            V_syn=-80,
            kr=58.0,
            kf=1.0e3,
            phi=2.0e3,
        )

        model = Model(
            param=param,
            m_inf=GatingFunction(theta=-65.0, sigma=7.8),
            h_inf=GatingFunction(theta=-81.0, sigma=-11.0),
            s_inf=GatingFunction(theta=-45.0, sigma=2.0),
            k_inf=GatingFunction(theta=-162.3, sigma=17.8),
            A=A,
        )

        return model


def model_source(model: Model, t: float, x: Array) -> Array:
    r"""Compute the Rinzel model.

    .. math::

        \begin{aligned}
        C \frac{d V_i}{d t} & =
            I_{Ca} + I_L + I_syn + I_el, \\
        \frac{d h_i}{d t} & =
            k_h(V_i) (h_\infty(V_i) - h_i), \\
        \frac{d s_i}{d t} & =
            k_f s_\infty(V) (1 - s_i) - k_r s_i.
        \end{aligned}
    """
    if x.dtype == object:
        import symengine as xp
    else:
        import numpy as xp

    # unpack variables
    n = model.n
    A, M = model.A, model.M

    p = model.param
    C = p.C
    g_Ca, g_L, g_syn, g_el = p.g_Ca, p.g_L, p.g_syn, p.g_el
    V_Ca, V_L, V_syn = p.V_Ca, p.V_L, p.V_syn
    kr, kf = p.kr, p.kf

    # separate variables
    V = x[:n]
    h = x[n : 2 * n]
    s = x[2 * n :]
    assert V.shape == h.shape == s.shape

    # compute functions
    m_inf = model.m_inf(V, xp=xp)
    h_inf = model.h_inf(V, xp=xp)
    s_inf = model.s_inf(V, xp=xp)
    kh = model.kh(V, xp=xp)

    # compute model
    I_Ca = g_Ca * m_inf**3 * h * (V - V_Ca)
    I_L = g_L * (V - V_L)
    I_syn = g_syn / (n - 1) * (V - V_syn) * (np.sum(s) - s)

    # NOTE: this is A_ij (V_i - V_j)
    if isinstance(g_el, (float, int)) and abs(g_el) < 1.0e-14:
        I_el = 0 * V
    else:
        # NOTE: (A @ V)_i == 0 whenever M_i == 0, so we just set I_el to zero there too
        AV = M * V - A @ V
        I_el = np.where(M == 0, AV, g_el / M * AV)

    return np.hstack([
        -(I_Ca + I_L + I_syn + I_el) / C,
        kh * (h_inf - h),
        kf * s_inf * (1 - s) - kr * s,
    ])


# }}}


# {{{ solve ivp


@dataclass(frozen=True)
class Result:
    t: Array
    y: Array


@contextmanager
def jitcode_module(suffix: str) -> Iterator[pathlib.Path]:
    import tempfile

    tempdir = pathlib.Path(tempfile.gettempdir())
    suffix = suffix.replace(".", "_")
    path = tempdir / f"jitcode_trn_{suffix}.so"

    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


def solve_ivp(
    model: Model,
    tspan: tuple[float, float],
    y0: Array,
    *,
    method: str = "RK45",
    backend: str = "jitcode",
    dt: float = 1.0e-3,
    # NOTE: larger tolerances may not work for some systems, i.e. the cluster
    # synchronization is different and most likely wrong!
    atol: float = 1.0e-8,
    rtol: float = 1.0e-10,
    module_location: pathlib.Path | None = None,
    verbose: bool = True,
) -> Result:
    if backend == "scipy":
        from scipy.integrate import solve_ivp

        t_start = time.time()
        func = partial(model_source, model)
        result = solve_ivp(  # ty: ignore[no-matching-overload]
            func,
            tspan,
            y0,
            method=method,
            atol=atol,
            rtol=rtol,
        )
        if verbose:
            log.info("Solve time: %.3fs (on %s) (scipy)", time.time() - t_start, tspan)

        t = result.t
        y = result.y
    elif backend == "jitcode":
        import jitcode

        # create symbolic variables
        n = model.n
        y = np.empty(y0.size, dtype=object)
        for i in range(n):
            y[i] = jitcode.y(i)
            y[i + n] = jitcode.y(i + n)
            y[i + 2 * n] = jitcode.y(i + 2 * n)

        import symengine as sp

        control_pars = (sp.Symbol("g_syn"), sp.Symbol("g_el"))
        control_vals = (model.param.g_syn, model.param.g_el)
        model = replace(
            model,
            param=replace(model.param, **{v.name: v for v in control_pars}),
        )

        # create integrator
        f = model_source(model, jitcode.t, y)
        if module_location and module_location.exists():
            ode = jitcode.jitcode(
                f,
                control_pars=control_pars,
                n=y.size,
                verbose=False,
                module_location=str(module_location),
            )
        else:
            ode = jitcode.jitcode(
                f,
                control_pars=control_pars,
                n=y.size,
                verbose=False,
            )

            if module_location is not None:
                t_start = time.time()
                # ode.generate_f_C(simplify=False, chunk_size=50)
                # ode.compile_C(omp=True, modulename=module_location.stem)
                newfilename = ode.save_compiled(str(module_location), overwrite=True)
                if verbose:
                    log.info("Compilation time: %.3fs.", time.time() - t_start)

                if newfilename != str(module_location):
                    log.warning(
                        "jitcode saved compiled module in different file: '%s'. "
                        "This may cause performance issues since it will be recompiled",
                        newfilename,
                    )

        t_start = time.time()

        # set parameters
        ode.set_integrator(method, atol=atol, rtol=rtol)
        ode.set_parameters(control_vals)
        ode.set_initial_value(y0, tspan[0])

        # evolve
        t = np.arange(tspan[0], tspan[1], dt)
        y = np.empty((y0.size, t.size), dtype=y0.dtype)
        for i in range(t.size):
            y[:, i] = ode.integrate(t[i])

        if verbose:
            log.info(
                "Solve time: %.3fs (on %s) (jitcode)", time.time() - t_start, tspan
            )
    else:
        raise ValueError(f"Unknown integrator: '{method}'")

    return Result(t=t, y=y)


# }}}


# {{{ clustering


@dataclass(frozen=True)
class Cluster:
    clusterid: int
    """The number of clusters in this block."""
    count: int
    """The number of times this cluster count was achieved."""
    tcount: int
    """The number of times this cluster count was achieved, while the signal was
    over threshold."""
    x0: Array
    """A representative initial condition that got this cluster count."""
    cluster_indices: tuple[Array, ...]
    """A list of indices for each cluster that was formed."""
    pfeuty_chi: float
    """Measure of synchrony based on Pfeuty (2007)."""

    window: int
    """Window size used to determine the clusters."""
    tspan: tuple[float, float]
    """Time span over which the solution was evolved to get the clusters."""


def is_fixed_point(x: Array, eps: float = 1.0e-4) -> bool:
    rtol = np.max(la.norm(np.diff(x), axis=1) / la.norm(x, axis=1, ord=np.inf))
    return bool(rtol < eps)


def get_cluster_error(cx: Cluster, V: Array) -> float:
    error = 0.0

    for i in cx.cluster_indices:
        if not i.size:
            continue

        Vi = V[i]

        # fmt: off
        e_i = np.max(
            la.norm(Vi[:, None, :] - Vi[None, :, :], axis=2)
            / np.max(la.norm(Vi, axis=1, ord=np.inf))
        )
        # fmt: on

        error = max(error, e_i)

    return error


def determine_clusters(
    model: Model, result: Result, window: int | None = None
) -> Cluster:
    if window is None:
        window = result.t.size // 8

    from orbitkit.clusters import find_clusters_from_timeseries
    from orbitkit.synchrony import pfeuty_chi

    # cut out voltage
    n = model.n
    V = result.y[:n, -window:]

    # find clusters
    clusters = find_clusters_from_timeseries(result.y[:n], window=window)

    # check if the solution is actually a fixed point
    clusterid = len(clusters)
    if is_fixed_point(V, eps=1.0e-4):
        clusterid = 0

    # determine if we are over or under the threshold
    offset = 0 if np.max(V) < model.V_threshold else 1

    # compute global synchrony
    chi = pfeuty_chi(V)

    return Cluster(
        clusterid=clusterid,
        count=1,
        tcount=offset,
        x0=result.y[:, 0],
        cluster_indices=clusters,
        pfeuty_chi=chi,
        window=window,
        tspan=(result.t[0], result.t[-1]),
    )


try:
    MAX_ATTEMPTS = int(os.environ.get("TRNLIB_MAX_ATTEMPTS", "0"))
except ValueError:
    log.error(
        "Failed to parse 'TRNLIB_MAX_ATTEMPTS': '%s'",
        os.environ.get("TRNLIB_MAX_ATTEMPTS", "0"),
    )
    MAX_ATTEMPTS = 0


def find_clusters_for_model(
    model: Model,
    index: tuple[int, int, int],
    *,
    g_syns: Array,
    g_els: Array,
    tspan: tuple[float, float],
    module_location: pathlib.Path,
    rng: np.random.Generator | None = None,
    attempt: int = 0,
    window: int | None = None,
    rtol: float = 1.0e-6,
) -> tuple[int, int, Cluster]:
    if rng is None:
        rng = np.random.default_rng(index)

    t_start = time.time()

    n = model.n
    x0 = np.hstack([
        rng.uniform(-60.0, 0.0, n),
        rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n),
    ])
    assert x0.shape == (3 * n,)

    i, j, k = index
    g_syn = g_syns[i]
    g_el = g_els[j]
    model = replace(model, param=replace(model.param, g_syn=g_syn, g_el=g_el))

    solution = solve_ivp(
        model,
        tspan,
        x0,
        module_location=module_location,
        verbose=False,
    )
    result = determine_clusters(model, solution, window=window)

    log.info(
        "[%3d] Running for g_syn = %.5f g_el = %.5f | "
        "Found %d clusters _%s_ threshold (time %.3fs)",
        k,
        g_syn,
        g_el,
        result.clusterid,
        "over" if result.tcount == 1 else "under",
        time.time() - t_start,
    )

    # NOTE: try to increase tspan if the solution doesn't look converged
    error = get_cluster_error(result, solution.y[:n, -result.window :])
    if error > rtol:
        if attempt < MAX_ATTEMPTS:
            log.warning("Cluster error too large (retrying): %.8e", error)
            _, _, result = find_clusters_for_model(
                model,
                index,
                g_syns=g_syns,
                g_els=g_els,
                tspan=(tspan[0], 2 * tspan[1]),
                module_location=module_location,
                attempt=attempt + 1,
            )
        else:
            log.warning("Cluster error too large (oh noes): %.8e", error)

    return (i, j, result)


# }}}


# {{{ adjacency


def stringify_adjacency(mat: Array, fmt: str = "box") -> str:
    if fmt == "box":
        symbols = {0: " ◻ ", 1: " ◼ "}

        return "\n".join(
            "".join(symbols[int(mat[i, j] != 0)] for j in range(mat.shape[1]))
            for i in range(mat.shape[0])
        )
    elif fmt == "latex":
        lines = []

        lines.append(r"\begin{bmatrix}")
        for i in range(mat.shape[0]):
            lines.append(" & ".join(str(mij) for mij in mat[i]))
            lines.append(r"\\")
        lines.append(r"\end{bmatrix}")

        return "\n".join(lines)
    else:
        raise ValueError(f"Unknown format: '{fmt}'")


def make_partition_from_fractions(m: int, ps: Array) -> Array:
    if not all(0 <= p <= 1.0 for p in ps):
        raise ValueError(f"Probabilities need to be in [0, 1]: {ps}")

    if not abs(np.sum(ps) - 1.0) < 1.0e-12:
        raise ValueError(f"Probabilities must sum up to 1: {ps} (sum {np.sum(ps)})")

    # compute partition
    weights = ps * m
    ms = np.floor(weights).astype(np.int32)
    leftover = m - np.sum(ms)

    # add 1 to smaller groups
    frac = weights - ms
    order = np.argsort(-frac)
    ms[order[:leftover]] += np.sign(frac[:leftover]).astype(np.int32) * 1

    return np.array(ms)


def make_adjacency_all(n: int, *, dtype: Any = None) -> Array:
    if dtype is None:
        dtype = np.int32

    result = np.ones((n, n), dtype=dtype)
    np.fill_diagonal(result, 0.0)

    return result


def make_adjacency_groups(
    ms: Array,
    gaps: int | Array,
    *,
    dtype: Any = None,
) -> tuple[Array, Array, Array]:
    if dtype is None:
        dtype = np.int32

    if isinstance(gaps, int):
        gaps = np.array([gaps] * ms.size)

    n = int(np.sum(ms) + np.sum(gaps))
    if ms.shape != gaps.shape:
        raise ValueError(
            "Cluster sizes and gap sizes must have the same shape: "
            f"got {ms.shape} and {gaps.shape}"
        )

    i = 0
    result = np.zeros((n, n), dtype=dtype)

    for m, g in zip(ms, gaps, strict=True):
        result[i : i + m, i : i + m] = 1.0
        i += m + g

    np.fill_diagonal(result, 0.0)
    return result, ms, gaps


def make_adjacency_set(
    name: str,
    n: int,
    *,
    dtype: Any = None,
) -> tuple[Array, Array, Array]:
    if name == "set1":
        return make_adjacency_set1(n, dtype=dtype)
    elif name == "set2":
        return make_adjacency_set2(n, dtype=dtype)
    elif name == "set3":
        return make_adjacency_set3(n, dtype=dtype)
    elif name == "set4":
        return make_adjacency_set4(n, dtype=dtype)
    else:
        raise ValueError(f"Unknown example set: '{name}'")


def make_adjacency_set1(n: int, *, dtype: Any = None) -> tuple[Array, Array, Array]:
    """An all-to-all gap junction adjacency matrix."""
    ms = np.array([n])
    gaps = np.array([0])

    result = make_adjacency_groups(ms, gaps, dtype=dtype)
    assert result[0].shape == (n, n)

    return result


def make_adjacency_set2(n: int, *, dtype: Any = None) -> tuple[Array, Array, Array]:
    """A two group adjacency matrix (10% and 90%)."""
    ps = np.array([0.9, 0.1])
    gaps = np.array([2, 2])

    m = n - np.sum(gaps)
    ms = make_partition_from_fractions(m, ps)

    result = make_adjacency_groups(ms, gaps, dtype=dtype)
    assert result[0].shape == (n, n)

    return result


def make_adjacency_set3(n: int, *, dtype: Any = None) -> tuple[Array, Array, Array]:
    ps = np.array([0.27, 0.40, 0.33])
    gaps = np.array([n // 15] * ps.size)

    m = n - np.sum(gaps)
    ms = make_partition_from_fractions(m, ps)

    result = make_adjacency_groups(ms, gaps, dtype=dtype)
    assert result[0].shape == (n, n)

    return result


def make_adjacency_set4(n: int, *, dtype: Any = None) -> tuple[Array, Array, Array]:
    ps = np.array([0.29, 0.25, 0.33, 0.13])
    gaps = np.array([n // 15] * ps.size)

    m = n - np.sum(gaps)
    ms = make_partition_from_fractions(m, ps)

    result = make_adjacency_groups(ms, gaps, dtype=dtype)
    assert result[0].shape == (n, n)

    return result


def generate_random_gap_junction_clusters(
    rng: np.random.Generator,
    n: int,
    m: int,
    *,
    alpha: float = 1.0,
    mean: int = 9,
    maximum: int = 24,
    maxiter: int = 512,
) -> Array:
    x = np.array([n // m] * m, dtype=np.int64)

    # FIXME: this seems like it'll have mean *mean* only if n > mean * m?
    i = 0
    smax = min(n, mean * m)
    while i < maxiter:
        p = rng.dirichlet((alpha,) * m)
        x = rng.multinomial(smax, p)
        if np.max(x) < maximum and np.min(x) >= 2:
            break

        i += 1

    return x


def make_adjacency_groups_random(
    n: int,
    m: int,
    *,
    dtype: Any = None,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array]:
    """Generate random gap junction groups that have vaguely realistic sizes.

    We know that gap junction clusters have an average size of 9 and can go up
    to 24 nodes in the TRN. This function generates *m* clusters in that range
    for a total of *n* nodes. Between the clusters, *max_gap* nodes are not
    connected in any gap junction.

    :arg n: total number of nodes in the network.
    :arg m: number of gap junction clusters.
    :arg max_gap: maximum gap between the clusters.
    """
    if rng is None:
        rng = np.random.default_rng()

    # generate gap junction clusters
    ms = generate_random_gap_junction_clusters(rng, n, m)
    leftover = n - np.sum(ms)

    # generate random gaps
    cuts = rng.choice(np.arange(1, leftover), size=m - 1, replace=False)
    pts = np.concatenate(([0], np.sort(cuts), [leftover]))
    gaps = np.diff(pts)

    log.info("Cluster sizes: %s", ms)
    log.info("Gap sizes:     %s", gaps)

    # create adjacency matrix
    result = make_adjacency_groups(ms, gaps, dtype=dtype)
    assert result[0].shape == (n, n)

    return result


def make_adjacency_set_random(
    name: str,
    n: int,
    *,
    dtype: Any = None,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array]:
    if rng is None:
        rng = np.random.default_rng()

    if name == "set1":
        return make_adjacency_set1(n, dtype=dtype)
    elif name == "set2":
        return make_adjacency_groups_random(n, 2, dtype=dtype, rng=rng)
    elif name == "set3":
        return make_adjacency_groups_random(n, 3, dtype=dtype, rng=rng)
    elif name == "set4":
        return make_adjacency_groups_random(n, 4, dtype=dtype, rng=rng)
    else:
        raise ValueError(f"Unknown example set: '{name}'")


# }}}
