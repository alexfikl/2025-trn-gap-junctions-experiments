# Gap junction architecture and synchronization clusters in the thalamic reticular nuclei

.. |badge-license| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://spdx.org/licenses/MIT.html
    :alt: MIT License

This code accompanies the paper by the same name. It contains all the scripts used
to generate the figures and results in the paper.

# Dependencies

The dependencies for the project are listed in the `requirements.txt` file. All
dependencies are pinned to the latest version known to work (i.e. as published
with the paper). If you want to reproduce the results exactly, it is highly
recommended to use these dependencies. However, if the results do not reproduce
with newer versions, this is likely a bug!

Besides this, the user is expected to have

* An up to date C compiler (e.g. GCC>=14.0). This is required by the JiTCODE
  library to compile the native Python extensions that significantly speed up the
  simulations.
* [Optional] An up to date (>=2025) LaTeX installation. This dependency can be
  removed by setting `text.usetex: False` in `default.mplstyle`, as it is only
  required for plotting.

# Installation

It is recommended to install the (Python) dependencies in a [virtual
environment](https://docs.python.org/3/library/venv.html) to avoid conflicts
with other packages. The dependencies can then be installed using
[pip](https://pip.pypa.io/en/stable/user_guide/)
```bash
pip install -r requirements.txt
```
or
[uv](https://docs.astral.sh/uv/pip/packages/#installing-packages)
```bash
uv pip sync requirements.txt
```

# Reproducing the results

> [!WARNING]
> The results for Figure 4 and 6 take a longer time to run (i.e. a couple of hours).
> These all generate `npz` files that can then be used to (re)plot the results,
> but users will have to do so manually.

The included `justfile` has all the necessary invocations to reproduce the figures
in the paper. You can just run
```bash
just figure1
just figure2
just figure3
just figure4
just figure5
just figure6 set1
just figure6 set2
just figure6 set3
just figure6 set4
```
to get all the figures. The runs for Figure 6 will also produce Figure 7, 8, 9
and 10, since they are all obtained from the same run over the `(gsyn, gel)`
parameter plane.

