PYTHON := "nice python -X dev"
EXT := ".jpg"

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc("Reformat all source code")]
format: isort black pyproject justfmt

[doc("Run ruff isort fixes over the source code")]
isort:
    ruff check --fix --select=I *.py
    ruff check --fix --select=RUF022 *.py
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc("Run ruff format over the source code")]
black:
    ruff format *.py
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc("Run pyproject-fmt over the configuration")]
pyproject:
    {{ PYTHON }} -m pyproject_fmt \
        --indent 4 --max-supported-python "3.14" \
        pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc("Run just --fmt over the justfile")]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}
# {{{ linting

[doc("Run all linting checks over the source code")]
lint: typos reuse ruff ty

[doc("Run typos over the source code and documentation")]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc('Check REUSE license compliance')]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc("Run ruff checks over the source code")]
ruff:
    ruff check *.py
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run ty checks over the source code")]
ty:
    ty check *.py
    @echo -e "\e[1;32mty clean!\e[0m"

# }}}
# {{{ develop

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

[doc("Remove all generated files")]
purge:
    rm -rf *.png
    rm -rf .ruff_cache __pycache__

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt

# }}}
# {{{ figures

[doc("Generate plots for Figure 1")]
figure1:
    {{ PYTHON }} -O figure-evolve.py \
        --size 100 --tfinal 1.0 \
        --adjacency set1 \
        --outfile 'figure1_trn{{ EXT }}'

[doc("Generate plots for Figure 2")]
figure2:
    for name in 'set1' 'set2' 'set3' 'set4'; do \
        {{ PYTHON }} -O figure-graph-visualization.py \
            --force \
            --size 100 \
            --adjacency "${name}" \
            --outfile "figure2_trn_graph_${name}{{ EXT }}"; \
    done

[doc("Generate plots for Figure 3")]
figure3 jobs="8":
    {{ PYTHON }} -O figure-clustering.py \
        --jobs {{ jobs }} \
        --size 100 --tfinal 5.0 \
        --adjacency 'set1' --params 'figure3' \
        --outfile 'figure3_trn_cluster{{ EXT }}'

[doc("Generate plots for Figure 4")]
figure4 jobs="8":
    {{ PYTHON }} -O figure-statistics.py \
        --jobs {{ jobs }} \
        --size 100 --tfinal 5.0 \
        --adjacency 'set1' \
        --outfile 'figure4_trn_stat_set1{{ EXT }}'

[doc("Generate plots for Figure 5")]
figure5 jobs="8":
    {{ PYTHON }} -O figure-clustering.py \
        --jobs {{ jobs }} \
        --size 100 --tfinal 5.0 \
        --adjacency 'set1' --params 'figure5' \
        --outfile 'figure5_trn_cluster{{ EXT }}'

[doc("Generate plots for Figure 6")]
figure6 setname="set1" jobs="8":
    {{ PYTHON }} -O figure-statistics.py \
        --jobs {{ jobs }} \
        --size 100 --tfinal 5.0 \
        --adjacency {{ setname }} \
        --outfile 'figure6_trn_stat_{{ setname }}{{ EXT }}'

# }}}
