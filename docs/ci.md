# Continuous integration

The `CI` workflow runs on pushes to `main` and on pull requests. It tests Python 3.10 and 3.11, installs the CPU-only PyTorch wheel, checks Ruff lint and formatting, runs the unit suite, and executes the synthetic one-epoch CPU smoke path. It never downloads ESM2-650M weights or accesses private datasets.
