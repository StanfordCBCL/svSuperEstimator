<h1 align="center">
<img src="doc/source/img/logo.png" width="300">
</h1>

<p align="center">
<img src="https://github.com/SimVascular/svSuperEstimator/actions/workflows/codechecks.yml/badge.svg"/>
<img src="https://github.com/SimVascular/svSuperEstimator/actions/workflows/test.yml/badge.svg"/>
<img src="https://github.com/SimVascular/svSuperEstimator/actions/workflows/documentation.yml/badge.svg"/>
</p>

Multi-fidelity parameter estimation framework for cardiovascular fluid dynamics
simulations.

## Installation

It is highly recommended to use a virtual environment like
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).
After installing Miniconda you can create and activate a new environment using:

```bash
conda create -n estimator python=3.9
conda activate estimator
```

svSuperEstimator and most of the dependencies can be installed via pip:

```bash
pip install git+https://github.com/SimVascular/svSuperEstimator.git
```

The remaining dependencies have to be installed manually:

* [C++ svZeroDSolver](https://github.com/richterjakob/svZeroDSolver): Install via pip and build release version in local folder.
* [C++ 3D result slicer](https://gitlab.com/sanddorn/sanddorn-toolbox/-/tree/main/slicer): Build release version and specify path to executable in config file.
* **QUEENS** (currently not publicly available)

### For Contributers

You can install svSuperEstimator **with development related dependencies**
using:

```bash
pip install -e .[dev]
```

*If you are using the `zsh` shell, enter: `pip install -e ".[dev]"`*

### Sherlock

To run svSuperEstimator on sherlock, the following modules are required:

```
module purge
module load system
module load binutils/2.38
module load qt
module load openmpi
module load mesa
module load x11
```
