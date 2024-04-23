<h1 align="center">
<img src="doc/source/img/logo.png" width="300">
</h1>

<p align="center">
<img src="https://github.com/StanfordCBCL/svSuperEstimator/actions/workflows/codechecks.yml/badge.svg"/>
<img src="https://github.com/StanfordCBCL/svSuperEstimator/actions/workflows/test.yml/badge.svg"/>
<img src="https://github.com/StanfordCBCL/svSuperEstimator/actions/workflows/documentation.yml/badge.svg"/>
</p>

Sequential multi-fidelity parameter estimation and model calibration in
cardiovascular hemodynamics.

svSuperEstimator is an open-source framework to perform boundary conditions
parameter estimation and 0D-3D model calibrations for
cardiovascular simulation models. Both tasks can either be performed alone or
in serial as a sequential multi-fidelity tuning approach.

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
pip install git+https://github.com/StanfordCBCL/svSuperEstimator.git
```

The remaining dependencies have to be installed manually:

* [svZeroDSolver](https://github.com/SimVascular/svZeroDSolver): Install via pip.
* [svSlicer](https://github.com/StanfordCBCL/svSlicer): Build release version and specify path to executable in config file.
* [svSolver](https://github.com/SimVascular/svSolver): Build release version and specify path to executables in config file.

### For Contributors

You can install svSuperEstimator **with development related dependencies**
using:

```bash
pip install -e .[dev]
```

*If you are using the `zsh` shell, enter: `pip install -e ".[dev]"`*

## Running

After installing svSuperEstimator, it can be executed with:

```bash
estimate path/to/my_config.yaml
```

In the following, a few example configuration files will be shown.

### Sequential multi-fidelity tuning

```yaml
# -------------------------------------------------------------------------------------------
# SimVascular project
# -------------------------------------------------------------------------------------------
project: path/to/my_simvascular_project

# -------------------------------------------------------------------------------------------
# (Optional) Global settings for all tasks (can also be specified individually for each task)
# -------------------------------------------------------------------------------------------
global:
  num_procs: 48             # Number of processors
  overwrite: True           # Overwrite existing task results
  report_html: True         # Export task report as html
  report_files: False       # Export task reports as separate files
  debug: False              # Generate debug output and files
  core_run: True            # Toggle core run
  post_proc: True           # Toggle post processing

# -------------------------------------------------------------------------------------------
# Task configuration (configure one or more tasks)
# -------------------------------------------------------------------------------------------
tasks:
  multi_fidelity_tuning:                     # Run sequential multi-fidelity tuning
    name: my_case                           # Name of the task (determines the folder name)
    num_iter: 3                             # Number of multi-fidelity iterations to perform
    theta_obs: [...]                        # Ground truth theta
    y_obs: [...]                            # Target observations
    smc_num_particles: 10000                # Number of particles for Sequential-Monte-Carlo
    smc_num_rejuvenation_steps: 2           # Number of rejuvenation steps for Sequential-Monte-Carlo
    smc_resampling_threshold: 0.5           # Resampling threshold for Sequential-Monte-Carlo
    smc_noise_factor: 0.05                  # Relative noise on observations (relative standard deviation); can be calculated from the signal-to-noise-ratio with 1/sqrt(SNR)
    svpre_executable: path/to/svpre         # Path to svpre
    svsolver_executable: path/to/svsolver   # Path to svsolver
    svpost_executable: path/to/svpost       # Path to svsolver
    svslicer_executable: path/to/slicer       # Path to result slicer

# -------------------------------------------------------------------------------------------
# (Optional) Submit as slurm job (creates and submits slurm job)
# -------------------------------------------------------------------------------------------
slurm:
  partition: amarsden           # Partition to use
  python-path: path/to/python   # Python path where estimator is installed
  walltime: "48:00:00"          # Walltime
  qos: normal                   # QOS
  nodes: 2                      # Number of nodes
  mem: 32GB                     # Memory
  ntasks-per-node: 24           # Number of tasks per node
```

### Boundary condition parameter estimation

```yaml
# -------------------------------------------------------------------------------------------
# SimVascular project
# -------------------------------------------------------------------------------------------
project: path/to/my_simvascular_project

# -------------------------------------------------------------------------------------------
# (Optional) Global settings for all tasks (can also be specified individually for each task)
# -------------------------------------------------------------------------------------------
global: 
  num_procs: 4                  # Number of processors

# -------------------------------------------------------------------------------------------
# Task configuration (configure one or more tasks)
# -------------------------------------------------------------------------------------------
tasks:
  windkessel_tuning:                            # Run sequential Windkessel tuning
    name: my_case                               # Name of the task (determines the folder name)
    zerod_config_file: path/to/zerod_config.in  # Path to svZeroDSolver input file
    theta_obs: [...]                            # Ground truth theta
    y_obs: [...]                                # Target observations
    num_particles: 100                          # Number of particles for Sequential-Monte-Carlo
    num_rejuvenation_steps: 2                   # Number of rejuvenation steps for Sequential-Monte-Carlo
    resampling_threshold: 0.5                   # Resampling threshold for Sequential-Monte-Carlo
    noise_factor: 0.05                          # Relative noise on observations (relative standard deviation); can be calculated from the signal-to-noise-ratio with 1/sqrt(SNR)

# -------------------------------------------------------------------------------------------
# (Optional) Submit as slurm job (creates and submits slurm job)
# -------------------------------------------------------------------------------------------
slurm:
  partition: amarsden           # Partition to use
  python-path: path/to/python   # Python path where estimator is installed
  walltime: "48:00:00"          # Walltime
  qos: normal                   # QOS
  nodes: 2                      # Number of nodes
  mem: 32GB                     # Memory
  ntasks-per-node: 24           # Number of tasks per node
```

### Model calibration based on 3D result using Levenberg-Marquardt optimization

```yaml
# -------------------------------------------------------------------------------------------
# SimVascular project
# -------------------------------------------------------------------------------------------
project: path/to/my_simvascular_project

# -------------------------------------------------------------------------------------------
# (Optional) Global settings for all tasks (can also be specified individually for each task)
# -------------------------------------------------------------------------------------------
global:
  num_procs: 4                  # Number of processors

# -------------------------------------------------------------------------------------------
# Task configuration (configure one or more tasks)
# -------------------------------------------------------------------------------------------
tasks:
  model_calibration_least_squares:
    name: my_case
    zerod_config_file: path/to/zerod_config.in                  # Path to svZeroDSolver input file
    threed_solution_file: path/to/threed_centerline_result.vtp  # Path 3D result mapped on centerline
    centerline_padding: False                                   # Toggle padding over border nodes in centerline result
    calibrate_stenosis_coefficient: True                        # Specify whether to calibrate stenosis coefficient
    set_capacitance_to_zero: False                              # Specify whether capacitance should be set to 0
    initial_damping_factor: 1.0                                 # Initial damping factor for Levenberg-Marquardt optimization
    maximum_iterations: 100                                     # Maximum number of calibration iterations

# -------------------------------------------------------------------------------------------
# (Optional) Submit as slurm job (creates and submits slurm job)
# -------------------------------------------------------------------------------------------
slurm:
  partition: amarsden           # Partition to use
  python-path: path/to/python   # Python path where estimator is installed
  walltime: "48:00:00"          # Walltime
  qos: normal                   # QOS
  nodes: 2                      # Number of nodes
  mem: 32GB                     # Memory
  ntasks-per-node: 24           # Number of tasks per node
```


### Grid sampling of posterior

```yaml
# -------------------------------------------------------------------------------------------
# SimVascular project
# -------------------------------------------------------------------------------------------
project: path/to/my_simvascular_project

# -------------------------------------------------------------------------------------------
# (Optional) Global settings for all tasks (can also be specified individually for each task)
# -------------------------------------------------------------------------------------------
global: 
  num_procs: 12                                                 # Number of processors

# -------------------------------------------------------------------------------------------
# Task configuration (configure one or more tasks)
# -------------------------------------------------------------------------------------------
tasks:
  grid_sampling:
    name: my_grid_sampling                                      # Name of the task (determines the folder name)
    zerod_config_file: path/to/zerod_config.in                  # Path to svZeroDSolver input file
    theta_range: [...]                       # Range of theta to sample
    num_samples: 100                                            # Total number of samples
    noise_factor: 0.3                                           # Relative noise on observations (relative standard deviation); can be calculated from the signal-to-noise-ratio with 1/sqrt(SNR)
    y_obs: [...]                                                # Target observations
```