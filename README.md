# Bridging the gap between the connectome and whole-brain activity in C. elegans
https://doi.org/10.1101/2024.09.22.614271

Codebase for fitting and analyzing dynamical models from the paper

# Installation:
Clone the project

```git clone https://github.com/Nondairy-Creamer/Creamer_LDS_2024```

Set up your python environment with pip. Within that environment run

```pip install -e .```

All requirements are listed in the setup.py file if you want to install them manually

## Usage
The files in the quick_start_examples folder will show you how to load the models from the paper and use the models to predict STAMs, correlations, and reconstruct missing neurons.

If you would like to train a model on synthetic data, simply run main.txt. To see the available parameters for fitting and generating synethic data look at submission_params/syn_test.txt

### Fitting experimental data
To fit a new model on the data from Randi et al 2023

Download the data from here: https://osf.io/qxhjd/
unzip the folder and place it somewhere convenient

modify submission_params/create_data_set.yml and change data_path: to the path of the saved data

run
```python create_data_set.py```

### Fitting locally (est time ~40 hours)
Take a look at submission_params/exp_test.yml. You can use this file to set the parameters of the model and the size of the data set. You can either edit exp_test or make a new .yml file with your own parameters.

To fit a model run

```python main.py submission_params/exp_test.yml```

Takes ~40 hours on a desktop to fit across 80 animals and 156 neurons .

If you want to run with debugging, change line 104 in main.py from

```param_name = 'submission_params/syn_test.yml'```

to

```param_name = 'submission_params/exp_test.yml'```

and run main.py in you preferred IDE

### Fitting locally parallelized across CPUs (est time ~10 hours parallelized across 10 CPUs)
If you computer has multiple CPUs you can reduce computation time using mpi4py
First disable multithreading in numpy by running the following in the terminal

```export MKL_NUM_THREADS=1```

```export OMP_NUM_THREADS=1```

```export NUMEXPR_NUM_THREADS=1```

On Linux, to fit a model run
```mpiexec -n <num_cpus> python -m mpi4py main.py submission_params/exp_test.yml```

### Fit a model on an HPC cluster using SLURM (est time ~4 hours)
Examine submissions_params/slurm_example.yml. Every entry in the slurm dict will be fed direclty to slurm. You can add / remove necessary commands as necessary

Note that this will depend on the exact specifications of your HPC cluster. You should be aware what size your nodes are to properly set 'cpus_per_task', 'tasks_per_node', and 'nodes'. Check main.py in the section after ```if 'slurm' in run_params.keys()``` to see how these are submitted.

Install the code according to instructions from the HPC specifications. Then run
```python main.py submission_params/slurm_script.yml```
