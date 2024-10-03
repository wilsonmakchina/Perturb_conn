import run_inference as ri

# this file will fit a model to synthetically generated data
# the data will be generated from a model with a specific sparsity pattern on the weights
param_name = synethtic_test.yml
save_folder = 'trained_models'
ri.fit_synthetic(param_name, save_folder)