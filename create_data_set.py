import numpy as np
import time
from pathlib import Path
import loading_utilities as lu
import pickle

run_params = lu.get_run_params(param_name='submission_params/create_data_set.yml')
data_path = Path(run_params['data_path'])
print('Data path:', data_path)
start_index = run_params['start_index']
filter_size = run_params['filter_size']
correct_photobleach = run_params['correct_photobleach']
interpolate_nans = run_params['interpolate_nans']
upsample_factor = run_params['upsample_factor']
randomize_cell_ids = run_params['randomize_cell_ids']
rng = np.random.default_rng(run_params['random_seed'])
sample_rate = 2 * upsample_factor

preprocess_filename = 'funcon_preprocessed_data.pkl'

# find each data set and make a numpy file version of it for easy access
for i in sorted(data_path.rglob('neural_data.txt'))[::-1]:
    preprocess_path = i.parent / preprocess_filename

    this_emissions = np.loadtxt(i, delimiter=',')

    if not interpolate_nans:
        # the data from francesco's paper has interpolated over nans. We usually add these nan's back in as the
        # LDS imputation of data is better than linear interpolation
        this_nan_mask = np.loadtxt(i.parent / 'nan_mask.txt', delimiter=',').astype(bool)
        this_emissions[this_nan_mask] = np.nan

    this_cell_ids = list(np.loadtxt(i.parent / 'cell_ids.txt', delimiter=',', dtype=str))
    this_cell_ids = [str(i) for i in this_cell_ids]
    this_cell_ids = ['' if i == '""' else i for i in this_cell_ids]

    # load stimulation data
    this_stim_cell_ids = np.loadtxt(i.parent / 'stim_cell_indicies.txt', delimiter=',', dtype=int)
    this_stim_volume_inds = np.loadtxt(i.parent / 'stim_frame_indicies.txt', delimiter=',', dtype=int)

    this_inputs = np.zeros_like(this_emissions)
    nonnegative_inds = (this_stim_cell_ids != -1) & (this_stim_cell_ids != -2) & (this_stim_cell_ids != -3)
    this_stim_volume_inds = this_stim_volume_inds[nonnegative_inds]
    this_stim_cell_ids = this_stim_cell_ids[nonnegative_inds]
    this_inputs[this_stim_volume_inds, this_stim_cell_ids] = 1

    start = time.time()
    this_emissions, this_inputs = lu.preprocess_data(this_emissions, this_inputs, start_index=start_index,
                                                     correct_photobleach=correct_photobleach,
                                                     filter_size=filter_size, upsample_factor=upsample_factor)

    if randomize_cell_ids:
        measured_neurons = np.mean(np.isnan(this_emissions), axis=0) <= 0.5

        # scramble IDs, but preserve scramble between measured and unmeasured neurons
        measured_neuron_inds = np.where(measured_neurons)[0]
        measured_neuron_inds_scrambled = rng.permutation(measured_neuron_inds)
        unmeasured_neuron_inds = np.where(~measured_neurons)[0]
        unmeasured_neuron_inds_scrambled = rng.permutation(unmeasured_neuron_inds)

        new_cell_ids = np.array(this_cell_ids.copy())
        this_cell_ids_array = np.array(this_cell_ids.copy())
        new_cell_ids[measured_neuron_inds] = this_cell_ids_array[measured_neuron_inds_scrambled]
        new_cell_ids[unmeasured_neuron_inds] = this_cell_ids_array[unmeasured_neuron_inds_scrambled]
        this_cell_ids = list(new_cell_ids)

    if interpolate_nans:
        full_nan_loc = np.all(np.isnan(this_emissions), axis=0)
        interp_emissions = lu.interpolate_over_nans(this_emissions[:, ~full_nan_loc])[0]
        this_emissions[:, ~full_nan_loc] = interp_emissions

    preprocessed_file = open(preprocess_path, 'wb')
    pickle.dump({'emissions': this_emissions, 'inputs': this_inputs, 'cell_ids': this_cell_ids, 'sample_rate': sample_rate}, preprocessed_file)
    preprocessed_file.close()

    print('Data set', i.parent, 'preprocessed')
    print('Took', time.time() - start, 's')
print('All data sets preprocessed')

