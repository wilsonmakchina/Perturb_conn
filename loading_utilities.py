import numpy as np
from pathlib import Path
import yaml
import pickle
# import tmac.preprocessing as tp
import analysis_utilities as au
import warnings
import os
from scipy import interpolate
import scipy.optimize as sio


# utilities for loading and saving the data
def get_run_params(param_name):
    # load in the parameters for the run which dictate how many data sets to use,
    # or how many time lags the model will fit etc

    with open(param_name, 'r') as file:
        params = yaml.safe_load(file)

    return params


def check_input_format(data):
    if type(data) is not np.ndarray:
        raise Exception('The red and green matricies must be the numpy arrays')

    if data.ndim != 1 and data.ndim != 2:
        raise Exception('The red and green matricies should be 1 or 2 dimensional')

    if data.ndim == 1:
        data = data[:, None]

    return data


def interpolate_over_nans(input_mat, t=None):
    """ Function to interpolate over NaN values along the first dimension of a matrix

    Args:
        input_mat: numpy array, [time, neurons]
        t: optional time vector, only useful if input_mat is not sampled regularly in time

    Returns: Interpolated input_mat, interpolated time
    """

    input_mat = check_input_format(input_mat)

    # if t is not specified, assume it has been sampled at regular intervals
    if t is None:
        t = np.arange(input_mat.shape[0])

    output_mat = np.zeros(input_mat.shape)
    output_mat[:] = np.nan

    # calculate the average sample rate and uses this to create an interpolated t
    sample_rate = 1 / np.mean(np.diff(t, axis=0))
    t_interp = np.arange(input_mat.shape[0]) / sample_rate

    # loop through each column of the data and interpolate them separately
    for c in range(input_mat.shape[1]):
        # check if all the data is nan and skip if it is
        if np.all(np.isnan(input_mat[:, c])):
            # print('column ' + str(c) + ' is all NaN, skipping')
            continue

        # find the location of all nan values
        no_nan_ind = ~np.isnan(input_mat[:, c])

        # remove nans from t and the data
        no_nan_t = t[no_nan_ind]
        no_nan_data_mat = input_mat[no_nan_ind, c]

        # interpolate values linearly
        interp_obj = interpolate.interp1d(no_nan_t, no_nan_data_mat, kind='linear', fill_value='extrapolate')
        output_mat[:, c] = interp_obj(t_interp)

    return output_mat, t_interp


def photobleach_correction(time_by_neurons_full, t=None, num_exp=1, fit_offset=False):
    """ Function to fit an exponential with a shared tau to all the columns of time_by_neurons

    This function fits the function A*exp(-t / tau) to the matrix time_by_neurons. Tau is a single time constant shared
    between every column in time_by_neurons. A is an amplitude vector that is fit separately for each column. The
    correction is time_by_neurons / exp(-t / tau), preserving the amplitude of the data.

    This function can handle nans in the input

    Args:
        time_by_neurons_full: numpy array [time, neurons]
        t: optional, only important if time_by_neurons is not sampled evenly in time

    Returns: time_by_neurons divided by the exponential
    """

    time_by_neurons_full = check_input_format(time_by_neurons_full)
    nan_neurons = np.all(np.isnan(time_by_neurons_full), axis=0)
    time_by_neurons = time_by_neurons_full.copy()
    time_by_neurons = time_by_neurons[:, ~nan_neurons]
    num_neurons = time_by_neurons.shape[1]

    if num_neurons == 0:
        return time_by_neurons_full

    if t is None:
        t = np.arange(time_by_neurons.shape[0])

    tau_0 = t[-1, None] / np.arange(2 + num_exp, 2, -1)
    data_max = np.nanmax(time_by_neurons, axis=0)
    a_0 = np.concatenate([data_max / i for i in np.arange(2, 2 + num_exp)], axis=0)
    offset_0 = np.zeros(num_neurons)

    # fit in log space to ensure everything stays positive
    if fit_offset:
        p_0 = np.concatenate((np.log(tau_0), offset_0, np.log(a_0)), axis=0)
    else:
        p_0 = np.concatenate((np.log(tau_0), np.log(a_0)), axis=0)

    # mask out any nans
    mask = ~np.isnan(time_by_neurons)
    time_by_neurons[~mask] = 0

    if fit_offset:
        amp_ind_start = num_exp + num_neurons
    else:
        amp_ind_start = num_exp

    def get_exponential_approx(p):
        tau = np.exp(p[:num_exp])
        offset = p[num_exp:amp_ind_start]
        amp = np.exp(p[amp_ind_start:])

        exponential = np.zeros_like(time_by_neurons)

        for ex in range(num_exp):
            exponential = exponential + amp[ex] * np.exp(-t[:, None] / tau[ex])

        if fit_offset:
            exponential = exponential + offset

        return exponential

    def loss_fn(p):
        exponential_approx = get_exponential_approx(p)

        squared_error = ((exponential_approx - time_by_neurons)**2)
        # set unmeasured values to 0, so they don't show up in the sum
        squared_error = squared_error * mask
        return squared_error.sum()

    p_hat = sio.minimize(loss_fn, p_0).x
    offset = p_hat[num_exp:amp_ind_start]

    exponential_approx = get_exponential_approx(p_hat)

    if fit_offset:
        time_by_neurons_corrected = (time_by_neurons - offset) / (exponential_approx - offset)
    else:
        time_by_neurons_corrected = time_by_neurons / exponential_approx

    # put the unmeasured value nans back in
    time_by_neurons_corrected[~mask] = np.nan

    time_by_neurons_final = time_by_neurons_full.copy()
    time_by_neurons_final[:, ~nan_neurons] = time_by_neurons_corrected

    return time_by_neurons_final


def preprocess_data(emissions, inputs, start_index=0, correct_photobleach=False, filter_size=2, upsample_factor=1):
    # remove the beginning of the recording which contains artifacts and mean subtract
    emissions = emissions[start_index:, :]
    inputs = inputs[start_index:, :]

    # remove stimulation events with interpolation
    window = np.array((-2, 3))
    for c in range(emissions.shape[1]):
        stim_locations = np.where(inputs[:, c])[0]

        for s in stim_locations:
            data_x = window + s
            interp_x = np.arange(data_x[0], data_x[1])
            emissions[interp_x, c] = np.interp(interp_x, data_x, emissions[data_x, c])

    if filter_size > 0:
        # filter out noise at the nyquist frequency
        filter_shape = np.ones(filter_size) / filter_size
        emissions_filtered = np.zeros((emissions.shape[0] - filter_size + 1, emissions.shape[1]))

        for c in range(emissions.shape[1]):
            emissions_filtered[:, c] = au.nan_convolve(emissions[:, c].copy(), filter_shape)
    else:
        emissions_filtered = emissions.copy()

    if correct_photobleach:
        # photobleach correction
        emissions_filtered_corrected = np.zeros_like(emissions_filtered)
        for c in range(emissions_filtered.shape[1]):
            emissions_filtered_corrected[:, c] = photobleach_correction(emissions_filtered[:, c], num_exp=2, fit_offset=False)[:, 0]

        # occasionally the fit fails check for outputs who don't have a mean close to 1
        # fit those with a single exponential
        # all the nan mean calls throw warnings when averaging over nans so supress those
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_fits_2exp = np.where(np.abs(np.nanmean(emissions_filtered_corrected, axis=0) - 1) > 0.1)[0]

            for bf in bad_fits_2exp:
                emissions_filtered_corrected[:, bf] = photobleach_correction(emissions_filtered[:, bf], num_exp=1, fit_offset=True)[:, 0]

            bad_fits_1xp = np.where(np.abs(np.nanmean(emissions_filtered_corrected, axis=0) - 1) > 0.2)[0]
            if len(bad_fits_1xp) > 0:
                warnings.warn('Photobleach correction problems found in neurons ' + str(bad_fits_1xp) + ' setting to nan')
                emissions_filtered_corrected[:, bad_fits_1xp] = np.nan

            # divide by the mean and subtract 1. Will throw warnings on the all nan data, ignore them
            emissions_time_mean = np.nanmean(emissions_filtered_corrected, axis=0)
            emissions_filtered_corrected = emissions_filtered_corrected / emissions_time_mean - 1

        emissions_filtered_corrected[emissions_filtered_corrected > 5] = np.nan

    else:
        emissions_filtered_corrected = emissions_filtered
        emissions_mean = np.nanmean(emissions_filtered_corrected, axis=0)
        emissions_std = np.nanstd(emissions_filtered_corrected, axis=0)
        emissions_filtered_corrected = (emissions_filtered_corrected - emissions_mean) / emissions_std

    # truncate inputs to match emissions after filtering
    inputs = inputs[:emissions_filtered_corrected.shape[0], :]

    if upsample_factor != 1:
        emissions_out = np.empty((emissions_filtered_corrected.shape[0] * upsample_factor, emissions_filtered_corrected.shape[1]))
        emissions_out[:] = np.nan
        emissions_out[::upsample_factor, :] = emissions_filtered_corrected

        inputs_out = np.zeros((inputs.shape[0] * upsample_factor, inputs.shape[1]))
        inputs_out[::upsample_factor, :] = inputs
    else:
        emissions_out = emissions_filtered_corrected
        inputs_out = inputs

    return emissions_out, inputs_out


def load_data(data_path, num_data_sets=None, neuron_freq=0.0, held_out_data=[],
              hold_out='worm', upsample_factor=1):
    data_path = Path(data_path)

    emissions_train = []
    inputs_train = []
    cell_ids_train = []
    path_name = []
    sample_rate = 2 * upsample_factor  # seconds per sample DEFAULT

    # find each folde that has recording data
    for i in sorted(data_path.rglob('funcon_preprocessed_data.pkl')):
        path_name.append(i.parts[-2])

        data_file = open(i, 'rb')
        preprocessed_data = pickle.load(data_file)
        data_file.close()

        this_emissions = preprocessed_data['emissions']
        this_inputs = preprocessed_data['inputs']
        this_cell_ids = preprocessed_data['cell_ids']

        emissions_train.append(this_emissions)
        inputs_train.append(this_inputs)
        cell_ids_train.append(this_cell_ids)

    emissions_test = []
    inputs_test = []
    cell_ids_test = []

    if hold_out == 'worm':
        for i in reversed(range(len(emissions_train))):
            # skip any data that is being held out
            if path_name[i] in held_out_data:
                emissions_test.append(emissions_train.pop(i))
                inputs_test.append(inputs_train.pop(i))
                cell_ids_test.append(cell_ids_train.pop(i))

        emissions_test += emissions_train[num_data_sets:]
        inputs_test += inputs_train[num_data_sets:]
        cell_ids_test += cell_ids_train[num_data_sets:]

        emissions_test = emissions_test[:num_data_sets]
        inputs_test = inputs_test[:num_data_sets]
        cell_ids_test = cell_ids_test[:num_data_sets]

        emissions_train = emissions_train[:num_data_sets]
        inputs_train = inputs_train[:num_data_sets]
        cell_ids_train = cell_ids_train[:num_data_sets]

    elif hold_out == 'middle':
        frac = 0.3

        for i in range(num_data_sets):
            num_time = emissions_train[i].shape[0]
            test_inds = np.arange((1 - frac) * num_time / 2, (1 + frac) * num_time / 2, dtype=int)
            emissions_test.append(emissions_train[i][test_inds, :])
            inputs_test.append(inputs_train[i][test_inds, :])
            cell_ids_test.append(cell_ids_train[i])

            emissions_train[i][test_inds, :] = np.nan
            inputs_train[i][test_inds, :] = 0

        emissions_train = emissions_train[:num_data_sets]
        inputs_train = inputs_train[:num_data_sets]

    elif hold_out == 'end':
        frac = 0.3

        for i in range(num_data_sets):
            num_time = emissions_train[i].shape[0]
            start_ind = int(frac * num_time)
            emissions_test.append(emissions_train[i][start_ind:, :])
            inputs_test.append(inputs_train[i][start_ind:, :])
            cell_ids_test.append(cell_ids_train[i])

            emissions_train[i] = emissions_train[i][:start_ind, :]
            inputs_train[i] = inputs_train[i][:start_ind, :]

        emissions_train = emissions_train[:num_data_sets]
        inputs_train = inputs_train[:num_data_sets]

    else:
        raise Exception('Hold out style is not recognized')

    print('Size of data set:', len(emissions_train))

    # align the data sets so that each column corresponds to the same cell ID
    data_train = {}
    data_test = {}

    data_train['emissions'], data_train['inputs'], data_train['cell_ids'] = \
        align_data_cell_ids(emissions_train, inputs_train, cell_ids_train)

    data_test['emissions'], data_test['inputs'], data_test['cell_ids'] = \
        align_data_cell_ids(emissions_test, inputs_test, cell_ids_test, cell_ids_unique=data_train['cell_ids'])

    # eliminate neurons that don't show up often enough
    measured_neurons = np.stack([np.mean(np.isnan(i), axis=0) <= 0.5 for i in data_train['emissions']])
    measured_freq = np.mean(measured_neurons, axis=0)
    neurons_to_keep = measured_freq >= neuron_freq

    data_train['emissions'] = [i[:, neurons_to_keep] for i in data_train['emissions']]
    data_train['inputs'] = [i[:, neurons_to_keep] for i in data_train['inputs']]
    data_train['cell_ids'] = [data_train['cell_ids'][i] for i in range(len(data_train['cell_ids'])) if neurons_to_keep[i]]
    data_train['sample_rate'] = sample_rate

    data_test['emissions'] = [i[:, neurons_to_keep] for i in data_test['emissions']]
    data_test['inputs'] = [i[:, neurons_to_keep] for i in data_test['inputs']]
    data_test['cell_ids'] = [data_test['cell_ids'][i] for i in range(len(data_test['cell_ids'])) if neurons_to_keep[i]]
    data_test['sample_rate'] = sample_rate

    return data_train, data_test


def save_run(save_folder, model_trained=None, model_true=None, ep=None, **vars_to_save):
    save_folder = Path(save_folder)
    model_save_folder = save_folder / 'models'

    # save the trained model
    if model_trained is not None:
        os.makedirs(model_save_folder, exist_ok=True)

        if ep is not None:
            trained_model_save_path = model_save_folder / ('model_trained_' + str(ep) + '.pkl')
            model_trained.save(path=trained_model_save_path)

        trained_model_save_path = model_save_folder / 'model_trained.pkl'
        model_trained.save(path=trained_model_save_path)

    # save the true model, if it exists
    if model_true is not None:
        os.makedirs(model_save_folder, exist_ok=True)

        true_model_save_path = model_save_folder / 'model_true.pkl'
        model_true.save(path=true_model_save_path)

    for k, v in vars_to_save.items():
        save_path = save_folder / (k + '.pkl')

        save_file = open(save_path, 'wb')
        pickle.dump(v, save_file)
        save_file.close()


def align_data_cell_ids(emissions, inputs, cell_ids, cell_ids_unique=None):
    if cell_ids_unique is None:
        cell_ids_unique = list(np.unique(np.concatenate(cell_ids)))
        if '' in cell_ids_unique:
            cell_ids_unique.pop(cell_ids_unique.index(''))

    num_neurons = len(cell_ids_unique)

    emissions_aligned = []
    inputs_aligned = []

    # now update the neural data and fill in nans where we don't have a recording from a neuron
    for e, i, c in zip(emissions, inputs, cell_ids):
        this_emissions = np.empty((e.shape[0], num_neurons))
        this_emissions[:] = np.nan
        this_inputs = np.zeros((e.shape[0], num_neurons))

        # loop through all the labels from this data set
        for unique_cell_index, cell_name in enumerate(cell_ids_unique):
            # find the index of the full list of cell ids
            if cell_name in c and cell_name != '':
                unaligned_cell_index = c.index(cell_name)
                this_emissions[:, unique_cell_index] = e[:, unaligned_cell_index]
                this_inputs[:, unique_cell_index] = i[:, unaligned_cell_index]

        emissions_aligned.append(this_emissions)
        inputs_aligned.append(this_inputs)

    return emissions_aligned, inputs_aligned, cell_ids_unique




