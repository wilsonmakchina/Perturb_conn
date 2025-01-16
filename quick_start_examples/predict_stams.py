import pickle
import lgssm_utilities as ssmu
import metrics as met
import numpy as np
from matplotlib import pyplot as plt

# This file will load in the measured STAMs and matrix of correlation coefficients
# and compare these measured values with the predictions of the three models
# highlighted in Fig 3

model_names = ['connectome_constrained', 'unconstrained', 'shuffled_constrained']
# time in seconds to calculate a stimulus triggered average [before, after]
window = [15, 30]
plot_color = {'connectome_constrained': np.array([27, 158, 119]) / 255,
              'unconstrained': np.array([117, 112, 179]) / 255,
              'shuffled_constrained': np.array([128, 128, 128]) / 255,
              }

# load in the trained models from the paper
models = {}
for mn in model_names:
    model_file = open('models/' + mn + '.pkl', 'rb')
    models[mn] = pickle.load(model_file)
    model_file.close()

# load in the measured STAMs and correlation coefficents between each pair of neurons
# these were calculated from 30 held-out recordings from the Randi et al 2023 data set

measured_metric = {}
stams_file = open('data/measured_stams.pkl', 'rb')
measured_metric['stams'] = pickle.load(stams_file)
stams_file.close()

corr_file = open('data/measured_corr.pkl', 'rb')
measured_metric['corr'] = pickle.load(corr_file)
corr_file.close()

# mask out the diagonals to only evaluate on neuron-to-neuron interactions
num_neurons = measured_metric['stams']['train'].shape[0]
diagonal_mask = np.eye(num_neurons, dtype=bool)
for mm in measured_metric:
    for type in ['train', 'test']:
        measured_metric[mm][type][diagonal_mask] = np.nan

# calculate the STAMS and correlation matrix from the models
model_pred = {}
for m in models:
    model_pred[m] = {'stams': ssmu.calculate_stams(models[m], window=window),
                     'corr': ssmu.predict_model_corr_coef(models[m])}
    model_pred[m]['stams'][diagonal_mask] = np.nan
    model_pred[m]['corr'][diagonal_mask] = np.nan

# calculate the train-test correlation for the stams and the matrix of correlation coefficents
train_test = {'stams': met.nan_corr(measured_metric['stams']['train'], measured_metric['stams']['test'])[0],
              'corr': met.nan_corr(measured_metric['corr']['train'], measured_metric['corr']['test'])[0]}

# calculate the correlation between model predictions and measurement
model_score = {'stams': [],
               'corr': []}
model_score_ci = {'stams': [],
                  'corr': []}

for m in models:
    for type in ['stams', 'corr']:
        score, score_ci = met.nan_corr(model_pred[m][type], measured_metric[type]['test'])
        model_score[type].append(score)
        model_score_ci[type].append(score_ci)

# plot the correlation of the model predictions to the actual measurements
y_limits = [0, 1.1]

for type in ['stams', 'corr']:
    plt.figure()
    y_val = np.array(model_score[type])
    y_val_ci = np.stack(model_score_ci[type]).T
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color[i] for i in model_names]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.axhline(train_test[type], linestyle='--', color='k')
    plt.xticks(plot_x, labels=model_names, rotation=45)
    plt.ylabel('correlation')
    plt.title(type)
    plt.ylim(y_limits)
    plt.tight_layout()

    plt.figure()
    y_val = np.array(model_score[type]) / train_test[type]
    y_val_ci = np.stack(model_score_ci[type]).T / train_test[type]
    plot_x = np.arange(y_val.shape[0])
    bar_colors = [plot_color[i] for i in model_names]
    plt.bar(plot_x, y_val, color=bar_colors)
    plt.errorbar(plot_x, y_val, y_val_ci, fmt='none', color='k')
    plt.axhline(1, linestyle='--', color='k')
    plt.xticks(plot_x, labels=model_names, rotation=45)
    plt.ylabel('relative correlation')
    plt.title(type)
    plt.ylim(y_limits)
    plt.tight_layout()

plt.show()
a=1