import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import results_manager as utils
import feature_selection as fsel

# Hyperparameters for gbr on mnist:
# n_estimators = 190, learning_rate = 0.046, max_depth = 5, subsample = 0.75
# Hyperparameters for gbr on cifar10:
# n_estimators = 400, learning_rate = 0.022, max_depth = 8, subsample = 0.55


def merge_datasets(input_files, output_file):
    d = [utils.load_binary(f) for f in input_files]
    out = ([], [])
    for l in range(len(d[0][0])):
        out[0].append(torch.cat([d[i][0][l] for i in range(len(d))]))
        out[1].append(torch.cat([d[i][1][l] for i in range(len(d))]))
    utils.store_binary(output_file, out)


def select_features(dataset, features, layer, downsample=1):
    x = dataset[0][layer][:, features]
    y = dataset[1][layer]
    examples = y.nelement()
    perm = torch.randperm(examples)[:int(examples / downsample)]
    x = x[perm, :]
    y = y[perm]
    return x, y


def stack_layers(dataset, iterations, downsample_rates=None):
    layers = len(dataset[1])
    x_list = []
    y_list = []
    for layer in range(layers):
        if downsample_rates is None:
            downsample = 1
        else:
            downsample = downsample_rates[layer]
        d = select_features(dataset, iterations, layer, downsample)
        x_list.append(d[0])
        y_list.append(d[1])
    x = torch.cat(x_list, 0)
    y = torch.cat(y_list, 0)
    return x, y


def eval_regressor_fixed(regressor, dataset, features):
    layers = len(dataset[1])
    spearmanrho = torch.zeros(layers)
    for layer in range(layers):
        x, y = select_features(dataset, features, layer)
        pred = x[:, -1]
        if regressor is not None:
            pred = torch.tensor(regressor.predict(x))
        pred_scores = torch.abs(pred)
        truth_scores = torch.abs(y)
        spearmanrho[layer] = utils.efficient_spearmanr(truth_scores.cpu(), pred_scores.cpu())
    return spearmanrho


def eval_regressor(regressors, tr_dataset, test_dataset, num_features, layers, feature_interval,
                   runs=10, name='unnamed', reg_path='results/regressors/reg_%d', downsample_rates=None):
    tot_features = tr_dataset[0][0].size()[1]
    spearmanrho = torch.zeros((tot_features, layers, runs))
    for f in range(tot_features):
        features = fsel.balanced(num_features, f)
        fsel.print_features(features)
        if type(regressors) is not list:
            regressor = regressors
            x, y = stack_layers(tr_dataset, features, downsample_rates)
            regressor.fit(x, y)
            utils.store_binary(reg_path % f, regressor)
        else:
            regressor = regressors[f]
        if type(regressor) == HandcraftedRegressor:
            regressor.tr_iterations = (np.array(features) * feature_interval).tolist()
        for run in range(runs):
            test_set = utils.load_binary(test_dataset % run)
            spearmanrho[f, :, run] = eval_regressor_fixed(regressor, test_set, features)
            print(spearmanrho[f, :, run])
        print(('[Snapshot %d] MEAN: ' % f) + str(spearmanrho[f, :, :].mean(dim=1)))
    utils.store_json('results/spearman_correlation_experiment/' + name, spearmanrho.tolist())
    return spearmanrho


class HandcraftedRegressor:
    def __init__(self, score_fn, tr_iterations):
        self.score_fn = score_fn
        self.tr_iterations = tr_iterations

    def predict(self, x):
        flat_history = []
        for i in range(x.size()[1]):
            flat_history.append(x[:, i])
        return self.score_fn(flat_history, self.tr_iterations)

    def fit(self, x, y):
        return


# Dataset building
# for i in range(10):
#     print(i)
#     build_dataset(Conv4Cifar10, 100, 67, 10, 'conv4_cifar10_100x67_tr_%d' % i)
# for i in range(10):
#     print(i + 10)
#     build_dataset(Conv4Cifar10, 100, 67, 10, 'conv4_cifar10_100x67_ev_%d' % i)
# for i in range(10):
#     print(i + 20)
#     build_dataset(Conv4Cifar10, 100, 67, 10, 'conv4_cifar10_100x67_te_%d' % i)

# Regressor evaluation
# reg = GradientBoostingRegressor(n_estimators=400, learning_rate=0.022, max_depth=8, subsample=0.55)
# reg = [utils.load_binary('results/regressors/conv4_cifar10_bal/reg_%d' % i) for i in range(100)]
# results = eval_regressor(reg, utils.load_binary('data/trajectories/fc_mnist_100x80_tr_0'),
#                          'data/trajectories/fc_mnist_100x80_te_%d', num_features=10, layers=6, feature_interval=80,
#                          runs=10, name='fc_mnist_gbr_tuned_bal_cross', reg_path='results/regressors/fc_mnist_bal/reg_%d',
#                          downsample_rates=None)

# Hyperparameter grid search
param_grid_ = {'n_estimators': [100, 200, 300, 400, 500],
               'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
               'max_depth': [3, 4, 5, 6],
               'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}
tr_set_ = utils.load_binary('results/datasets/vgg16-cifar100_tr')
features_ = [0, 16, 24, 32, 40, 48, 56, 64, 72, 80]
x_, y_ = stack_layers(tr_set_, features_)
cv = GridSearchCV(GradientBoostingRegressor(), param_grid_, n_jobs=-1, verbose=2)
cv.fit(np.array(x_), np.array(y_))

# Training set downsampling
# tr_set = utils.load_binary('data/trajectories/conv4_cifar10_100x67_tr_0')
# ev_set = utils.load_binary('data/trajectories/conv4_cifar10_100x67_ev_0')
# features_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# x_, y_ = stack_layers(tr_set, features_)
# x_d_, y_d_ = stack_layers(tr_set, features_, downsample_rates=[1, 1, 1, 1, 1, 1, 1.5, 1, 20, 1, 1, 1, 1, 1])
# print(y_.nelement())
# print(y_d_.nelement())
# # print('training full')
# # reg = GradientBoostingRegressor()
# # reg.fit(x, y)
# reg = utils.load_binary('results/regressors/temp_full')
# print('training downsampled')
# reg_d = GradientBoostingRegressor()
# reg_d.fit(x_d_, y_d_)
# print(eval_regressor_fixed(reg_d, ev_set, features_) - eval_regressor_fixed(reg, ev_set, features_))
