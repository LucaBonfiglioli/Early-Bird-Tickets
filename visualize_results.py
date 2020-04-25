import matplotlib.pyplot as plt
import results_manager as resm
import numpy as np


def plot_with_intervals(ax: plt.Axes, x_vals, y_vals, band_fn='minmax', const=False, pos=1, lfactor=50, alpha=1, decay=5, label=None, **kwargs):
    ax.plot(x_vals, np.mean(y_vals, 0), label=label, alpha=alpha, **kwargs)
    if decay > 0:
        alpha /= decay
    else:
        alpha = 0
    label = None

    def minmax():
        return np.min(y_vals, 0), np.max(y_vals, 0)

    def std():
        mean = np.mean(y_vals, 0)
        _std = np.std(y_vals, 0)
        return mean - _std, mean + _std

    if band_fn is not None:
        y1, y2 = locals()[band_fn]()
        if not const:
            ax.fill_between(x_vals, y1, y2, alpha=alpha, label=label, **kwargs)
        else:
            ax.plot((x_vals[pos], x_vals[pos]), (y1, y2), alpha=alpha, linestyle='-', linewidth=1, color='k')
            l = (x_vals[-1] - x_vals[0]) / lfactor
            ax.plot((x_vals[pos] - l, x_vals[pos] + l), (y1, y1), alpha=alpha, linestyle='-', linewidth=1, color='k')
            ax.plot((x_vals[pos] - l, x_vals[pos] + l), (y2, y2), alpha=alpha, linestyle='-', linewidth=1, color='k')


def merge_runs(inputs, output):
    data = []
    out = {}
    for file in inputs:
        data.append(resm.load_json(file))
    for d in data:
        for key in d.keys():
            if key not in out.keys():
                out[key] = []
            out[key] += d[key]
    resm.store_json(output, out)


def visualize(filenames, labels, colors, snap_list, pr_list, name='eb_results'):
    fig: plt.Figure = plt.figure(name)
    ax = []
    for j in range(len(pr_list)):
        ax.append(fig.add_subplot(len(pr_list), 1, j+1))
        ax[-1].set_title('Pruning rate '+str(pr_list[j]))
    for i in range(len(filenames)):
        data = resm.load_json(filenames[i])
        pruned_test_acc = np.array(data['pruned_test_accuracy'])
        for j in range(len(pruned_test_acc[0])):
            plot_with_intervals(ax[j], snap_list, pruned_test_acc[:, j, :], color=colors[i], label=labels[i])
            ax[j].grid(True)
            ax[j].set_axisbelow(True)
            ax[j].legend()
            ax[j].set_ylabel('Test Accuracy')
    ax[-1].set_xlabel('Epochs')
    fig.tight_layout()
    plt.show()


# base_names = './results/vgg16-cifar100_lfg_%d.json'
# out = './results/vgg16-cifar100_lfg.json'
# files = [base_names % (i+1) for i in range(5)]
# merge_runs(files, out)
#
filenames = ['./results/vgg16-cifar100_lf.json', './results/vgg16-cifar100_lfg.json']
labels = ['lf', 'lfg']
colors = plt.cm.tab10.colors
snap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
pr_list = [30, 50, 70]
visualize(filenames, labels, colors, snap_list, pr_list)
