import json
import os
import pickle


def store_json(filename, obj):
    with open(filename, 'w') as handle:
        json.dump(obj, handle, indent=4)


def load_json(filename):
    with open(filename, 'r') as handle:
        s = json.load(handle)
    return s


def load_binary(filename):
    with open(filename, 'rb') as handle:
        w = pickle.load(handle)
    return w


def store_binary(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=0)


def init(name, force=False):
    if not os.path.isfile(name) or force:
        data = {
            # [runs]
            'unpruned_test_accuracy': [],
            # [runs, pr, epoch]
            'pruned_test_accuracy': [],
            # [runs, pr]
            'eb_test_accuracy': [],
            'eb_epoch': []
        }
        store_json(name, data)