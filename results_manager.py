import json
import os


def store_json(filename, obj):
    with open(filename, 'w') as handle:
        json.dump(obj, handle, indent=4)


def load_json(filename):
    with open(filename, 'r') as handle:
        s = json.load(handle)
    return s


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