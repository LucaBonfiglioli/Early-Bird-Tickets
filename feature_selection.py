def print_features(features):
    body = ' ' * features[0]
    for i in range(1, len(features)):
        body += 'O' + ' ' * (features[i] - features[i - 1] - 1)
    body += 'O'
    print(('%4d|' + body) % features[-1])


def fifo(num_features, index):
    def add_feature_fifo(index_):
        if len(features) >= num_features:
            features.pop(0)
        features.append(index_)
    features = []
    for f in range(index + 1):
        add_feature_fifo(f)
    return features


def odd(num_features, index):
    def add_feature_delete_odd(index_):
        if len(features) >= num_features:
            features.pop((index_ - 1) % (num_features - 1) + 1)
        features.append(index_)
    features = []
    for f in range(index + 1):
        add_feature_delete_odd(f)
    return features


def balanced(num_features, index):
    a = 0
    b = 0
    c = 1

    def add_feature_balanced(index_):
        nonlocal a, b, c
        if len(features) >= num_features:
            if a == 0:
                features.pop(b + 1)
            else:
                features.pop(num_features - 1)
            a += 1
            if a >= c:
                a = 0
                b += 1
            if b >= num_features - 1:
                a = 0
                b = 0
                c *= 2
        features.append(index_)
    features = []
    for f in range(index + 1):
        add_feature_balanced(f)
    return features
