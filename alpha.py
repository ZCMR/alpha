def sigmoid_alpha(x, k,dm):
    betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
    # dm = max(np.max(x), 0.0001)
    res = (2 / (1 + np.exp(-x*k/dm)) - 1)*betak
    return np.maximum(0, res)