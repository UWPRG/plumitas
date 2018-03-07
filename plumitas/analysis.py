import numpy as np


def get_frame_weights(df, temp, static_bias=None):
    if not static_bias:
        raise ValueError('You must supply a static bias column with the'
                         '"static_bias" arg to get frame weights.')

    # calculate normalized weight
    k = 8.314e-3
    beta = 1 / (temp * k)
    w = np.exp(beta * df.loc[:, static_bias])
    df['weight'] = w / w.sum()
    return
