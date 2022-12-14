import math
import numpy as np


def truncate(record, limit=128):
    """
    Safely truncate a tomogram in the depth dimension
    Number of slices to keep are defined by limit
    TODO: Fix this implementation
    """

    depth = record["shape"][0]
    center_slice = record["slice_2"]  # 5 slices
    mid = depth // 2
    d = limit // 2

    if depth <= limit:  # nothing to do
        return record

    # if there is enough data around the central slices
    if center_slice - d >= 0 and center_slice + d <= depth:
        lo, hi = center_slice - d, center_slice + d

    # Otherwise, one of the limits is either 0 or depth
    elif center_slice >= mid:
        lo, hi = depth - limit, depth

    else:
        lo, hi = 0, limit

    # crop all the volumes to the new dimension
    for key in record["data_keys"]:
        record[key] = record[key][lo:hi]

    # slices may have been translated by the truncation
    for i in range(5):
        record[f"slice_{i}"] -= lo

    record["shape"] = (limit, *record["shape"][1:])
    return record


def pad(x, r=32):
    """
    Pad all dimensions to be multiple of r
    Choose r based on UNet architecture requirements
    """

    d, h, w = x.shape

    d_new = math.ceil(d / r) * r
    h_new = math.ceil(h / r) * r
    w_new = math.ceil(w / r) * r

    if (d_new, h_new, w_new) == x.shape:
        return x

    x_new = np.zeros((d_new, h_new, w_new), dtype=x.dtype)
    x_new[:d, :h, :w] = x
    return x_new


def pad_and_expand(X, Y, W, record):
    X = np.expand_dims(pad(X), 0)
    Y = {key: np.expand_dims(pad(y), 0) for key, y in Y.items()}
    W = {key: pad(w) for key, w in W.items() if w is not None}

    return dict(X=X, Y=Y, W=W, tomo_name=record["tomo_name"])


def weight(target, record, slice_ids, include_zlimits):
    data = record[target]
    slices = [record[f"slice_{i}"] for i in range(5)]

    if include_zlimits:
        weight = np.where(data == -1, 0, 1).astype(np.float32)
        exclude_ids = list(set(range(5)) - slice_ids)
        weight[[slices[i] for i in exclude_ids]] = 0

        print(exclude_ids)

    else:
        weight = np.zeros_like(data)
        weight[[slices[i] for i in slice_ids]] = 1

    return weight.astype(bool)


if __name__ == '__main__':
    pass
