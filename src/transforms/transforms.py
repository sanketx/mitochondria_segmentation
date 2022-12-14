import cv2
import h5py
import numpy as np
import albumentations as A

cv2.setNumThreads(1)

transform = A.ReplayCompose([
    A.GridDistortion(
        num_steps=10,
        distort_limit=0.25,
        interpolation=cv2.INTER_LANCZOS4,
        p=1
    ),
    A.RandomToneCurve(p=1, scale=0.25),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=1),
    A.RandomRotate90(p=1),
    A.SafeRotate(limit=30, interpolation=cv2.INTER_LANCZOS4, p=1)
])


def transform_volumes(X, Y):
    # scale from (-1.0, 1.0) to (0, 255)
    X = (255 * (X + 1) / 2).astype(np.uint8)
    dummy_image = X[0]
    
    transformation = transform(image=dummy_image)
    params = transformation["replay"]

    aug_image = transformation["image"]
    new_shape = (X.shape[0], *aug_image.shape)

    aug_X = np.zeros(new_shape)
    aug_Y = {key: np.zeros(new_shape) for key, y in Y.items()}

    for i, image in enumerate(X):
        aug_X[i] = A.ReplayCompose.replay(params, image=image)["image"]

    # reverse the scaling
    aug_X = (2 * aug_X.astype(np.float32) / 255) - 1

    for y, ay in zip(Y.values(), aug_Y.values()):
        result = A.ReplayCompose.replay(params, image=dummy_image, masks=list(y))
        
        for i, mask in enumerate(result["masks"]):
            ay[i] = mask

    return aug_X, aug_Y


if __name__ == '__main__':
    with h5py.File("/sdf/home/s/sanketg/projects/mito_methods/src/transforms/tests/sample.hdf") as fh:
        data = fh["data"][()]
        mito = fh["mito"][()]

    X, Y = transform_volumes(data, {"mito": mito})

    with h5py.File("aug_sample.hdf", 'w') as fh:
        fh.create_dataset("data", data=X)
        fh.create_dataset("mito", data=Y["mito"], compression="gzip")
