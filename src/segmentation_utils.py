import sys
import h5py
import logging
import warnings
from pathlib import Path

import torch
import numpy as np
from scipy.ndimage import zoom

from unet3d import Basic3DUNet

warnings.filterwarnings(action='ignore', category=UserWarning)
log = logging.getLogger(__name__)
PAD = 64


def segment_mitochondria(input_path, output_path, copy, bin2):
    dataset = load_tomogram(input_path)
    dataset, shape = preprocess_data(dataset, bin2)
    model = load_model()
    
    result = predict(model, dataset)
    save_tomogram(result, output_path, shape, dataset, copy)


def load_tomogram(input_path):
    # callback function to walk through the HDF file to find the data
    def find_dataset(name, obj):
        return obj if isinstance(obj, h5py.Dataset) else None

    with h5py.File(input_path) as fh:
        dataset = fh.visititems(find_dataset)

        if dataset is None:
            log.error(f"Unable to find an HDF5 dataset in {input_path}")
            sys.exit(1)

        else:
            log.info(f"Reading tomogram of shape {dataset.shape} from {dataset.name}")
            return np.squeeze(dataset)


def preprocess_data(dataset, bin2):
    # assuming that the data follows a standard normal distribution, it is clipped and scaled
    dataset = np.clip(dataset, -3, 3) / 3.0

    if bin2:  # downsample by a factor of 2
        old_shape = dataset.shape
        dataset = zoom(dataset, 0.5, order=1)
        log.info(f"Tomogram binned by 2 from {old_shape} -> {dataset.shape}")

    # compute size of the padded data
    d, h, w = dataset.shape
    x = int(PAD * np.ceil(d / PAD))
    y = int(PAD * np.ceil(h / PAD))
    z = int(PAD * np.ceil(w / PAD))

    if (x, y, z) != (d, h, w):  # check if padding is required
        new_dataset = np.zeros((x, y, z), dtype=np.float32)
        x = (x - d) // 2
        y = (y - h) // 2
        z = (z - w) // 2

        # apply zero padding
        new_dataset[x: x + d, y: y + h, z: z + w] = dataset
        dataset = new_dataset
        log.info(f"Tomogram padded from {d, h, w} -> {dataset.shape}")

    return dataset, (d, h, w)  # the original shape


def load_model():
    # the model is bundled along with the code, so it should always be here
    model_path = Path(__file__).parents[1] / "models" / "mito_weights.pt"
    
    if not model_path.exists():
        log.error(f"Unable to locate model file at {model_path}")

    model = Basic3DUNet()
    model.load_state_dict(torch.load(model_path))
    log.info(f"Model loaded from checkpoint {model_path}")

    model.cuda()
    model.eval()
    return model


def predict(model, dataset):
    # reshape the dataset to compatible shape and convert it to a tensor
    dataset = np.expand_dims(dataset, axis=(0, 1))
    input_tensor = torch.from_numpy(dataset).cuda()

    with torch.no_grad():
        log.info("Predicting mitochondria probabilities in the tomogram")
        result = model(input_tensor).cpu().numpy().astype(np.float32)

    return np.squeeze(result)  # remove the extra dimensions from the result


def save_tomogram(result, output_path, shape, dataset, copy):
    d, h, w = shape  # the original shape
    x, y, z = result.shape

    # check if the data was padded
    if (x, y, z) != (d, h, w):
        x = (x - d) // 2
        y = (y - h) // 2
        z = (z - w) // 2

        # undo the padding for the data and the predictions
        result = result[x: x + d, y: y + h, z: z + w]
        dataset = dataset[x: x + d, y: y + h, z: z + w]

    with h5py.File(output_path, 'w') as fh:
        log.info(f"Saving mitochondria predictions to {output_path}:/mito_pred")
        fh.create_dataset("mito_pred", data=result)

        if copy:
            log.info(f"Copying original tomogram to {output_path}:/data")
            fh.create_dataset("data", data=dataset)
