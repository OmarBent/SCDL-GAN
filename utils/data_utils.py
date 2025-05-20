import os
import numpy as np


def loadfeatures(data_dir):
    """
    Load all sequence feature files from the given directory.

    Args:
        data_dir (str): Path to the directory containing sparse code files.

    Returns:
        list[np.ndarray]: A list of numpy arrays, each representing a sequence's sparse codes.
    """
    data = []
    file_list = sorted(os.listdir(data_dir))  # Sort to ensure consistent order
    for file_name in file_list:
        if file_name.endswith(".txt") and file_name.startswith("sequence_"):
            file_path = os.path.join(data_dir, file_name)
            sequence = np.loadtxt(file_path)
            data.append(sequence)
    return data


def tsne_data_format_exp(codes, labels, opt):
    """
    Format data for t-SNE visualization. It duplicates the data and adds label annotations.

    Args:
        codes (list[np.ndarray]): List of 3D arrays representing sequence codes.
        labels (list[int]): Ground truth labels for the sequences.
        opt (Namespace): Configuration object containing `n_classes`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Formatted input `x` and label vector `y` for t-SNE.
    """
    # Flatten each code matrix and prepare data + label tensors
    dim = np.prod(np.asarray(codes[0]).shape)
    x = np.zeros((len(codes) * 2, dim + 1))
    y = np.zeros((len(codes) * 2))

    for i, code in enumerate(codes):
        flat_code = code.reshape(1, dim)
        x[i, :dim] = flat_code
        x[i, dim] = 0  # Fake or placeholder label
        y[i] = 0       # Label for original

        x[i + len(codes), :dim] = flat_code
        x[i + len(codes), dim] = labels[i]  # Real label
        y[i + len(codes)] = 1               # Label for real

    return x, y
