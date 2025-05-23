from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from pathlib import Path


def create_sequence_ds(
    x: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    window_size: int = 200,
    window_offset: int = 0,
):
    """
    Create a sequence dataset by reshaping the input data.

    Args:
        x (np.ndarray): The input data array of shape (num_samples, num_features).
        y (np.ndarray): The target data array of shape (num_samples,).
        seq_len (int): The length of each sequence.
        window_size (int, optional): The size of the window for each sequence. Defaults to 200.
        window_offset (int, optional): The offset of the window for each sequence. Defaults to 0.

    Returns:
        np.ndarray: The reshaped input data array of shape (num_samples - seq_len, window_size * seq_len, num_features).
        np.ndarray: The target data array of shape (num_samples - seq_len,).
    """
    new_x = np.zeros((x.shape[0] - seq_len, window_size * seq_len, x.shape[2]))
    new_y = np.zeros((y.shape[0] - seq_len))
    for i in range(x.shape[0] - seq_len):
        x_t = x[i : i + seq_len]
        x_t = x_t[:, window_offset : window_offset + window_size, :]
        new_x[i] = x_t.reshape(-1, 2)
        new_y[i] = y[i + seq_len]

    return new_x, new_y


class MyScaler:

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, x):
        s_0, s_1, s_2 = x.shape
        self.scaler.fit(x.reshape(-1, s_2))

    def transform(self, x):
        s_0, s_1, s_2 = x.shape
        x = self.scaler.transform(x.reshape(-1, s_2))
        return x.reshape(s_0, s_1, s_2)

    def inverse_transform(self, x):
        s_0, s_1, s_2 = x.shape
        x = self.scaler.inverse_transform(x.reshape(-1, s_2))


def premute_windows(num_samples: int, n: int) -> np.ndarray:
    """
    Shuffles the indices of a 1D array of samples in a way that preserves the order of the samples within each window of size n.

    Args:
        num_samples: The number of samples in the dataset.
        n: The size of each window.

    Returns:
        A numpy array of shuffled indices
    """
    # Check divisibility
    if num_samples % n != 0:
        rest = num_samples % n
        num_samples = num_samples - rest

    # Generate indices for each window
    num_windows = num_samples // n
    windows = np.arange(num_samples).reshape(num_windows, n)

    # Shuffle the windows
    shuffled_windows = np.random.permutation(windows)

    # Flatten back to 1D array
    shuffled_indices = shuffled_windows.flatten()

    if num_samples % n != 0:
        shuffled_indices = np.append(
            shuffled_indices, np.arange(num_samples, num_samples + rest)
        )
    return shuffled_indices



def save_ds_to_disk(ds, path: Path, filename: Path):
    path.mkdir(parents=True, exist_ok=True)
    file_path = path.joinpath(filename)
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump(ds, f)


def save_all_ds_ids(
    recon_train_ds,
    recon_val_ds,
    recon_test_ds,
    class_train_ds,
    class_val_ds,
    class_test_ds,
    path: Path,
):
    save_ds_to_disk(recon_train_ds, path, "recon_train_ds")
    save_ds_to_disk(recon_val_ds, path, "recon_val_ds")
    save_ds_to_disk(recon_test_ds, path, "recon_test_ds")
    save_ds_to_disk(class_train_ds, path, "class_train_ds")
    save_ds_to_disk(class_val_ds, path, "class_val_ds")
    save_ds_to_disk(class_test_ds, path, "class_test_ds")


def load_all_ds_ids(path: Path):
    recon_train_ds = pickle.load(open(f"{path}/recon_train_ds.pkl", "rb"))
    recon_val_ds = pickle.load(open(f"{path}/recon_val_ds.pkl", "rb"))
    recon_test_ds = pickle.load(open(f"{path}/recon_test_ds.pkl", "rb"))
    class_train_ds = pickle.load(open(f"{path}/class_train_ds.pkl", "rb"))
    class_val_ds = pickle.load(open(f"{path}/class_val_ds.pkl", "rb"))
    class_test_ds = pickle.load(open(f"{path}/class_test_ds.pkl", "rb"))
    return (
        recon_train_ds,
        recon_val_ds,
        recon_test_ds,
        class_train_ds,
        class_val_ds,
        class_test_ds,
    )
