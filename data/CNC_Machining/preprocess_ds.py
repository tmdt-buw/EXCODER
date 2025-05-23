import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm


def mvg_avg(x: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Calculates the moving average of a multi-dimensional NumPy array.

    Args:
        x (np.ndarray): The input NumPy array with shape [seq_len, n_dim].
        window_size (int): The size of the window.
        step_size (int): The step size.

    Returns:
        np.ndarray: The moving average array with shape [new_seq_len, n_dim],
                   where new_seq_len = seq_len // step_size.
    """
    if window_size < 1 or step_size < 1:
        raise ValueError("window_size and step_size should be greater than 0")

    if len(x.shape) != 2:
        raise ValueError(f"Expected input shape [seq_len, n_dim], got {x.shape}")

    # Create a window of uniform weights and normalize it
    window = np.ones(window_size) / window_size

    # Initialize output array
    mvg_avg = np.zeros_like(x)

    # Calculate moving average for each dimension
    n_dim = x.shape[1]
    for dim in range(n_dim):
        # Use the numpy convolve function to calculate the moving average
        mvg_avg[:, dim] = np.convolve(x[:, dim], window, mode="same")

    # Select the values according to the step_size
    mvg_avg = mvg_avg[::step_size]

    return mvg_avg


def save_as_np_files(data_path: Path, verbose: bool = True):
    """
    load data (good and bad) from the research data storages

    Keyword Arguments:
            data_path {str} -- [path to the directory]
            verbose {bool}

        Returns:
            datalist --  [list of the the X samples]
            label --  [list of the the y labels ]
    """

    # list all .h5 files
    list_paths = list(data_path.glob("*/*/*/*.h5"))
    list_paths.sort()
    if not list_paths and verbose:
        print(f"skipping {data_path} empty directory...")

    # read and append the samples with the corresponding labels
    if verbose:
        print(f"loading files from {data_path}... ")
    for element in tqdm(list_paths):
        # check if additional label needed ("Mxx_Aug20xx_Tool,nrX")
        label = element.parent.name

        with h5py.File(element, "r") as f:
            vibration_data = f["vibration_data"][:]
            # vibration_data = mvg_avg(vibration_data, window_size=300, step_size=150)
            vibration_data = vibration_data.astype(np.float32)
            new_file_path = element
            new_file_path = new_file_path.parent / (
                new_file_path.stem + "_" + label + ".npy"
            )
            # save as npy file
            np.save(new_file_path, vibration_data)


def load_np_files(data_path: Path, verbose: bool = True):
    """
    Load and process numpy files containing CNC machining data.

    Args:
        data_path (Path): Path to directory containing .npy files
        verbose (bool): Whether to print progress information

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - Processed data array of shape (n_samples, max_seq_len, 3)
            - Labels array of shape (n_samples,) with 1 for 'good' and 0 for 'bad'
    """
    list_paths = list(data_path.glob("*/*/*/*.npy"))
    list_paths.sort()
    if not list_paths and verbose:
        print(f"skipping {data_path} empty directory...")

    data_lists = []
    data_labels = []
    for element in tqdm(list_paths):
        data_lists.append(np.load(element))
        label = element.parent.name
        label_int = 1 if label == "good" else 0
        data_labels.append(label_int)

    max_seq_len = 0
    for x in data_lists:
        if x.shape[0] > max_seq_len:
            max_seq_len = x.shape[0]

    data = np.zeros((len(data_lists), max_seq_len, 3))
    for i, val in enumerate(data_lists):
        data[i, : val.shape[0], :] = val

    data_labels = np.array(data_labels)
    return data, data_labels


def mvg_avg_batch(x: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Calculates the moving average of a multi-dimensional NumPy array.

    Args:
        x (np.ndarray): The input NumPy array with shape [batch, seq_len, n_dim].
        window_size (int): The size of the window.
        step_size (int): The step size.

    Returns:
        np.ndarray: The moving average array with shape [batch, new_seq_len, n_dim],
                    where new_seq_len = seq_len // step_size.
    """
    batch_size, seq_len, n_dim = x.shape

    # Calculate the new sequence length
    new_seq_len = seq_len // step_size

    # Create an array to store the moving averages
    mvg_avg_arr = np.zeros((batch_size, new_seq_len, n_dim))

    for i in range(new_seq_len):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        mvg_avg_arr[:, i, :] = np.mean(x[:, start_idx:end_idx, :], axis=1)

    return mvg_avg_arr


def cut_away_zeros_at_end(x: np.ndarray) -> np.ndarray:
    """
    Remove trailing zero rows from a 2D array.

    Args:
        x (np.ndarray): Input array of shape (seq_len, features)

    Returns:
        np.ndarray: Array with trailing zero rows removed
    """
    return x[~np.all(x == 0, axis=1)]


def scale_data_length(x: np.ndarray, length: int) -> np.ndarray:
    """
    Scale time series data to a specified length using linear interpolation.

    Args:
        x (np.ndarray): Input array of shape (seq_len, features)
        length (int): Target sequence length

    Returns:
        np.ndarray: Interpolated array of shape (length, features)
    """
    x = cut_away_zeros_at_end(x)
    original_length = x.shape[0]
    feature_dim = x.shape[1]

    linspace_original = np.linspace(0, original_length - 1, original_length)
    linspace_new = np.linspace(0, original_length - 1, length)
    # interpolate to length
    x_interp = np.zeros((length, feature_dim))
    for i in range(feature_dim):
        x_interp[:, i] = np.interp(linspace_new, linspace_original, x[:, i])
    return x_interp

def preprocess_ds(
    data_arrays: np.ndarray,
    data_labels: np.ndarray,
    mvg_window_size: int = 300,
    mvg_step_size: int = 150,
    reduce_length: int = 2000,
    n_split_windows: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess CNC machining data by applying moving average and sequence length normalization.

    Args:
        data_arrays (np.ndarray): Input array of shape (n_samples, seq_len, 3) containing
            vibration data in x, y, z dimensions.
        data_labels (np.ndarray): Labels array of shape (n_samples,) with 1 for 'good'
            and 0 for 'bad' samples.
        mvg_window_size (int, optional): Window size for moving average. Defaults to 300.
        mvg_step_size (int, optional): Step size for moving average. Defaults to 150.
        reduce_length (int, optional): Target sequence length after interpolation. Defaults to 2000.
        n_split_windows (int, optional): Number of windows to split each sequence into. Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - Processed data array of shape (n_samples * n_split_windows, reduce_length // n_split_windows, 3)
            - Labels array of shape (n_samples * n_split_windows,) with repeated labels for each window

    Processing steps:
        1. Apply moving average to smooth the signal
        2. Scale each sequence to a fixed length using linear interpolation
        3. Split each sequence into multiple windows
        4. Repeat labels for each window
    """
    mvg_data = mvg_avg_batch(
        data_arrays, window_size=mvg_window_size, step_size=mvg_step_size
    )
    mvg_data_scaled = np.zeros((mvg_data.shape[0], reduce_length, 3))
    for i in range(mvg_data.shape[0]):
        mvg_data_scaled[i] = scale_data_length(mvg_data[i], reduce_length)

    # split into 200 windows
    mvg_data_scaled = mvg_data_scaled.reshape(
        n_split_windows * mvg_data.shape[0], -1, 3
    )
    # repeat labels
    data_labels_scaled = np.repeat(data_labels, n_split_windows)

    return mvg_data_scaled, data_labels_scaled


def main():
    path_to_dataset = Path("data/CNC_Machining").absolute()

    save_as_np_files(path_to_dataset, verbose=True)
    data_arrays, data_labels = load_np_files(path_to_dataset, verbose=True)

    mvg_data_scaled, data_labels_scaled = preprocess_ds(
        data_arrays,
        data_labels,
        mvg_window_size=300,
        mvg_step_size=150,
        reduce_length=2000,
        n_split_windows=10,
    )

    np.save(path_to_dataset / "mvg_data_scaled.npy", mvg_data_scaled)
    np.save(path_to_dataset / "mvg_labels.npy", data_labels_scaled)


if __name__ == "__main__":
    main()
