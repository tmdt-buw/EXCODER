import logging as log
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from typing import Literal
from data_loader.utils import create_sequence_ds, MyScaler
from data_loader.dataset import ReconDataset, ClassificationDataset


class DataModuleBase(pl.LightningDataModule):

    def __init__(
        self,
        dataset_path: Path,
        ds_type: Literal["reconstruction", "classification"] = "reconstruction",
        val_split_idx: np.ndarray | None = None,
        test_split_idx: np.ndarray | None = None,
        batch_size: int = 256,
        shuffle_train: bool = True,
        weigthed_sampling: bool = False,
    ):
        self.dataset_path = dataset_path
        ds, labels = self.load_raw_data()
        self.ds = ds
        self.labels = labels
        self.val_idx = val_split_idx
        self.test_idx = test_split_idx
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.ds_type = ds_type
        self.weigthed_sampling = weigthed_sampling
        self.scaler = MyScaler()
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        super().__init__()

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the raw data from the dataset path.

        Returns:
            tuple[np.ndarray, np.ndarray]: The raw data and labels.
        """
        raise NotImplementedError("load_raw_data method not implemented")

    def scale_data(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
        """
        Scale the input data using a scaler object.

        Args:
            x_train (np.ndarray): The training data array.
            x_val (np.ndarray): The validation data array.
            x_test (np.ndarray): The test data array.

        Returns:
            np.ndarray: The scaled training data array.
            np.ndarray: The scaled validation data array.
            np.ndarray: The scaled test data array.
        """
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        x_val = self.scaler.transform(x_val)
        x_test = self.scaler.transform(x_test)
        return x_train, x_val, x_test

    def split_train_val_test_by_index(
        self, train_x: np.ndarray, train_y: np.ndarray, seq_len: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the training data into training, validation, and test sets based on provided indices.

        Args:
            train_x (np.ndarray): The training data array.
            train_y (np.ndarray): The training labels array.
            seq_len (int, optional): The sequence length for creating sequence datasets. Defaults to 1.

        Returns:
            np.ndarray: The training data array.
            np.ndarray: The validation data array.
            np.ndarray: The test data array.
            np.ndarray: The training labels array.
            np.ndarray: The validation labels array.
            np.ndarray: The test labels array.
        """
        num_samples = train_x.shape[0]
        val_idx = self.val_idx
        test_idx = self.test_idx
        if seq_len > 1:
            val_idx = val_idx[val_idx < num_samples - seq_len]
            test_idx = test_idx[test_idx < num_samples - seq_len]

        val_test_indices = np.concatenate([val_idx, test_idx])

        x_train = train_x[~np.isin(np.arange(num_samples), val_test_indices)]
        x_val = train_x[val_idx]
        x_test = train_x[test_idx]
        y_train = train_y[~np.isin(np.arange(num_samples), val_test_indices)]
        y_val = train_y[val_idx]
        y_test = train_y[test_idx]
        return x_train, x_val, x_test, y_train, y_val, y_test

    def train_val_test_split(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_size: float,
        test_size: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits a dataset into training, validation, and testing sets using np.random.choice.

        Args:
            train_x: A numpy array of the training features.
            train_y: A numpy array of the training labels.
            val_size: The proportion of the data to be used for validation (between 0 and 1).
            test_size: The proportion of the data to be used for testing (between 0 and 1).

        Returns:
            A tuple of six numpy arrays: (x_train, x_val, x_test, y_train, y_val, y_test).
        """
        if test_size is None:
            test_size = len(self.test_idx) / train_x.shape[0]

        # Check if sizes are valid
        if not 0 <= val_size <= 1 or not 0 <= test_size <= 1:
            raise ValueError("Validation and test sizes must be between 0 and 1.")

        if val_size + test_size > 1:
            raise ValueError("Combined validation and test size cannot exceed 1.")

        # Calculate the number of samples for each set
        num_samples = train_x.shape[0]
        num_val = int(num_samples * val_size)
        if self.test_idx is None:
            num_test = int(num_samples * test_size)

            # Randomly select indices for validation and test sets
            val_test_indices = np.random.choice(
                num_samples, num_val + num_test, replace=False
            )
            # val_test_indices = premute_windows(num_samples=num_samples, n=200)
            val_indices = val_test_indices[:num_val]
            test_indices = val_test_indices[num_val:]
        else:
            num_test = len(self.test_idx)
            val_indices = np.random.choice(
                num_samples - num_test, num_val, replace=False
            )
            test_indices = self.test_idx
            val_test_indices = np.concatenate([val_indices, test_indices])

        # Use boolean indexing to create the splits
        x_train = train_x[~np.isin(np.arange(num_samples), val_test_indices)]
        x_val = train_x[val_indices]
        x_test = train_x[test_indices]
        y_train = train_y[~np.isin(np.arange(num_samples), val_test_indices)]
        y_val = train_y[val_indices]
        y_test = train_y[test_indices]

        self.val_idx = val_indices
        self.test_idx = test_indices
        return x_train, x_val, x_test, y_train, y_val, y_test

    def setup(self, stage: str | None = None, seq_len: int = 1):
        """
        Setup the data module by preprocessing the data and creating the appropriate datasets.

        Args:
            stage (None, optional): The current stage of training. Defaults to None.
            seq_len (int, optional): The sequence length for creating sequence datasets. Defaults to 1.
        """
        if seq_len > 1:
            raise NotImplementedError("seq_len > 1 not implemented")
        if self.val_idx is not None and self.test_idx is not None:
            log.info("Using provided val and test indices")
            x_train, x_val, x_test, y_train, y_val, y_test = (
                self.split_train_val_test_by_index(train_x=self.ds, train_y=self.labels)
            )
        else:
            x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(
                train_x=self.ds,
                train_y=self.labels,
                val_size=0.1,
                test_size=0.1 if self.test_idx is None else len(self.test_idx),
            )

        x_train, x_val, x_test = self.scale_data(
            x_train=x_train, x_val=x_val, x_test=x_test
        )

        if self.ds_type == "reconstruction":
            self.train_ds = ReconDataset(x_train)
            self.val_ds = ReconDataset(x_val)
            self.test_ds = ReconDataset(x_test)
        elif self.ds_type == "classification":
            self.train_ds = ClassificationDataset(x_train, y_train)
            self.val_ds = ClassificationDataset(x_val, y_val)
            self.test_ds = ClassificationDataset(x_test, y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


class WeldingDataModule(DataModuleBase):
    """
    LightningDataModule subclass for handling data loading and preprocessing.

    Args:
        dataset_path (Path): The path to the dataset.
        ds_type (Literal["reconstruction", "classification"], optional): The type of dataset. Defaults to "reconstruction".
        val_split_idx (np.ndarray | None, optional): The indices to use for validation split. Defaults to None.
        test_split_idx (np.ndarray | None, optional): The indices to use for test split. Defaults to None.
        batch_size (int, optional): The batch size for data loading. Defaults to 256.
        shuffle_train (bool, optional): Whether to shuffle the training data. Defaults to True.
    """

    def __init__(
        self,
        dataset_path: Path,
        ds_type: Literal["reconstruction", "classification"] = "reconstruction",
        val_split_idx: np.ndarray | None = None,
        test_split_idx: np.ndarray | None = None,
        batch_size: int = 256,
        shuffle_train: bool = True,
        weigthed_sampling: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            ds_type=ds_type,
            val_split_idx=val_split_idx,
            test_split_idx=test_split_idx,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            weigthed_sampling=weigthed_sampling,
        )

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the raw data from the dataset path.

        Returns:
            tuple[np.ndarray, np.ndarray]: The raw data and labels.
        """
        cycles_path = self.dataset_path / "ds_1_4_data.npy"
        id_data_path = self.dataset_path / "ds_1_4_quality.npy"

        ds = np.load(cycles_path)
        id_ds = np.load(id_data_path)
        labels = id_ds[:, 3]
        return ds, labels

    def setup(
        self,
        stage=None,
        seq_len: int = 1,
    ):
        """
        Setup the data module by preprocessing the data and creating the appropriate datasets.

        Args:
            stage (None, optional): The current stage of training. Defaults to None.
            seq_len (int, optional): The sequence length for creating sequence datasets. Defaults to 1.
        """
        if seq_len > 1:
            ds, labels = create_sequence_ds(self.ds, self.labels, seq_len)
            log.info(
                f"Creating sequence dataset {seq_len} | new dataset shape: {ds.shape}"
            )
        else:
            ds, labels = self.ds, self.labels

        if self.val_idx is not None and self.test_idx is not None:
            log.info("Using provided val and test indices")
            x_train, x_val, x_test, y_train, y_val, y_test = (
                self.split_train_val_test_by_index(
                    train_x=ds, train_y=labels, seq_len=seq_len
                )
            )
        else:
            x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(
                train_x=ds, train_y=labels, val_size=0.1, test_size=0.1
            )
        log.info(
            f"Train shape: {x_train.shape} Val shape: {x_val.shape} Test shape: {x_test.shape}"
        )
        log.info(
            f"Train labels shape: {y_train.shape} Val labels shape: {y_val.shape} Test labels shape: {y_test.shape}"
        )
        x_train, x_val, x_test = self.scale_data(
            x_train=x_train, x_val=x_val, x_test=x_test
        )

        if self.ds_type == "reconstruction":
            # assert seq_len == 1, "Reconstruction dataset only works with seq_len=1"
            self.train_ds = ReconDataset(x_train)
            self.val_ds = ReconDataset(x_val)
            self.test_ds = ReconDataset(x_test)
        elif self.ds_type == "classification":
            x_train, y_train = self.filter_out_not_labbeld(x_train, y_train)
            x_val, y_val = self.filter_out_not_labbeld(x_val, y_val)
            x_test, y_test = self.filter_out_not_labbeld(x_test, y_test)

            self.train_ds = ClassificationDataset(x_train, y_train)
            self.val_ds = ClassificationDataset(x_val, y_val)
            self.test_ds = ClassificationDataset(x_test, y_test)
        else:
            raise ValueError("ds_type must be 'reconstruction' or 'classification'")

    @staticmethod
    def filter_out_not_labbeld(ds: np.ndarray, labels: np.ndarray):
        """
        Filter out samples with labels equal to -1.

        Args:
            ds (np.ndarray): The input data array.
            labels (np.ndarray): The labels array.

        Returns:
            np.ndarray: The filtered data array.
            np.ndarray: The filtered labels array.
        """
        return ds[labels != -1], labels[labels != -1]


class CNCDataModule(DataModuleBase):
    """
    LightningDataModule for handling CNC machining data.

    Loads and processes CNC machining data from scaled numpy arrays, supporting both
    reconstruction and classification tasks.

    Args:
        dataset_path (Path): Path to the dataset directory
        ds_type (Literal["reconstruction", "classification"]): Type of dataset to create
        val_split_idx (np.ndarray | None): Optional validation split indices
        test_split_idx (np.ndarray | None): Optional test split indices
        batch_size (int): Batch size for dataloaders
        shuffle_train (bool): Whether to shuffle training data
    """

    def __init__(
        self,
        dataset_path: Path,
        ds_type: Literal["reconstruction", "classification"] = "reconstruction",
        val_split_idx: np.ndarray | None = None,
        test_split_idx: np.ndarray | None = None,
        batch_size: int = 256,
        shuffle_train: bool = True,
    ):
        super().__init__(
            dataset_path=dataset_path,
            ds_type=ds_type,
            val_split_idx=val_split_idx,
            test_split_idx=test_split_idx,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
        )

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray]:
        mvg_data = np.load(self.dataset_path / "mvg_data_scaled.npy")
        mvg_labels = np.load(self.dataset_path / "mvg_labels.npy")
        return mvg_data, mvg_labels


class ECGDataModule(DataModuleBase):
    """
    LightningDataModule for handling ECG (electrocardiogram) data.
    
    Loads and processes ECG data from MIT-BIH Arrhythmia Database CSV files, supporting both
    reconstruction and classification tasks.

    Args:
        dataset_path (Path): Path to the dataset directory containing mitbih CSV files
        ds_type (Literal["reconstruction", "classification"]): Type of dataset to create
        val_split_idx (np.ndarray | None): Optional validation split indices
        test_split_idx (np.ndarray | None): Optional test split indices
        batch_size (int): Batch size for dataloaders
        shuffle_train (bool): Whether to shuffle training data
    """

    def __init__(
        self,
        dataset_path: Path,
        ds_type: Literal["reconstruction", "classification"] = "reconstruction",
        val_split_idx: np.ndarray | None = None,
        test_split_idx: np.ndarray | None = None,
        batch_size: int = 256,
        shuffle_train: bool = True,
    ):
        super().__init__(
            dataset_path=dataset_path,
            ds_type=ds_type,
            val_split_idx=val_split_idx,
            test_split_idx=test_split_idx,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
        )

    @staticmethod
    def pad_sequences(data: np.ndarray, target_length: int) -> np.ndarray:
        """
        Zero pad sequences to target length along axis 1.
        
        Args:
            data (np.ndarray): Input array of shape (N, seq_len, channels)
            target_length (int): Desired sequence length
            
        Returns:
            np.ndarray: Padded array of shape (N, target_length, channels)
        """
        current_length = data.shape[1]
        pad_length = target_length - current_length
        
        if pad_length <= 0:
            return data
            
        # Create padding tuple: ((0,0) for first dim, (0,pad_length) for second dim, (0,0) for last dim)
        padding = ((0, 0), (0, pad_length), (0, 0))
        return np.pad(data, padding, mode='constant', constant_values=0)

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray]:
        ecg_train_path = self.dataset_path / "mitbih_train.csv"
        ecg_test_path = self.dataset_path / "mitbih_test.csv"
        ecg_train = pd.read_csv(ecg_train_path, header=None)
        ecg_test = pd.read_csv(ecg_test_path, header=None)

        ecg_train_data = ecg_train.iloc[:, :-1].values
        ecg_test_data = ecg_test.iloc[:, :-1].values
        ecg_train_labels = ecg_train.iloc[:, -1].values
        ecg_test_labels = ecg_test.iloc[:, -1].values

        self.test_idx = np.arange(ecg_test_data.shape[0]) + ecg_train_data.shape[0]

        ds = np.concatenate([ecg_train_data, ecg_test_data])
        ds = np.expand_dims(ds, axis=-1)  # add channel dimension
        ds = self.pad_sequences(ds, target_length=200)
        labels = np.concatenate([ecg_train_labels, ecg_test_labels])
        return ds, labels


class SimpleDataModule(pl.LightningDataModule):
    """
    LightningDataModule for transformer model data handling.

    Provides data loading functionality for transformer models with configurable batch sizes
    and separate train/val/test datasets.

    Args:
        train_ds (Dataset): Training dataset
        val_ds (Dataset): Validation dataset
        test_ds (Dataset): Test dataset
        batch_size (int, optional): Size of each batch. Defaults to 256.
    """

    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        batch_size: int = 256,
    ) -> None:
        """
        Initialize the TransformerDataModule with provided datasets and batch size.

        Args:
            train_ds (Dataset): Training dataset
            val_ds (Dataset): Validation dataset
            test_ds (Dataset): Test dataset
            batch_size (int, optional): Size of each batch. Defaults to 256.
        """
        self.batch_size = batch_size
        self.train_ds: Dataset = train_ds
        self.val_ds: Dataset = val_ds
        self.test_ds: Dataset = test_ds
        super().__init__()

    def setup(self, stage=None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
