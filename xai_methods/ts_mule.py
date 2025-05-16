"""
This file contains the implementation of the TS-MULE explainer.
Ref: https://github.com/dbvis-ukon/ts-mule/tree/main 
"""
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from xai_methods.xai_blackbox_base import BlackBoxExplainer
from tqdm import tqdm
import scipy.stats
from scipy.signal import find_peaks


class TimeSeriesSegmenter:
    """
    Base class for time series segmentation strategies.
    
    Time series segmentation divides a continuous time series into discrete, 
    non-overlapping intervals (segments) where each segment represents a specific
    pattern or behavior in the data. Segmentation is the foundation of TS-MULE,
    enabling local interpretability by identifying coherent regions of the time series.
    
    Attributes:
        segment_size (int): The default size of segments to create (implementation-specific)
    """
    
    def __init__(self, segment_size: int = 5):
        self.segment_size = segment_size
    
    @abstractmethod
    def segment(self, ts: np.ndarray) -> list[tuple[int, int]]:
        """
        Segment a time series into intervals.
        
        Args:
            ts (np.ndarray): Time series data of shape (length, features) or (length,)
            
        Returns:
            List[Tuple[int, int]]: List of (start, end) tuples representing segments,
                where start is inclusive and end is exclusive
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError


class UniformSegmenter(TimeSeriesSegmenter):
    """Uniform segmentation based on a fixed window size."""
    
    def segment(self, ts: np.ndarray) -> list[tuple[int, int]]:
        length = ts.shape[0]
        segments = []
        
        for i in range(0, length, self.segment_size):
            end = min(i + self.segment_size, length)
            segments.append((i, end))
            
        return segments


class ExponentialSegmenter(TimeSeriesSegmenter):
    """Exponential segmentation based on exponentially growing window."""
    
    def __init__(self, base: float = 2.0, min_size: int = 1):
        super().__init__()
        self.base = base
        self.min_size = min_size
    
    def segment(self, ts: np.ndarray) -> list[tuple[int, int]]:
        length = ts.shape[0]
        segments = []
        
        i = 0
        while i < length:
            # Calculate segment size based on exponential growth
            size = max(self.min_size, int(self.base ** (len(segments) + 1)))
            end = min(i + size, length)
            segments.append((i, end))
            i = end
            
        return segments


class SAXSegmenter(TimeSeriesSegmenter):
    """
    Segmentation based on Symbolic Aggregate approXimation (SAX).
    
    SAX transforms a time series into a symbolic representation by first applying
    Piecewise Aggregate Approximation (PAA) to reduce dimensionality, then quantizing
    the resulting values into equiprobable regions using Gaussian breakpoints.
    
    This segmenter creates segments at points where the SAX symbols change, which
    naturally align with significant pattern changes in the time series.
    
    Attributes:
        num_symbols (int): Number of symbol types to use in the SAX representation
        word_size (int): Number of PAA segments to create
    
    References:
        Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). 
        Experiencing SAX: a novel symbolic representation of time series. 
        Data Mining and Knowledge Discovery, 15(2), 107-144.
    """
    
    def __init__(self, num_symbols: int = 3, word_size: int = 3):
        """
        Initialize SAX segmenter.
        
        Args:
            num_symbols (int): Number of discrete symbols to use (alphabet size)
            word_size (int): Number of PAA segments to create (word length)
        """
        super().__init__()
        self.num_symbols = num_symbols
        self.word_size = word_size
    
    def _to_sax(self, ts: np.ndarray) -> np.ndarray:
        """
        Convert time series to SAX symbols.
        
        Process:
        1. Z-normalize the time series
        2. Apply PAA (Piecewise Aggregate Approximation)
        3. Convert PAA values to symbols using Gaussian breakpoints
        
        Args:
            ts (np.ndarray): 1D time series data
            
        Returns:
            np.ndarray: Array of SAX symbols (integers from 0 to num_symbols-1)
        """
        # Z-normalize the time series
        z_ts = (ts - np.mean(ts)) / np.std(ts)
        
        # PAA (Piecewise Aggregate Approximation)
        length = ts.shape[0]
        window_size = length // self.word_size
        paa = np.array([np.mean(z_ts[i:i+window_size]) for i in range(0, length - window_size + 1, window_size)])
        
        # SAX symbolization using Gaussian breakpoints
        breakpoints = scipy.stats.norm.ppf(np.linspace(0, 1, self.num_symbols + 1)[1:-1])
        symbols = np.zeros_like(paa, dtype=int)
        
        for i, point in enumerate(paa):
            symbols[i] = np.sum(point > breakpoints)
            
        return symbols
    
    def segment(self, ts: np.ndarray) -> list[tuple[int, int]]:
        """
        Segment a time series based on SAX symbol changes.
        
        Creates segments at points where the symbolic representation changes,
        indicating shifts in the underlying pattern of the time series.
        
        Args:
            ts (np.ndarray): Time series data of shape (length, features) or (length,)
            
        Returns:
            List[Tuple[int, int]]: List of (start, end) tuples representing segments
        """
        # Handle multidimensional time series by using the first dimension
        if len(ts.shape) > 1:
            ts_1d = ts[:, 0]
        else:
            ts_1d = ts
            
        symbols = self._to_sax(ts_1d)
        length = ts.shape[0]
        window_size = length // self.word_size
        segments = []
        
        # Create segments where SAX symbols change
        prev_symbol = symbols[0]
        start = 0
        
        for i in range(1, len(symbols)):
            if symbols[i] != prev_symbol:
                end = min(i * window_size, length)
                segments.append((start, end))
                start = end
                prev_symbol = symbols[i]
        
        # Add the last segment
        if start < length:
            segments.append((start, length))
            
        return segments


class MatrixProfileSegmenter(TimeSeriesSegmenter):
    """Base class for segmentation based on matrix profile."""
    
    def __init__(self, window_size: int = 5, n_segments: int = 10):
        super().__init__(segment_size=window_size)
        self.n_segments = n_segments
    
    def _compute_matrix_profile(self, ts: np.ndarray) -> np.ndarray:
        """Compute the matrix profile of a time series."""
        if len(ts.shape) > 1:
            ts = ts[:, 0]  # Use first dimension for matrix profile
            
        n = len(ts)
        w = self.segment_size
        
        # Simple matrix profile computation (this could be optimized)
        profile = np.zeros(n - w + 1)
        
        for i in range(n - w + 1):
            subsequence = ts[i:i+w]
            distances = np.zeros(n - w + 1)
            
            for j in range(n - w + 1):
                if abs(i - j) < w:  # Avoid trivial matches
                    distances[j] = np.inf
                else:
                    distances[j] = np.sqrt(np.sum((subsequence - ts[j:j+w])**2))
            
            profile[i] = np.min(distances[distances != np.inf])
            
        return profile
    
    @abstractmethod
    def _find_segment_points(self, profile: np.ndarray) -> list[int]:
        """Find segment points based on the matrix profile."""
        raise NotImplementedError
    
    def segment(self, ts: np.ndarray) -> list[tuple[int, int]]:
        profile = self._compute_matrix_profile(ts)
        segment_points = self._find_segment_points(profile)
        segments = []
        
        for i in range(len(segment_points) - 1):
            segments.append((segment_points[i], segment_points[i+1]))
            
        return segments


class SlopesSortedSegmenter(MatrixProfileSegmenter):
    """Segmentation based on sorted gradients of the matrix profile."""
    
    def _find_segment_points(self, profile: np.ndarray) -> list[int]:
        """
        Find segment points based on sorted gradient magnitudes.
        
        Args:
            profile (np.ndarray): The matrix profile of the time series
            
        Returns:
            List[int]: Sorted list of segment points (indexes in the time series)
        """
        # Compute gradients
        gradients = np.gradient(profile)
        
        # Sort points by absolute gradient magnitude
        sorted_indices = np.argsort(np.abs(gradients))[::-1]
        
        # Select top n_segments-1 points (plus start and end)
        top_indices = sorted_indices[:self.n_segments-1].tolist()
        segment_points = sorted([0] + top_indices + [len(profile)])
        
        return segment_points


class SlopesNotSortedSegmenter(MatrixProfileSegmenter):
    """Segmentation based on threshold of gradients of the matrix profile."""
    
    def _find_segment_points(self, profile: np.ndarray) -> list[int]:
        """
        Find segment points at peaks in gradient magnitude.
        
        Args:
            profile (np.ndarray): The matrix profile of the time series
            
        Returns:
            List[int]: Sorted list of segment points (indexes in the time series)
        """
        # Compute gradients
        gradients = np.gradient(profile)
        
        # Find peaks in the gradient magnitude
        peaks, _ = find_peaks(np.abs(gradients), height=np.std(gradients))
        
        # Limit to n_segments-1 peaks
        if len(peaks) > self.n_segments - 1:
            peak_heights = np.abs(gradients[peaks])
            top_indices = np.argsort(peak_heights)[:-(self.n_segments-1):-1]
            peaks = peaks[top_indices]
        
        # Add start and end points
        peaks_list = peaks.tolist()
        segment_points = sorted([0] + peaks_list + [len(profile)])
        
        return segment_points


class BinsSegmenter(MatrixProfileSegmenter):
    """Base class for binning-based segmentation of the matrix profile."""
    
    @abstractmethod
    def _select_from_bin(self, bin_values: np.ndarray) -> float:
        """Select a representative value from a bin."""
        raise NotImplementedError
    
    def _find_segment_points(self, profile: np.ndarray) -> list[int]:
        # Bin the profile into n_segments bins
        bins = np.array_split(np.arange(len(profile)), self.n_segments)
        
        # Get start and end of each bin
        segment_points = [bins[0][0]]
        
        for bin_indices in bins:
            if len(bin_indices) > 0:
                segment_points.append(bin_indices[-1] + 1)
        
        # Ensure the last point is included
        if segment_points[-1] != len(profile):
            segment_points.append(len(profile))
        
        return segment_points


class BinsMinSegmenter(BinsSegmenter):
    """Segmentation using minimum value in bins of the matrix profile."""
    
    def _select_from_bin(self, bin_values: np.ndarray) -> float:
        return np.min(bin_values)


class BinsMaxSegmenter(BinsSegmenter):
    """Segmentation using maximum value in bins of the matrix profile."""
    
    def _select_from_bin(self, bin_values: np.ndarray) -> float:
        return np.max(bin_values)


class TimeSeriesPerturber:
    """Base class for time series perturbation strategies."""
    
    @abstractmethod
    def perturb(self, ts: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
        """
        Perturb a segment of the time series.
        
        Args:
            ts: Time series data
            segment: (start, end) tuple defining the segment to perturb
            
        Returns:
            Perturbed time series
        """
        raise NotImplementedError


class ZeroPerturber(TimeSeriesPerturber):
    """Replace segment with zeros."""
    
    def perturb(self, ts: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
        start, end = segment
        perturbed = ts.copy()
        perturbed[start:end] = 0
        return perturbed


class InversePerturber(TimeSeriesPerturber):
    """Replace segment with inverse values."""
    
    def perturb(self, ts: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
        start, end = segment
        perturbed = ts.copy()
        perturbed[start:end] = -perturbed[start:end]
        return perturbed


class MeanPerturber(TimeSeriesPerturber):
    """Replace segment with mean value of the time series."""
    
    def perturb(self, ts: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
        start, end = segment
        perturbed = ts.copy()
        
        if len(ts.shape) > 1:
            # Handle multidimensional time series
            means = np.mean(ts, axis=0)
            perturbed[start:end] = np.tile(means, (end - start, 1))
        else:
            # Handle 1D time series
            mean = np.mean(ts)
            perturbed[start:end] = mean
            
        return perturbed


class TSMULE(BlackBoxExplainer):
    """
    Time Series - Model-agnostic, User-friendly, Local Explanations (TS-MULE).
    
    TS-MULE adapts LIME (Local Interpretable Model-agnostic Explanations) for time series data
    by using specialized segmentation strategies to create meaningful feature groups.
    The approach works by:
    
    1. Segmenting the input time series into interpretable segments
    2. Generating perturbed samples by systematically removing/altering segments
    3. Measuring the model's response to these perturbations
    4. Fitting a simple linear model to determine segment importance
    5. Converting segment importances back to a time series format
    
    This process enables local interpretability by identifying which segments
    of the time series most influence the model's prediction.
    
    Attributes:
        segmenter (TimeSeriesSegmenter): Strategy for segmenting time series
        perturber (TimeSeriesPerturber): Strategy for perturbing segments
        num_samples (int): Number of perturbation samples to generate
        kernel_width (float): Width parameter for the RBF kernel
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        dataset_type: str,
        segmenter: TimeSeriesSegmenter,
        perturber: TimeSeriesPerturber,
        num_samples: int = 1000,
        use_latent_input: bool = False,
        kernel_width: float = 0.75,
        conf: dict[str, any] | None = None,
    ):
        """
        Initialize TS-MULE explainer.
        
        Args:
            model (nn.Module): PyTorch model to explain
            model_type (str): Type of model (e.g., "MLP", "LSTM", "Transformer")
            dataset_type (str): Type of dataset being analyzed
            segmenter (TimeSeriesSegmenter): Strategy for segmenting time series
            perturber (TimeSeriesPerturber): Strategy for perturbing segments
            num_samples (int): Number of perturbation samples to generate
            use_latent_input (bool): Whether the model operates on latent representations
            kernel_width (float): Width parameter for the RBF kernel in the LIME algorithm
            conf (dict[str, Any] | None): Additional configuration parameters
        """
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)
        self.segmenter = segmenter
        self.perturber = perturber
        self.num_samples = num_samples
        self.kernel_width = kernel_width
    
    def _generate_perturbations(
        self, 
        ts: np.ndarray, 
        segments: list[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbed samples for a time series.
        
        Creates multiple versions of the input time series by selectively perturbing
        different segments. Each perturbation randomly alters approximately 50% of 
        segments using the configured perturbation strategy.
        
        Args:
            ts (np.ndarray): Time series data of shape (seq_length, feature_dim)
            segments (List[Tuple[int, int]]): List of (start, end) tuples representing segments
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Array of perturbed samples with shape (num_samples, seq_length, feature_dim)
                - Binary perturbation indicators with shape (num_samples, num_segments),
                  where 1 indicates original segment retained, 0 indicates perturbed segment
        """
        n_segments = len(segments)
        perturbations = []
        perturbation_labels = []
        
        # Original sample
        perturbations.append(ts.copy())
        perturbation_labels.append(np.ones(n_segments))
        
        # Generate random perturbations
        for _ in range(self.num_samples - 1):
            perturbed = ts.copy()
            binary_vector = np.ones(n_segments)
            
            # Randomly perturb segments
            for i, segment in enumerate(segments):
                if np.random.random() < 0.5:  # 50% chance to perturb
                    perturbed = self.perturber.perturb(perturbed, segment)
                    binary_vector[i] = 0
            
            perturbations.append(perturbed)
            perturbation_labels.append(binary_vector)
        
        return np.array(perturbations), np.array(perturbation_labels)
    
    def _compute_distances(self, perturbation_labels: np.ndarray) -> np.ndarray:
        """
        Compute distances from original sample to perturbations.
        
        Calculates how different each perturbed sample is from the original
        by counting the number of segments that have been altered.
        
        Args:
            perturbation_labels (np.ndarray): Binary vectors indicating perturbation status,
                where 1 means segment is unchanged and 0 means segment is perturbed
            
        Returns:
            np.ndarray: Vector of distances, where each value represents the
                number of perturbed segments in the corresponding sample
        """
        # First perturbation is the original, so distance is 0
        distances = np.zeros(len(perturbation_labels))
        
        # Compute L1 distance for other perturbations
        original = perturbation_labels[0]
        for i in range(1, len(perturbation_labels)):
            # L1 distance (number of perturbed segments)
            distances[i] = np.sum(original != perturbation_labels[i])
        
        return distances
    
    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute weights using RBF kernel.
        
        Converts distances to weights using a radial basis function (RBF) kernel.
        Samples closer to the original (smaller distances) receive higher weights,
        emphasizing their importance in the linear model fitting process.
        
        Args:
            distances (np.ndarray): Vector of distances between original and perturbed samples
            
        Returns:
            np.ndarray: Vector of weights, where higher values indicate samples
                more similar to the original
        """
        return np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))
    
    def _fit_linear_model(
        self, 
        perturbation_labels: np.ndarray, 
        predictions: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Fit weighted linear model to get feature importances.
        
        Uses weighted least squares regression to learn a linear relationship between
        segment presence/absence and model predictions. The coefficients of this linear
        model represent the importance of each segment to the prediction.
        
        Args:
            perturbation_labels (np.ndarray): Binary vectors indicating perturbation status
                with shape (num_samples, num_segments)
            predictions (np.ndarray): Model predictions for perturbed samples
                with shape (num_samples, num_classes)
            weights (np.ndarray): Sample weights with shape (num_samples,)
            
        Returns:
            np.ndarray: Coefficient vector representing segment importances
                with shape (num_segments,)
        """
        # Handle multiclass case - use prediction for predicted class
        if predictions.shape[1] > 1:
            # Get the most likely class from the original prediction
            predicted_class = np.argmax(predictions[0])
            y = predictions[:, predicted_class]
        else:
            y = predictions.ravel()
        
        # Use weighted linear regression
        diag_weights = np.diag(weights)
        weighted_x = np.matmul(diag_weights, perturbation_labels)
        weighted_y = weights * y
        
        # Compute coefficients using normal equation
        xtx = np.matmul(weighted_x.T, weighted_x)
        xty = np.matmul(weighted_x.T, weighted_y)
        
        # Add small value to diagonal for numerical stability
        xtx += np.eye(xtx.shape[0]) * 1e-8
        
        coefficients = np.linalg.solve(xtx, xty)
        return coefficients
    
    def _convert_segment_importances_to_timeseries(
        self, 
        importances: np.ndarray, 
        segments: list[tuple[int, int]], 
        ts_length: int
    ) -> np.ndarray:
        """
        Convert segment importances to time series format.
        
        Maps the importance score for each segment back to the original time series format,
        assigning the same importance value to all time points within a segment.
        
        Args:
            importances (np.ndarray): Segment importance coefficients with shape (num_segments,)
            segments (List[Tuple[int, int]]): List of (start, end) tuples representing segments
            ts_length (int): Length of the original time series
            
        Returns:
            np.ndarray: Time series of importances with shape (ts_length,),
                where each point has the importance value of its containing segment
        """
        ts_importances = np.zeros(ts_length)
        
        for i, (start, end) in enumerate(segments):
            ts_importances[start:end] = importances[i]
        
        return ts_importances
    
    def predict_fn(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Converts input data to appropriate format and passes it through the model
        to get prediction probabilities.
        
        Args:
            input_tensor (np.ndarray): Input data tensor with shape 
                (batch_size, seq_length) or (batch_size, seq_length, feature_dim)
            
        Returns:
            np.ndarray: Model prediction probabilities with shape (batch_size, num_classes)
        """
        if isinstance(input_tensor, np.ndarray):
            if self.use_latent_input:
                input_tensor_torch = torch.tensor(input_tensor, dtype=torch.int64)
                if hasattr(self.model, 'device'):
                    input_tensor_torch = input_tensor_torch.to(self.model.device)
            else:
                input_tensor_torch = torch.tensor(input_tensor, dtype=torch.float32)
                if hasattr(self.model, 'device'):
                    input_tensor_torch = input_tensor_torch.to(self.model.device)
        else:
            input_tensor_torch = input_tensor
            
        # Ensure correct shape for model
        if len(input_tensor_torch.shape) == 2:
            input_tensor_torch = input_tensor_torch.unsqueeze(-1)
        
        return self.classify(input_tensor_torch)
    
    def explain(
        self, 
        input_tensor: torch.Tensor,
        save_to_pickle: bool = False,
        save_path: str = "tsmule_explanations.pkl"
    ) -> torch.Tensor:
        """
        Explain model predictions for input time series.
        
        Generates local explanations by:
        1. Segmenting each input time series
        2. Creating perturbed versions by altering segments
        3. Running these through the model and analyzing prediction changes
        4. Building a linear surrogate model to identify important segments
        5. Mapping importance scores back to the original time series format
        
        Args:
            input_tensor (torch.Tensor): Input time series data of shape (batch_size, seq_length)
                or (batch_size, seq_length, feature_dim)
            save_to_pickle (bool): Whether to save explanations to a pickle file
            save_path (str): Path where to save the explanations
            
        Returns:
            torch.Tensor: Tensor of importance scores for each time step with the same
                shape as the input tensor, where higher values indicate greater influence
                on the model's prediction
        """
        # Convert input to numpy if needed
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = input_tensor
            
        batch_size = input_np.shape[0]
        seq_length = input_np.shape[1]
        
        # Prepare output tensor
        if len(input_np.shape) > 2:
            feature_dim = input_np.shape[2]
            explanations = np.zeros((batch_size, seq_length, feature_dim))
        else:
            explanations = np.zeros((batch_size, seq_length))
            feature_dim = 1
            input_np = input_np.reshape((batch_size, seq_length, feature_dim))
        
        # Process each instance in the batch
        for i in tqdm(range(batch_size), desc="Explaining instances"):
            ts = input_np[i]
            
            # Segment the time series
            segments = self.segmenter.segment(ts)
            
            # Generate perturbations
            perturbed_samples, perturbation_labels = self._generate_perturbations(ts, segments)
            
            # Reshape perturbed samples for prediction
            if feature_dim == 1:
                perturbed_samples_reshaped = perturbed_samples.reshape((-1, seq_length))
            else:
                perturbed_samples_reshaped = perturbed_samples
                
            # Get model predictions for perturbations
            predictions = self.predict_fn(perturbed_samples_reshaped)
            
            # Compute weights based on distances
            distances = self._compute_distances(perturbation_labels)
            weights = self._compute_weights(distances)
            
            # Fit linear model to get feature importances
            segment_importances = self._fit_linear_model(perturbation_labels, predictions, weights)
            
            # Convert segment importances to time series format
            ts_importances = self._convert_segment_importances_to_timeseries(
                segment_importances, segments, seq_length
            )
            
            # Store in output tensor
            if feature_dim == 1:
                explanations[i] = ts_importances
            else:
                # Replicate importances across all features
                for j in range(feature_dim):
                    explanations[i, :, j] = ts_importances
        
        # Convert to torch tensor
        explanations_tensor = torch.tensor(explanations)
        
        # Save if requested
        if save_to_pickle:
            self.save(explanations_tensor, save_path)
            
        return explanations_tensor


# Factory functions to create segmenters and perturbers

def create_segmenter(method: str, **kwargs) -> TimeSeriesSegmenter:
    """
    Create a time series segmenter.
    
    Args:
        method: Segmentation method name
        **kwargs: Additional parameters for the segmenter
        
    Returns:
        TimeSeriesSegmenter instance
    """
    if method == "uniform":
        return UniformSegmenter(**kwargs)
    elif method == "exponential":
        return ExponentialSegmenter(**kwargs)
    elif method == "sax":
        return SAXSegmenter(**kwargs)
    elif method == "slopes-sorted":
        return SlopesSortedSegmenter(**kwargs)
    elif method == "slopes-not-sorted":
        return SlopesNotSortedSegmenter(**kwargs)
    elif method == "bins-min":
        return BinsMinSegmenter(**kwargs)
    elif method == "bins-max":
        return BinsMaxSegmenter(**kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

def create_perturber(method: str) -> TimeSeriesPerturber:
    """
    Create a time series perturber.
    
    Args:
        method: Perturbation method name
        
    Returns:
        TimeSeriesPerturber instance
    """
    if method == "zero":
        return ZeroPerturber()
    elif method == "inverse":
        return InversePerturber()
    elif method == "mean":
        return MeanPerturber()
    else:
        raise ValueError(f"Unknown perturbation method: {method}")
