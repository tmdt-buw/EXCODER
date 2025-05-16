import torch
import numpy as np
from scipy.stats import norm


class SAX:
    def __init__(self, word_size: int, alphabet_size: int, epsilon: float = 1e-6):
        """
        Initialize SAX transformer
        
        Args:
            word_size (int): Number of segments (PAA segments)
            alphabet_size (int): Number of symbols (alphabet size)
            epsilon (float): Small value to avoid division by zero
        """
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.epsilon = epsilon
        
        # Generate breakpoints for Gaussian distribution
        # These divide the area under N(0,1) into equiprobable regions
        self.breakpoints = self._generate_breakpoints()
    
    def _generate_breakpoints(self):
        """
        Generate breakpoints for the Gaussian distribution
        
        Returns:
            torch.Tensor: Breakpoints tensor of shape [alphabet_size-1]
        """
        beta = np.array([norm.ppf((i+1)/self.alphabet_size) 
                        for i in range(self.alphabet_size-1)])
        return torch.tensor(beta, dtype=torch.float32)
    
    def fit(self, data: torch.Tensor | np.ndarray) -> "SAX":
        """
        Fit the SAX transformer to the data (learn breakpoints)
        
        Args:
            data (torch.Tensor | np.ndarray): Input time series tensor of shape [batch_size, sequence_length] or 
                 [batch_size, sequence_length, dimension]
            
        Returns:
            SAX: The fitted transformer (self)
        """
        # Currently using pre-defined Gaussian breakpoints
        # This method is provided for sklearn-like API compatibility
        # Could be extended to learn breakpoints from empirical data distribution
        return self
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Z-normalize the time series
        
        Args:
            data (torch.Tensor): Input time series tensor of shape [batch_size, sequence_length * input_dim]
            
        Returns:
            torch.Tensor: Normalized time series tensor of same shape as input
        """
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True) + self.epsilon
        return (data - mean) / std
    
    def paa_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Perform Piecewise Aggregate Approximation
        
        Args:
            normalized_data (torch.Tensor): Z-normalized time series tensor [batch_size, sequence_length]       

        Returns:
            torch.Tensor: PAA representation tensor [batch_size, word_size]
        """
       
        batch_size, seq_length = normalized_data.shape
        
        # Reshape and average
        reshaped = normalized_data.view(batch_size, self.word_size, -1)
        return torch.mean(reshaped, dim=2)
    
    def discretize(self, paa_data: torch.Tensor) -> torch.Tensor:
        """
        Discretize PAA representation into symbols
        
        Args:
            paa_data (torch.Tensor): PAA representation tensor [batch_size, word_size]
            
        Returns:
            torch.Tensor: SAX symbolic representation tensor [batch_size, word_size] or
                      [batch_size, word_size, dimension]
        """
        # Compare PAA values to breakpoints to determine symbols
        # For each value, count how many breakpoints it's greater than
        symbolic_data = torch.zeros_like(paa_data, dtype=torch.long)
        
        for i in range(self.alphabet_size-1):
            symbolic_data += (paa_data > self.breakpoints[i]).long()
            
        return symbolic_data
    
    def transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Transform time series to SAX representation
        
        Args:
            data (torch.Tensor | np.ndarray): Input time series tensor of shape [batch_size, sequence_length] or
                 [batch_size, sequence_length, dimension]
            
        Returns:
            torch.Tensor: SAX representation tensor [batch_size, word_size]
        """
        batch_size = data.shape[0]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        normalized_data = self.normalize(data.reshape(batch_size, -1))
        paa_data = self.paa_transform(normalized_data)
        sax_data = self.discretize(paa_data)
        
        return sax_data
    
    def fit_transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Fit to data, then transform it
        
        Args:
            data (torch.Tensor | np.ndarray): Input time series tensor of shape [batch_size, sequence_length] or
                 [batch_size, sequence_length, dimension]
                 
        Returns:
            torch.Tensor: SAX representation tensor [batch_size, word_size]
        """
        return self.fit(data).transform(data)


if __name__ == "__main__":
    # Example usage
    # Create random time series data
    batch_size = 200_000
    seq_length = 10
    dim = 2
    data_0 = np.arange(20).reshape(1, 10, 2)
    data_1 = np.arange(20)[::-1].reshape(1, 10, 2)

    data = np.concatenate([data_0, data_1], axis=0)
    print(data.reshape(2, -1))
    data = torch.tensor(data, dtype=torch.float32)
    
    # Initialize SAX transformer
    # word_size = int(seq_length / 25) * dim
    alphabet_size = 32
    word_size = 10
    print(f"word_size: {word_size} alphabet_size: {alphabet_size}")
    sax = SAX(word_size, alphabet_size)
    
    # Transform data to SAX representation
    sax_representation = sax.transform(data)
    
    print(f"Original data shape: {data.shape}")
    print(f"SAX representation shape: {sax_representation.shape}")
    print(f"SAX representation:\n{sax_representation[[0]]}")
    print(f"SAX representation:\n{sax_representation[[1]]}")




