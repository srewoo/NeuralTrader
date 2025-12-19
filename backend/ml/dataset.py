
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

class StockDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[Dataset, MinMaxScaler]:
        """
        Prepare DataFrame for LSTM training.
        Returns PyTorch Dataset and the fitted scaler (for inverse transform).
        """
        # Feature Engineering: Use Close price for now (can expand to multi-feature)
        # Ensure we have data
        if df.empty:
            raise ValueError("Empty DataFrame provided")
            
        data = df[[target_col]].values
        
        # Fit Scaler
        scaled_data = self.scaler.fit_transform(data)
        
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - self.sequence_length):
            # Sequence: [t, t+1, ..., t+59]
            # Target: [t+60]
            seq = scaled_data[i : i + self.sequence_length]
            target = scaled_data[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(target)
            
        if not sequences:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1}")
            
        return StockDataset(np.array(sequences), np.array(targets)), self.scaler
    
    def prepare_inference_data(self, df: pd.DataFrame, target_col: str = 'Close') -> torch.Tensor:
        """
        Prepare the *last* sequence for predicting the *next* value.
        """
        data = df[[target_col]].values
        # Use existing scaler fit (warning: should be fitted on training data, 
        # but for online inference we often re-fit on recent history window)
        scaled_data = self.scaler.fit_transform(data) 
        
        last_sequence = scaled_data[-self.sequence_length:]
        if len(last_sequence) < self.sequence_length:
             raise ValueError("Insufficient data for inference")
             
        # Shape: (1, seq_len, features)
        return torch.FloatTensor(last_sequence).unsqueeze(0)
