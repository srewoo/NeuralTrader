
import torch
import torch.nn as nn

class PricePredictor(nn.Module):
    """
    LSTM-based model for time-series forecasting.
    Input: (batch, seq_len, features)
    Output: (batch, prediction_horizon)
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(PricePredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out shape: (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]
        
        # Decode
        out = self.fc(out)
        return out
