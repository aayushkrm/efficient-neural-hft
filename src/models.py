import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- ACTIVATIONS & BLOCKS ---
class Mish(nn.Module):
    """
    Mish Activation: x * tanh(softplus(x)).
    Proved superior to ReLU for deep RNNs in low-signal regimes.
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D sequences.
    Acts as dynamic feature selection/gating.
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.fc(x)
        return x * y

# --- 1. THE WINNER: SE-Mish-DeepResGRU ---
class PreNormGRUCell(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.gru = nn.GRU(dim, dim, 1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        self.act = Mish()
    def forward(self, x, h):
        x_norm = self.ln(x)
        o, h_new = self.gru(x_norm, h)
        o = self.drop(self.act(o))
        return x + o, h_new # Residual Connection

class SEMishGRU(nn.Module):
    def __init__(self, input_size=32, hidden_size=240, layers=6):
        super().__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.se = SEBlock(hidden_size)
        self.layers = nn.ModuleList([PreNormGRUCell(hidden_size, 0.15) for _ in range(layers)])
        self.head = nn.Linear(hidden_size, 32)
        self.final_ln = nn.LayerNorm(hidden_size)
        self.layers_count = layers
        
    def forward(self, x, h_list=None):
        if h_list is None: h_list = [None] * self.layers_count
        x = self.embed(x)
        x = self.se(x) 
        new_h = []
        for i, layer in enumerate(self.layers):
            x, h = layer(x, h_list[i])
            new_h.append(h)
        x = self.final_ln(x)
        return self.head(x), new_h

# --- 2. THE CHALLENGER: Transformer-Encoder (Failed due to overfitting) ---
class TransformerModel(nn.Module):
    def __init__(self, input_size=32, hidden_size=256, layers=4):
        super().__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(hidden_size, 32)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.head(x[:, -1, :]) # Predict on last step

# --- FACTORY ---
def get_model(name, **kwargs):
    if name == "winner":
        return SEMishGRU(**kwargs)
    elif name == "transformer":
        return TransformerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")