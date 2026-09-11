"""
Neural Network Architectures for MetroPT-3 Anomaly Detection.
"""
try:
    from src.models.dense_ae import DenseAutoencoder
    from src.models.lstm_ae import LSTMAutoencoder
    from src.models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
except ImportError:
    from models.dense_ae import DenseAutoencoder
    from models.lstm_ae import LSTMAutoencoder
    from models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder

__all__ = [
    "DenseAutoencoder",
    "LSTMAutoencoder",
    "AttentionDenoisingLSTMAutoencoder",
]
