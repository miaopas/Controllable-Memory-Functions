"""Model size parameter key registry.

This module contains the registry mapping model names to their size parameter keys
in configuration dictionaries. This is separated to avoid heavy dependencies.
"""

# Registry for model size parameter keys
# Add new models here as needed
MODEL_KEY_REGISTRY = {
    'LSTM': 'module.layers.LSTM.hidden_size',
    'S4': 'module.layers.Layer2.d_model',
    'TCN': 'module.layers.Layer2.channel',
    'Trans_hid_dim': 'module.layers.Layer1.hid_dim',
    'Trans_nhead': 'module.layers.Layer1.nhead',
    'Mamba': 'module.layers.Layer1.d_model',
    'Mixed': 'module.model_size'
}
