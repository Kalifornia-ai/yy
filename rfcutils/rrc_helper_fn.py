import numpy as np
import tensorflow as tf

# In newer Sionna versions (>=0.15)
from sionna.signal import RootRaisedCosineFilter

def get_psf(samples_per_symbol, span_in_symbols, beta):
    """
    Creates a Root Raised Cosine Filter with the given parameters.
    
    Args:
        samples_per_symbol (int): Oversampling factor
        span_in_symbols (int): Filter span in symbols
        beta (float): Roll-off factor

    Returns:
        rrcf (callable): A RootRaisedCosineFilter instance
    """
    # RootRaisedCosineFilter: (span_in_symbols, samples_per_symbol, beta)
    rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
    return rrcf

def matched_filter(sig, samples_per_symbol, span_in_symbols, beta):
    """
    Applies the RRC filter to the signal 'sig' with 'same' padding.
    
    Args:
        sig (tf.Tensor): Input complex baseband signal of shape [batch_size, time_length]
        samples_per_symbol (int): Oversampling factor
        span_in_symbols (int): Filter span in symbols
        beta (float): Roll-off factor

    Returns:
        x_mf (tf.Tensor): Filtered output with the same shape as 'sig'
    """
    rrcf = get_psf(samples_per_symbol, span_in_symbols, beta)
    x_mf = rrcf(sig, padding="same")
    return x_mf
