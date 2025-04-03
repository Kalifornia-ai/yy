import numpy as np
import tensorflow as tf

# Sionna submodules for version ≥0.15
from sionna.utils.misc import BinarySource
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.channel import AWGN
from sionna.signal import Upsampling, Downsampling
from sionna.utils import ebnodb2no

from .rrc_helper_fn import get_psf, matched_filter

###########################################
# Global Parameters
###########################################
samples_per_symbol = 4
span_in_symbols   = 8
beta             = 0.5

# QPSK => 2 bits per symbol
NUM_BITS_PER_SYMBOL = 2

###########################################
# Blocks
###########################################
# (1) Binary source
binary_source = BinarySource()

# (2) 4-QAM constellation for QPSK (2 bits/symbol)
constellation = Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=False)

# (3) Mapper / Demapper
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# (4) AWGN channel
awgn_channel = AWGN()

###########################################
# Main Functions
###########################################
def generate_qpsk2_signal(batch_size, num_symbols, ebno_db=None):
    """
    Generate QPSK signals with 'num_symbols' 
    for a batch of 'batch_size'.
    Optionally pass through AWGN (ebno_db).
    """
    bits = binary_source([batch_size, num_symbols * NUM_BITS_PER_SYMBOL])
    return modulate_qpsk2_signal(bits, ebno_db)

def qpsk2_matched_filter_demod(sig, no=1e-4, soft_demod=False):
    """
    Matched filter + downsample + demod -> LLR or hard bits.
    """
    # 1) RRC matched filter
    x_mf = matched_filter(sig, samples_per_symbol, span_in_symbols, beta)

    # 2) Downsample
    num_symbols = sig.shape[-1] // samples_per_symbol
    ds = Downsampling(samples_per_symbol, samples_per_symbol//2, num_symbols)
    x_hat = ds(x_mf)

    # 3) Scale
    x_hat = x_hat / tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))

    # 4) Soft demapping -> LLR
    llr = demapper([x_hat, no])

    if soft_demod:
        return llr, x_hat
    else:
        return tf.cast(llr > 0, tf.float32), x_hat

def modulate_qpsk2_signal(info_bits, ebno_db=None):
    """
    Bits -> QPSK mapping -> Upsampling -> RRC filter -> AWGN (optional).
    Returns (y, x, info_bits, constellation).
    """
    # 1) QPSK map
    x = mapper(info_bits)

    # 2) Upsampling
    us = Upsampling(samples_per_symbol)
    x_us = us(x)

    # 3) Time-align
    x_us = tf.pad(
        x_us,
        tf.constant([[0, 0], [samples_per_symbol // 2, 0]]),
        "CONSTANT"
    )
    x_us = x_us[:, :-samples_per_symbol // 2]

    # 4) Transmit RRC filter
    x_rrcf = matched_filter(x_us, samples_per_symbol, span_in_symbols, beta)

    if ebno_db is None:
        y = x_rrcf
    else:
        # Convert Eb/N0 -> noise power
        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=1.0
        )
        y = awgn_channel([x_rrcf, no])

    # Final scaling factor
    y = y * tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))

    return y, x, info_bits, constellation
