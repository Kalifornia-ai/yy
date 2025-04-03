import numpy as np
import tensorflow as tf

# Sionna Submodules (adjust paths if something is missing in your version)
from sionna.utils.misc import BinarySource
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.channel import AWGN
from sionna.signal import Upsampling, Downsampling
from sionna.utils import ebnodb2no

from .rrc_helper_fn import get_psf, matched_filter

###########################################
# Global Parameters
###########################################
samples_per_symbol = 16
span_in_symbols = 8
beta = 0.5

# 4-QAM (QPSK) uses 2 bits per symbol
NUM_BITS_PER_SYMBOL = 2

###########################################
# Instantiate Core Blocks
###########################################
# Binary source to generate uniform i.i.d bits
binary_source = BinarySource()

# Constellation: QAM with 2 bits/symbol = QPSK
# The dtype can be tf.complex128 or tf.complex64; just ensure consistency
constellation = Constellation(
    "qam",
    NUM_BITS_PER_SYMBOL,
    trainable=False
)

# Mapper and demapper
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = AWGN()

###########################################
# Main Functions
###########################################
def generate_qpsk_signal(batch_size, num_symbols, ebno_db=None):
    """
    Generate a QPSK signal (num_symbols) for a batch of size batch_size.
    Optionally pass it through AWGN at Eb/N0 = ebno_db.
    """
    # bits shape: [batch_size, num_symbols * NUM_BITS_PER_SYMBOL]
    with tf.device('/CPU:0'):
        bits = binary_source([batch_size, num_symbols * NUM_BITS_PER_SYMBOL])
    return modulate_qpsk_signal(bits, ebno_db)

def qpsk_matched_filter_demod(sig, no=1e-4, soft_demod=False):
    """
    Matched-filter + downsample + demodulate the QPSK signal.
    Returns (llr or hard bits, x_hat).
    """
    # 1) Apply matched filter
    x_mf = matched_filter(sig, samples_per_symbol, span_in_symbols, beta)

    # 2) Downsample
    num_symbols = sig.shape[-1] // samples_per_symbol
    ds = Downsampling(samples_per_symbol, samples_per_symbol // 2, num_symbols)
    x_hat = ds(x_mf)

    # 3) Scale
    x_hat = x_hat / tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))

    # 4) Soft demapping -> LLR
    llr = demapper([x_hat, no])

    if soft_demod:
        return llr, x_hat
    else:
        # Hard bits
        bits_hat = tf.cast(llr > 0, tf.float32)
        return bits_hat, x_hat

def modulate_qpsk_signal(info_bits, ebno_db=None):
    """
    Mapper -> upsampling -> RRC filter -> optional AWGN -> scaling.
    Returns (y, x, info_bits, constellation).
    """
    # 1) Map bits -> QPSK symbols
    x = mapper(info_bits)

    # 2) Upsampling
    us = Upsampling(samples_per_symbol)
    x_us = us(x)

    # 3) Pad for proper alignment, then remove tail
    x_us = tf.pad(x_us,
                  tf.constant([[0, 0], [samples_per_symbol // 2, 0]]),
                  "CONSTANT")
    x_us = x_us[:, :-samples_per_symbol // 2]

    # 4) RRC filter
    x_rrcf = matched_filter(x_us, samples_per_symbol, span_in_symbols, beta)

    if ebno_db is None:
        y = x_rrcf
    else:
        # Convert Eb/N0 dB to noise power (No)
        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=1.0 # Uncoded
        )
        y = awgn_channel([x_rrcf, no])

    # Final scaling by sqrt(samples_per_symbol)
    y = y * tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))

    return y, x, info_bits, constellation
