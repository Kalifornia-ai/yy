import numpy as np
import tensorflow as tf

# (1) Instead of "sn.utils.BinarySource()":
from sionna.utils.misc import BinarySource

# (2) Constellation, Mapper, Demapper
# old: sn.phy.mapping.Constellation, sn.mapping.Mapper, sn.mapping.Demapper
from sionna.mapping import Constellation, Mapper, Demapper

# (3) AWGN channel
# old: sn.channel.AWGN
from sionna.channel import AWGN

# (4) ebnodb2no for converting Eb/N0 to noise power
# old: sn.utils.ebnodb2no
from sionna.utils import ebnodb2no

# (5) For upsampling/downsampling signals
# old: sn.signal.Upsampling, sn.signal.Downsampling
from sionna.signal import Upsampling, Downsampling

# Local custom utilities
from .rrc_helper_fn import get_psf, matched_filter

########################################
# Global Parameters
########################################
samples_per_symbol = 16
span_in_symbols = 8
beta = 0.5

# 16-QAM constellation
NUM_BITS_PER_SYMBOL = 4

# Instantiate the binary source
binary_source = BinarySource()

# Instantiate the QAM16 Constellation (trainable=False is typically fine)
constellation = Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=False)

# Mapper and Demapper
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = AWGN()

########################################
# Main Functions
########################################
def generate_qam16_signal(batch_size, num_symbols, ebno_db=None):
    """
    Generate a 16-QAM baseband signal, optionally adding AWGN at ebno_db.
    Returns (y, x, bits, constellation).
    """
    # bits: [batch_size, num_symbols * NUM_BITS_PER_SYMBOL]
    bits = binary_source([batch_size, num_symbols * NUM_BITS_PER_SYMBOL])
    return modulate_qam16_signal(bits, ebno_db)


def qam16_matched_filter_demod(sig, no=1e-4, soft_demod=False):
    """
    Apply matched filter, downsample, and demodulate (LLR -> bits).
    If soft_demod=True, returns LLR instead of hard bits.
    """
    # Matched filter
    x_mf = matched_filter(sig, samples_per_symbol, span_in_symbols, beta)
    num_symbols = sig.shape[-1] // samples_per_symbol

    # Downsample
    ds = Downsampling(samples_per_symbol, samples_per_symbol // 2, num_symbols)
    x_hat = ds(x_mf)

    # Scale
    x_hat = x_hat / tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))

    # Soft demapping
    llr = demapper([x_hat, no])  # shape: [batch_size, num_bits]

    # Return either soft or hard bits
    if soft_demod:
        return llr, x_hat
    else:
        bits_hat = tf.cast(llr > 0, tf.float32)
        return bits_hat, x_hat


def modulate_qam16_signal(info_bits, ebno_db=None):
    """
    Mapper -> Upsample -> (RRC filter) -> (AWGN optional).
    Returns (y, x, info_bits, constellation).
    """
    # 1) Map bits -> QAM symbols
    x = mapper(info_bits)

    # 2) Upsample
    us = Upsampling(samples_per_symbol)
    x_us = us(x)

    # 3) Time-align (pad) before filtering
    x_us = tf.pad(x_us,
                  tf.constant([[0, 0], [samples_per_symbol // 2, 0]]),
                  "CONSTANT")
    x_us = x_us[:, :-samples_per_symbol // 2]

    # 4) RRC filter (transmit pulse shaping)
    x_rrcf = matched_filter(x_us, samples_per_symbol, span_in_symbols, beta)

    if ebno_db is None:
        y = x_rrcf
    else:
        # Convert Eb/N0 (dB) -> noise power No
        no = ebnodb2no(ebno_db=ebno_db,
                       num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                       coderate=1.0)  # Uncoded
        # Pass through AWGN
        y = awgn_channel([x_rrcf, no])

    # Scale by sqrt(samples_per_symbol)
    y = y * tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))

    return y, x, info_bits, constellation
