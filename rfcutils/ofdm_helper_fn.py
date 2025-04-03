import numpy as np
import tensorflow as tf

#####################################
# New Sionna Imports
#####################################
# Depending on your Sionna version, these classes might be 
# in slightly different submodules. Adjust as needed.

# BinarySource was previously sn.utils.BinarySource.
from sionna.utils.misc import BinarySource

# Constellation, Mapper, Demapper were previously in sn.mapping.
from sionna.mapping import Constellation, Mapper, Demapper

# StreamManagement was previously in sn.mimo.
from sionna.mimo import StreamManagement

# AWGN channel was previously sn.channel.AWGN().
from sionna.channel import AWGN

# OFDM classes that were previously sn.ofdm.*.
from sionna.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    OFDMModulator,
    OFDMDemodulator
)

# ebnodb2no was previously sn.utils.ebnodb2no.
from sionna.utils import ebnodb2no

#####################################
# Global Parameters
#####################################
NFFT = 64
CP_LEN = 16
OFDM_LEN = NFFT + CP_LEN
CODERATE = 1
n_streams_per_tx = 1

#####################################
# Instantiate Core Components
#####################################
# 1) Binary source to generate uniform i.i.d. bits
binary_source = BinarySource()

# 2) 4-QAM constellation (i.e., QAM with 2 bits/symbol)
NUM_BITS_PER_SYMBOL = 2
constellation = Constellation("qam",
                              NUM_BITS_PER_SYMBOL,
                              trainable=False)

# 3) Stream management 
# (MIMO indexing; for single TX, single stream, typically a 1x1 matrix)
stream_manager = StreamManagement(np.array([[1]]), 1)

# 4) Mapper and demapper
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# 5) AWGN channel
awgn_channel = AWGN()

#####################################
# Helper: Resource Grid
#####################################
def get_resource_grid(num_ofdm_symbols):
    """
    Create a Sionna ResourceGrid object for the desired OFDM parameters.
    """
    rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                      fft_size=NFFT,
                      subcarrier_spacing=20e6/NFFT,
                      num_tx=1,
                      num_streams_per_tx=n_streams_per_tx,
                      num_guard_carriers=(4, 3),
                      dc_null=True,
                      cyclic_prefix_length=CP_LEN,
                      pilot_pattern=None,
                      pilot_ofdm_symbol_indices=[])
    return rg

#####################################
# Main OFDM Generation + Transmission
#####################################
def generate_ofdm_signal(batch_size, num_ofdm_symbols, ebno_db=None):
    """
    Generate OFDM signal, optionally pass it through AWGN channel at ebno_db.
    Returns (rx_signal, transmitted_symbols, transmitted_bits, resource_grid).
    """
    RESOURCE_GRID = get_resource_grid(num_ofdm_symbols)

    # Number of coded bits in resource grid
    n = int(RESOURCE_GRID.num_data_symbols * NUM_BITS_PER_SYMBOL)
    # Number of information bits in resource grid
    k = int(n * CODERATE)  # If CODERATE=1, then k=n (uncoded)

    # Random bits: shape [batch_size, 1, n_streams_per_tx, k]
    with tf.device('/CPU:0'):
        bits = binary_source([batch_size, 1, n_streams_per_tx, k])

    return modulate_ofdm_signal(bits, RESOURCE_GRID, ebno_db)


def modulate_ofdm_signal(info_bits, RESOURCE_GRID, ebno_db=None):
    """
    Map bits -> QAM -> OFDM -> optional AWGN.
    Returns (y, x_rg, info_bits, RESOURCE_GRID).
    """
    # Placeholder if you want to do coding. Right now, uncoded:
    # codewords = encoder(info_bits)
    codewords = info_bits

    rg_mapper = ResourceGridMapper(RESOURCE_GRID)
    ofdm_mod = OFDMModulator(RESOURCE_GRID.cyclic_prefix_length)

    # 1) QAM mapping
    x = mapper(codewords)
    # 2) Map symbols onto OFDM resource grid
    x_rg = rg_mapper(x)
    # 3) Modulate to time domain
    x_ofdm = ofdm_mod(x_rg)

    if ebno_db is None:
        # No noise added
        y = x_ofdm
    else:
        # Convert Eb/N0(dB) to noise power
        no = ebnodb2no(ebno_db=ebno_db,
                       num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                       coderate=CODERATE,
                       resource_grid=RESOURCE_GRID)
        y = awgn_channel([x_ofdm, no])

    # Squeeze axes corresponding to (num_tx, num_streams_per_tx) = 1
    y = tf.squeeze(y, axis=[1, 2])
    x_rg = tf.squeeze(x_rg, axis=[1, 2])
    info_bits = tf.squeeze(info_bits, axis=[1, 2])

    return y, x_rg, info_bits, RESOURCE_GRID

#####################################
# OFDM Demod + Bit Decisions
#####################################
def ofdm_demod(sig, RESOURCE_GRID, no=1e-4):
    """
    Demodulate an OFDM signal, compute LLR, and
    return hard-decoded bits + the freq-domain symbols.
    """
    rg_demapper = ResourceGridDemapper(RESOURCE_GRID, stream_manager)
    ofdm_demod_block = OFDMDemodulator(NFFT, 0, CP_LEN)

    # 1) OFDM demod (time->freq)
    x_ofdm_demod = ofdm_demod_block(sig)
    # x_ofdm_demod shape: [batch, num_tx=1, num_streams=1, num_ofdm_symbols, NFFT]
    x_demod = rg_demapper(tf.reshape(x_ofdm_demod,
                                     (sig.shape[0], 1, 1, -1, NFFT)))
    # 2) Soft-demapping to LLR
    llr = demapper([x_demod, no])
    # Hard decisions
    bits_hat = tf.cast(llr > 0, tf.float32)

    # Squeeze axis=[1,2] → shape [batch, <num_bits>]
    bits_hat = tf.squeeze(bits_hat, axis=[1, 2])
    return bits_hat, x_ofdm_demod


