import numpy as np
import tensorflow as tf

# For optional AWGN channel, we can reuse Sionna’s AWGN operator
try:
    from sionna.channel import AWGN
    from sionna.utils import ebnodb2no
    AWGN_AVAILABLE = True
except ImportError:
    AWGN_AVAILABLE = False

##################################################
# Global Parameters
##################################################
if AWGN_AVAILABLE:
    awgn_channel = AWGN()  # Reusable AWGN layer from Sionna
else:
    awgn_channel = None

##################################################
# Main Functions
##################################################

def generate_cw_signal(batch_size,
                       num_samples,
                       freq_hz,
                       sample_rate,
                       amplitude=1.0,
                       phase=0.0,
                       ebno_db=None,
                       bits_per_symbol=1,
                       coderate=1.0):
    """
    Generate a continuous-wave (CW) signal, optionally adding AWGN at a given Eb/N0 (dB).

    Args:
    -----
      batch_size    : int
                      Number of CW waveforms to create at once (for batch processing).
      num_samples   : int
                      Number of time-domain samples in each waveform.
      freq_hz       : float
                      Desired CW frequency in Hz (baseband representation assumed).
      sample_rate   : float
                      Sampling rate in Hz.
      amplitude     : float
                      Amplitude of the continuous wave (default: 1.0).
      phase         : float
                      Initial phase in radians (default: 0.0).
      ebno_db       : float or None
                      If not None, add AWGN based on this Eb/N0 in dB.
      bits_per_symbol : int
                      Needed only if using ebno_db to convert to noise power
                      (default=1 means treat each sample as “1 bit/symbol”).
      coderate      : float
                      Code rate used in ebnodb2no() if desired (default=1.0).

    Returns:
    --------
      y : tf.Tensor of shape [batch_size, num_samples], dtype=complex64
          The generated CW signal (with optional noise).
    """
    # 1) Create time vector [num_samples], shape -> (num_samples,)
    t = tf.range(num_samples, dtype=tf.float32) / sample_rate  # shape [num_samples]

    # 2) Build the angle as a real TF tensor
    #    angle(t) = 2π * freq_hz * t + phase
    angle = 2.0 * np.pi * freq_hz * t + phase  # shape [num_samples], float32

    # 3) Convert to a complex exponential:
    #    e^{j * angle}
    #    We'll use tf.complex(0., 1.) to represent j
    j_const = tf.complex(0.0, 1.0)  # shape ()
    exp_signal = tf.exp(j_const * tf.cast(angle, tf.complex64))  # shape [num_samples], complex64

    # 4) Scale by amplitude
    single_cw = tf.cast(amplitude, tf.complex64) * exp_signal  # shape [num_samples]

    # 5) Broadcast to batch dimension -> shape [batch_size, num_samples]
    single_cw = single_cw[tf.newaxis, :]  # [1, num_samples]
    x = tf.tile(single_cw, [batch_size, 1])  # [batch_size, num_samples]

    # 6) If no Eb/N0 specified or AWGN not available, return the clean CW
    if (ebno_db is None) or (not AWGN_AVAILABLE):
        return x

    # 7) Otherwise, add AWGN based on Eb/N0 -> noise power 'no'
    no = ebnodb2no(ebno_db=ebno_db,
                   num_bits_per_symbol=bits_per_symbol,
                   coderate=coderate)
    # Use the AWGN layer:
    y = awgn_channel([x, no])  # shape [batch_size, num_samples]

    return y


# def test_cw_script():
#     """
#     Simple test/demo of CW generation with or without AWGN.
#     """
#     # Example parameters
#     batch_size   = 4
#     num_samples  = 1024
#     freq_hz      = 1e3      # 1 kHz tone
#     sample_rate  = 16e3     # 16 kHz
#     amplitude    = 1.0
#     phase        = 0.0
#     ebno_db      = 10.0     # 10 dB Eb/N0, for demonstration

#     # Generate a batch of pure CW signals
#     cw_clean = generate_cw_signal(batch_size=batch_size,
#                                   num_samples=num_samples,
#                                   freq_hz=freq_hz,
#                                   sample_rate=sample_rate,
#                                   amplitude=amplitude,
#                                   phase=phase,
#                                   ebno_db=None)  # No AWGN
#     print("cw_clean shape:", cw_clean.shape, "dtype:", cw_clean.dtype)

#     # Generate a batch of CW signals with AWGN (if Sionna is available)
#     if AWGN_AVAILABLE:
#         cw_noisy = generate_cw_signal(batch_size=batch_size,
#                                       num_samples=num_samples,
#                                       freq_hz=freq_hz,
#                                       sample_rate=sample_rate,
#                                       amplitude=amplitude,
#                                       phase=phase,
#                                       ebno_db=ebno_db,
#                                       bits_per_symbol=1,
#                                       coderate=1.0)
#         print("cw_noisy shape:", cw_noisy.shape, "dtype:", cw_noisy.dtype)

#         # Example usage: measure average power
#         power_clean = tf.reduce_mean(tf.abs(cw_clean)**2, axis=1)  # per waveform
#         power_noisy = tf.reduce_mean(tf.abs(cw_noisy)**2, axis=1)  # per waveform
#         print("Avg clean power:", tf.reduce_mean(power_clean).numpy())
#         print("Avg noisy  power:", tf.reduce_mean(power_noisy).numpy())
#     else:
#         print("Sionna not installed; skipping noisy CW generation.")


# if __name__ == "__main__":
#     test_cw_script()
