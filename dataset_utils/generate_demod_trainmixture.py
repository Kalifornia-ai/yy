import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
import random
import h5py
from tqdm import tqdm
import pickle
import argparse

import rfcutils
import tensorflow as tf

# Some simple helper lambdas
get_db = lambda p: 10*np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2, axis=-1)
get_sinr = lambda s, i: get_pow(s)/get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s,i))

sig_len = 40960         # number of samples in each waveform
default_n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)  # discrete SINR levels
seed_number = 0

###############################################################################
# 1) Get the appropriate SOI generation function
###############################################################################
def get_soi_generation_fn(soi_sig_type):
    """
    Returns a (generate_soi, demod_soi) tuple for the specified soi_sig_type.
    generate_soi(batch_size, sig_len) -> (wave, _, bits, _)
    demod_soi(...) -> demodulator function
    """
    # Case A: Traditional waveforms from rfcutils
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len//16)
        demod_soi = rfcutils.qpsk_matched_filter_demod

    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len//16)
        demod_soi = rfcutils.qam16_matched_filter_demod

    elif soi_sig_type == 'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len//4)
        demod_soi = rfcutils.qpsk2_matched_filter_demod

    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len//80)
        _,_,_,RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len//80)
        demod_soi = lambda s: rfcutils.ofdm_demod(s, RES_GRID)

    # Case B: Passband CW for SOI
    elif soi_sig_type == 'BASEBAND_CW_SOI':
        # We'll generate a single frequency at 2.05 GHz, with e.g. 5 GHz sampling
        def generate_passband_soi(batch_size, _):
            sample_rate = 50e6
            freq_hz     = 0
            # Reuse amplitude=1.0, phase=0.0 (change if needed)
            soi_wave = rfcutils.cw_helper_fn.generate_cw_signal(
                batch_size=batch_size,
                num_samples=sig_len,
                freq_hz=freq_hz,
                sample_rate=sample_rate,
                amplitude=1.0,
                phase=0.0
            )
            # For CW, we have no real "bits". Let's return placeholders
            # to match the 4-return structure. We'll just do zeros for bits
            bits_dummy = np.zeros((batch_size, 1), dtype=np.float32)
            # Return: (wave, _, bits, _)
            return (soi_wave, None, bits_dummy, None)

        generate_soi = generate_passband_soi
        demod_soi = None  # No demod needed for pure CW

    else:
        raise Exception(f"SOI Type '{soi_sig_type}' not recognized")

    return generate_soi, demod_soi

###############################################################################
# 2) The main function to combine SOI + Interference across multiple SINRs
###############################################################################
def generate_demod_testmixture(soi_type, interference_sig_type, n_per_batch=default_n_per_batch):

    # 2.1) Get the function for generating the SOI
    generate_soi, demod_soi = get_soi_generation_fn(soi_type)

    ############################################################################
    # 2.2) If the interference type is "PASSBAND_CW_INTERFERENCE",
    # we will skip reading from .h5 and just generate in code.
    # Otherwise, read from HDF5 as usual.
    ############################################################################
    if interference_sig_type == 'BASEBAND_CW_INTERFERENCE':
        # We'll generate a random freq in [2.0372 GHz, 2.0628 GHz] for each wave
        sample_rate = 50e6
        freq_min = -12.8e6
        freq_max = 12.8e6

        def generate_interference(batch_size):
            # For each item in batch, pick a random freq
            freqs = np.random.uniform(low=freq_min, high=freq_max, size=batch_size)
            # We'll create an empty array first, shape [batch_size, sig_len]
            arr = np.zeros((batch_size, sig_len), dtype=np.float32)
            # Convert to TF and fill each row
            arr_tf = tf.convert_to_tensor(arr)
            arr_tf = tf.identity(arr_tf)  # no-op to keep shape

            result_list = []
            for b_idx in range(batch_size):
                # single freq
                freq_b = freqs[b_idx]
                cw_b = rfcutils.cw_helper_fn.generate_cw_signal(
                    batch_size=1,
                    num_samples=sig_len,
                    freq_hz=freq_b,
                    sample_rate=sample_rate,
                    amplitude=1.0,    # adjust as needed
                    phase=0.0
                )
                result_list.append(cw_b[0])  # shape [sig_len]
            
            # Combine
            result = tf.stack(result_list, axis=0) # shape [batch_size, sig_len]
            return result.numpy()

        # define a function so we can re-use in the loop
        read_interference_fn = generate_interference

    else:
        # The default path: read from HDF5
        h5_path = os.path.join('dataset', 'interferenceset_frame', f'{interference_sig_type}_raw_data.h5')
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"No .h5 file found at {h5_path} for interference {interference_sig_type}")

        data_h5file = h5py.File(h5_path, 'r')
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")
        data_h5file.close()

        def read_interference_fn(batch_size):
            # random pick from sig_data
            idxs = np.random.randint(sig_data.shape[0], size=batch_size)
            sub = sig_data[idxs, :]  # shape [batch_size, full_length]
            # now pick random start indices
            rand_start = np.random.randint(sub.shape[1] - sig_len, size=batch_size)
            # gather
            # We'll do this in TF for consistency:
            sub_tf = tf.constant(sub)
            # build gather indices
            offset = tf.range(sig_len, dtype=tf.int32)
            offset = offset[tf.newaxis, :] # shape [1, sig_len]
            rand_start_tf = tf.constant(rand_start, dtype=tf.int32)[:, tf.newaxis] # [batch_size, 1]
            gather_idx = rand_start_tf + offset  # [batch_size, sig_len]
            # gather
            sub_extracted = tf.gather(sub_tf, tf.range(batch_size))  # same shape
            # Instead of tf.experimental.numpy.take_along_axis, we can do tf.gather_nd
            # but let's keep it consistent with your approach:
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
            out = tf.experimental.numpy.take_along_axis(sub_extracted, gather_idx, axis=1)
            return out.numpy()

    ############################################################################
    # 2.3) Set seeds for reproducibility
    ############################################################################
    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.random.set_seed(seed_number)

    # Arrays to store final results
    all_sig_mixture, all_sig1, all_bits1, meta_data = [], [], [], []

    # 2.4) Loop over each SINR in all_sinr and build the mixture
    for idx, sinr in tqdm(enumerate(all_sinr), total=len(all_sinr)):
        # (A) Generate the SOI
        sig1, _, bits1, _ = generate_soi(n_per_batch, sig_len)  # shape [n_per_batch, sig_len]
        # (B) Generate (or read) the interference
        sig2 = read_interference_fn(n_per_batch)  # shape [n_per_batch, sig_len]

        # Convert to TF for the next steps
        sig_target = tf.constant(sig1, dtype=tf.complex64)
        sig_interf = tf.constant(sig2, dtype=tf.complex64)
        scalar_gain = np.sqrt(10**(-sinr/10)).astype(np.float32)  # shape ()
        rand_gain = np.full((n_per_batch,), scalar_gain, dtype=np.float32)  # shape [n_per_batch]

        # (C) Compute interference scaling for desired sinr
        #rand_gain = np.sqrt(10**(-sinr/10)).astype(np.float32)  # shape [n_per_batch]
        rand_gain = tf.constant(rand_gain, shape=(n_per_batch,1), dtype=tf.float32)
        rand_phase = tf.random.uniform(shape=[n_per_batch, 1])  # uniform in [0,1)
        rand_gain_c = tf.complex(rand_gain, tf.zeros_like(rand_gain))  # shape [n_per_batch]
        rand_phase_c = tf.complex(rand_phase, tf.zeros_like(rand_phase))
        coeff = rand_gain_c * tf.math.exp(1j*2*np.pi*rand_phase_c)  # shape [n_per_batch, 1]

        # broadcast coeff across time dimension
        coeff = tf.tile(coeff, [1, sig_len])  # shape [n_per_batch, sig_len]

        # (D) Mixture
        print("sig1 shape:", sig_target.shape)
        print("sig2 shape:", sig_interf.shape)
        print("coef shape", coeff.shape)

        sig_mixture = sig_target + sig_interf * coeff

        # (E) Accumulate
        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig_target)
        all_bits1.append(bits1)

        # (F) Compute actual SINR
        actual_sinr = get_sinr_db(sig_target, sig_interf * coeff)
        # Build metadata row, shape [5, n_per_batch]
        # e.g. [gain, sinr, actual_sinr, soi_type, interference_type]
        soi_str_col = np.array([soi_type]*n_per_batch)
        int_str_col = np.array([interference_sig_type]*n_per_batch)
        rand_gain_for_meta = rand_gain.numpy().reshape(-1)
        meta_row = np.vstack((
            rand_gain_for_meta,    # real scale factor
            np.full(n_per_batch, sinr, dtype=np.float32),
            actual_sinr,  # shape [n_per_batch]
            soi_str_col,
            int_str_col
        ))
        meta_data.append(meta_row)

    ############################################################################
    # 2.5) Concatenate everything & save
    ############################################################################
    with tf.device('CPU'):
        all_sig_mixture = tf.concat(all_sig_mixture, axis=0).numpy()  # shape [#sinr*n_per_batch, sig_len]
        all_sig1        = tf.concat(all_sig1, axis=0).numpy()
        all_bits1       = np.concatenate(all_bits1, axis=0)  # pure numpy bits

    # meta_data is a list of [5, n_per_batch], we want to stack along axis=1 => shape [5, #sinr*n_per_batch]
    meta_data = np.concatenate(meta_data, axis=1).T  # => shape [#sinr*n_per_batch, 5]

    # final pickle
    out_path = os.path.join('../dataset', f'Training_Dataset_{soi_type}_{interference_sig_type}.pkl')
    pickle.dump((all_sig_mixture, all_sig1, all_bits1, meta_data), open(out_path, 'wb'), protocol=4)
    print(f"Saved training dataset -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-b', '--n_per_batch', default=100, type=int, help='Number of waveforms per SINR level')
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--soi_sig_type', required=True, help='SOI type: QPSK, QAM16, OFDMQPSK, PASSBAND_CW_SOI, etc.')
    parser.add_argument('--interference_sig_type', required=True, help='Interference type: e.g. from HDF5 or PASSBAND_CW_INTERFERENCE')

    args = parser.parse_args()

    seed_number = args.random_seed

    generate_demod_testmixture(
        soi_type=args.soi_sig_type,
        interference_sig_type=args.interference_sig_type,
        n_per_batch=args.n_per_batch
    )

