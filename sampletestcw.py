#!/usr/bin/env python3
"""
Example script to load a generated training dataset and plot sample waveforms.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot sample waveforms from generated dataset.")
    parser.add_argument("pkl_file", type=str,
                        help="Path to the .pkl file (e.g., dataset/Training_Dataset_BASEBAND_CW_SOI_BASEBAND_CW_INTERFERENCE.pkl)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of random waveforms to plot")
    parser.add_argument("--plot_len", type=int, default=2000,
                        help="Number of time samples to plot in the time domain")
    parser.add_argument("--sample_rate", type=float, default=50e6,
                        help="Sample rate (Hz) for optional frequency-domain plots")
    args = parser.parse_args()

    # 1) Load the dataset
    if not os.path.exists(args.pkl_file):
        raise FileNotFoundError(f"Could not find dataset file: {args.pkl_file}")

    print(f"Loading dataset from: {args.pkl_file}")
    with open(args.pkl_file, "rb") as f:
        all_sig_mixture, all_sig1, all_bits1, meta_data = pickle.load(f)

    # 2) Basic info about shapes
    total_waveforms = all_sig_mixture.shape[0]
    sig_len = all_sig_mixture.shape[1]
    print(f"Dataset loaded.")
    print(f"  Mixtures shape: {all_sig_mixture.shape} (complex)")
    print(f"  Clean SOI shape: {all_sig1.shape} (complex)")
    print(f"  Bits shape: {all_bits1.shape}")
    print(f"  Metadata shape: {meta_data.shape}")
    print(f"Plotting {args.num_samples} random waveforms from total {total_waveforms}...\n")

    # 3) Randomly select waveforms to plot
    np.random.seed(0)  # fix seed for reproducibility (optional)
    indices_to_plot = np.random.choice(total_waveforms, size=args.num_samples, replace=False)

    # 4) Plot each selected waveform
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(indices_to_plot, start=1):
        # Extract waveforms
        mixture = all_sig_mixture[idx]   # shape (sig_len,) complex
        clean_soi = all_sig1[idx]       # shape (sig_len,) complex
        bits = all_bits1[idx]           # depends on your SOI type
        info_line = meta_data[idx]      # array of shape [5] (or however you stored it)

        # Print some metadata info
        # e.g. [gain, sinr, actual_sinr, soi_type, interference_type]
        print(f"--- Waveform index: {idx} ---")
        print(f"Metadata: {info_line}")
        print(f"Bits (first 10): {bits[:10] if len(bits)>10 else bits}")

        # Let's do a time-domain plot for the mixture (real part & imag part).
        # We'll limit ourselves to the first 'plot_len' samples.
        t_plot = np.arange(args.plot_len)
        mixture_real = np.real(mixture[:args.plot_len])
        mixture_imag = np.imag(mixture[:args.plot_len])

        plt.subplot(args.num_samples, 2, 2*(i-1)+1)
        plt.plot(t_plot, mixture_real, label='Real')
        plt.plot(t_plot, mixture_imag, label='Imag')
        plt.title(f"Mixture idx={idx} (Time Domain)")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.legend()

        # Optionally do a frequency-domain plot (magnitude spectrum)
        # We'll do a simple FFT or use matplotlib's built-in function
        plt.subplot(args.num_samples, 2, 2*(i-1)+2)
        freqs = np.fft.fftfreq(sig_len, d=1/args.sample_rate)  # freq axis
        spectrum = np.fft.fft(mixture)
        spectrum_mag = np.abs(spectrum)
        # We might just plot the first half for a single-sided amplitude spectrum
        half = sig_len // 2
        plt.plot(freqs*1e-6, 20*np.log10(spectrum_mag + 1e-12))
        plt.title(f"Mixture idx={idx} (Freq Domain)")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Mag (dB)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
