#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

###############################################
# 1) Simple Dataset for the Test-Subset PKL
###############################################
class TestSubsetDataset:
    """
    Loads a dictionary from the test_subset PKL that includes:
      {
        "mixture": shape (N, L, 2),
        "so1":     shape (N, L, 2),
        "so2":     shape (N, L, 2) or None,
        "meta":    shape (N, M)  (optional columns)
      }
    We'll pick items to plot or run inference.

    If meta is missing or None, we handle that gracefully.
    """
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            data_dict = pickle.load(f)

        self.x_mixture = data_dict["mixture"]  # shape (N, L, 2)
        self.x_so1     = data_dict["so1"]      # shape (N, L, 2)
        self.x_so2     = data_dict.get("so2", None)
        self.meta      = data_dict.get("meta", None)

        if self.x_so2 is None:
            # If no so2 was stored, create zeros
            self.x_so2 = np.zeros_like(self.x_so1)

        self.N = self.x_mixture.shape[0]

        if self.meta is not None and len(self.meta) != self.N:
            raise ValueError("meta_data length does not match mixture length!")

    def __len__(self):
        return self.N

    def get_item(self, idx):
        """
        Returns (mix_ri, so1_ri, so2_ri, meta_row).
        each wave shape => (L,2). meta_row => (M,) or None.
        """
        mix_ri = self.x_mixture[idx]
        so1_ri = self.x_so1[idx]
        so2_ri = self.x_so2[idx]
        meta_row = self.meta[idx] if self.meta is not None else None
        return (mix_ri, so1_ri, so2_ri, meta_row)

###############################################
# 2) Plotting utilities
###############################################
def plot_time_and_freq(mix_ri, so1_ri, est_ri, so2_ri=None,
                       plot_path="plot.png", title_prefix=""):
    """
    Plots time-domain and frequency-domain magnitude for mixture, so1, estimate, and optional so2.
    Saves to 'plot_path', producing two PNGs (time/freq).
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
    fig.suptitle(title_prefix)

    t_axis = np.arange(len(mix_ri))

    # Time-domain, row=0 => mixture
    axs[0,0].plot(t_axis, mix_ri[:,0], label="Mix Real", color='b')
    axs[0,0].set_title("Mixture Real")
    axs[0,1].plot(t_axis, mix_ri[:,1], label="Mix Imag", color='r')
    axs[0,1].set_title("Mixture Imag")

    # Time-domain, row=1 => so1 vs estimate
    axs[1,0].plot(t_axis, so1_ri[:,0], label="SOI Real", color='b')
    axs[1,0].plot(t_axis, est_ri[:,0], label="Est Real", linestyle='--', color='g')
    axs[1,0].set_title("SOI vs. Est Real")
    axs[1,0].legend()

    axs[1,1].plot(t_axis, so1_ri[:,1], label="SOI Imag", color='r')
    axs[1,1].plot(t_axis, est_ri[:,1], label="Est Imag", linestyle='--', color='g')
    axs[1,1].set_title("SOI vs. Est Imag")
    axs[1,1].legend()

    plt.tight_layout()
    plt.savefig(plot_path.replace(".png","_time.png"), dpi=150)
    plt.close()

    # Frequency-domain
    def get_mag_spectrum(sig_ri):
        cplx = sig_ri[:,0] + 1j*sig_ri[:,1]
        spec = np.fft.fftshift(np.fft.fft(cplx))
        mag  = np.abs(spec)
        return mag

    mix_mag = get_mag_spectrum(mix_ri)
    so1_mag = get_mag_spectrum(so1_ri)
    est_mag = get_mag_spectrum(est_ri)
    so2_mag = get_mag_spectrum(so2_ri) if (so2_ri is not None) else None
    freq_bins = np.arange(len(mix_mag)) - (len(mix_mag)//2)

    plt.figure(figsize=(10,6))
    plt.plot(freq_bins, mix_mag, label="Mixture", color='b')
    plt.plot(freq_bins, so1_mag, label="SOI", color='r')
    plt.plot(freq_bins, est_mag, label="Estimate", color='g', linestyle='--')
    if so2_mag is not None:
        plt.plot(freq_bins, so2_mag, label="Interf", color='k', linestyle=':')
    plt.title(title_prefix + " (FFT Magnitude)")
    plt.xlabel("FFT Bin (shifted)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path.replace(".png","_freq.png"), dpi=150)
    plt.close()

###############################################
# 3) Model definition (must match training)
###############################################
import torch
import torch.nn as nn

class SELayer1D(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        squeezed = self.avg_pool(x).view(b, c)
        excitation = self.fc(squeezed).view(b, c, 1)
        return x * excitation

class LSTMSeperatorSingle(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_sources=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_sources = num_sources
        self.conv_in = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.lstm_layers = nn.ModuleList()
        self.se_blocks   = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_size if i == 0 else hidden_size*2
            self.lstm_layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True, bidirectional=True))
            self.se_blocks.append(SELayer1D(hidden_size*2))

        self.conv_out = nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2 * num_sources),
            nn.Sigmoid()
        )
        self.decoder = nn.Conv1d(hidden_size*2, input_size, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.conv_in(x)     
        x = x.permute(0, 2, 1)  

        lstm_out = x
        for lstm_layer, se_block in zip(self.lstm_layers, self.se_blocks):
            lstm_out, _ = lstm_layer(lstm_out)
            lstm_out = lstm_out.permute(0, 2, 1)
            lstm_out = se_block(lstm_out)
            lstm_out = lstm_out.permute(0, 2, 1)

        coding_feat = lstm_out.permute(0, 2, 1)
        coding_feat = self.conv_out(coding_feat)
        coding_feat = coding_feat.permute(0, 2, 1)
        coding_feat = self.layer_norm(coding_feat)

        b, seq_len, feat_dim = coding_feat.shape
        masks = self.mask_generator(coding_feat)
        masks = masks.view(b, seq_len, feat_dim, self.num_sources)

        coding_feat_expanded = coding_feat.unsqueeze(-1)
        masked_features = coding_feat_expanded * masks

        out_sources = []
        src_feat = masked_features[..., 0]
        src_feat = src_feat.permute(0, 2, 1)
        decoded = self.decoder(src_feat) 
        decoded = decoded.permute(0, 2, 1)
        out_sources.append(decoded)

        return torch.stack(out_sources, dim=1)

###############################################
# 4) Main Script
###############################################
def main():
    parser = argparse.ArgumentParser(
        description="Automatically pick in-band and out-of-band samples from test data (with meta) and plot."
    )
    parser.add_argument("--test_pkl", type=str, required=True,
                        help="Path to the test_subset PKL (with 'meta' that has band_label).")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to best_model.pth checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./test_plots",
                        help="Where to save plots.")
    parser.add_argument("--band_col_idx", type=int, default=5,
                        help="Index in meta row that stores the band label (1=inband,0=outband).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load test dataset
    ds = TestSubsetDataset(args.test_pkl)
    print(f"Loaded test subset with {len(ds)} items: {args.test_pkl}")
    if ds.meta is None:
        print("ERROR: The test PKL does not contain 'meta'. Cannot pick in/out-of-band. Exiting.")
        return

    # 2) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSeperatorSingle(num_sources=1, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    inband_sample = None
    outband_sample = None

    # 3) Find first in-band (band_label=1) and first out-of-band (0)
    for i in range(len(ds)):
        mix_ri, so1_ri, so2_ri, meta_row = ds.get_item(i)
        band_label = meta_row[args.band_col_idx]
        if band_label == 1.0 and (inband_sample is None):
            inband_sample = (i, mix_ri, so1_ri, so2_ri)
        elif band_label == 0.0 and (outband_sample is None):
            outband_sample = (i, mix_ri, so1_ri, so2_ri)
        if inband_sample and outband_sample:
            break

    if inband_sample is None:
        print("WARNING: No in-band sample (band_label=1) found in test data.")
    if outband_sample is None:
        print("WARNING: No out-of-band sample (band_label=0) found in test data.")

    # 4) Inference + plot
    def run_and_plot(sample_tuple, label):
        if sample_tuple is None:
            return
        idx, mix_ri, so1_ri, so2_ri = sample_tuple

        # Forward pass
        mix_t = torch.from_numpy(mix_ri).unsqueeze(0).to(device)
        with torch.no_grad():
            est_out = model(mix_t)  # => (1,1,L,2)
        est_ri = est_out[0,0].cpu().numpy()

        # Save plots
        out_path = os.path.join(args.output_dir, f"{label}_idx{idx}.png")
        title_str = f"{label} (test idx={idx})"
        plot_time_and_freq(mix_ri, so1_ri, est_ri, so2_ri,
                           plot_path=out_path,
                           title_prefix=title_str)

    run_and_plot(inband_sample,  "inband_sample")
    run_and_plot(outband_sample, "outband_sample")

    print(f"Done. Plots saved in {args.output_dir}")

if __name__ == "__main__":
    main()

