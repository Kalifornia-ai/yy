#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F

########################################
# 1) Arg Parsing
########################################
def parse_args():
    parser = argparse.ArgumentParser("Single-Source LSTM separation from PKL dataset.")
    parser.add_argument("--pkl_file", type=str,
                        default="./dataset/Training_Dataset_BASEBAND_CW_SOI_QPSK.pkl",
                        help="Path to the .pkl dataset (single source + mixture).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./results/",
                        help="Output directory for saving best_model.pth and plots.")
    parser.add_argument("--train_split", type=float, default=0.64,
                        help="Fraction of dataset for training (remaining is val/test)")
    parser.add_argument("--val_split", type=float, default=0.16,
                        help="Fraction of dataset for validation (rest is test)")
    #
    # New argument: sinr_db (so we can name model/plots accordingly)
    #
    parser.add_argument("--sinr_db", type=float, default=None,
                        help="SINR in dB (used only for naming output files).")
    return parser.parse_args()

########################################
# 2) PKL Dataset
########################################
class SingleSourceDataset(Dataset):
    """
    Expects a .pkl with either:
      (all_sig_mixture, all_sig1, bits, meta_data)
      or
      (all_sig_mixture, all_sig1, all_sig2, bits, meta_data)
    We only *require* so1; so2 is optional.
    """
    def __init__(self, pkl_file):
        super().__init__()
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # We'll see if there's a so2 in the data
        # (the 3rd item if length=5).
        self.all_sig_mixture = None
        self.all_sig1        = None
        self.all_sig2        = None
        self.all_bits1       = None
        self.meta_data       = None

        if len(data) == 5:
            self.all_sig_mixture, self.all_sig1, self.all_sig2, self.all_bits1, self.meta_data = data
        elif len(data) == 4:
            self.all_sig_mixture, self.all_sig1, self.all_bits1, self.meta_data = data
        else:
            raise ValueError("Expected 4 or 5 items in PKL: (mixture, so1, so2, bits, meta).")

        self.x_mixture = self._complex2twochannel(self.all_sig_mixture)
        self.x_so1     = self._complex2twochannel(self.all_sig1)

        if self.all_sig2 is not None:
            self.x_so2 = self._complex2twochannel(self.all_sig2)
        else:
            self.x_so2 = None

        self.N = self.x_mixture.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mix_ri = self.x_mixture[idx]  # shape => (L, 2)
        so1_ri = self.x_so1[idx]      # (L, 2)
        so2_ri = self.x_so2[idx] if (self.x_so2 is not None) else None

        mix_t = torch.from_numpy(mix_ri)  # (L,2)
        so1_t = torch.from_numpy(so1_ri)
        if so2_ri is not None:
            so2_t = torch.from_numpy(so2_ri)
        else:
            so2_t = torch.zeros_like(so1_t)  # or None

        return (mix_t, so1_t, so2_t)

    @staticmethod
    def _complex2twochannel(arr_cx):
        """
        arr_cx => shape (N, L) with complex
        => returns float32 shape (N, L, 2)
        """
        arr_cx = np.asarray(arr_cx)
        arr_out = np.stack([arr_cx.real, arr_cx.imag], axis=-1).astype(np.float32)
        return arr_out

########################################
# 3) Single-Source LSTM Model
########################################
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
        # x => (batch, channels, seq_len)
        b, c, _ = x.shape
        squeezed = self.avg_pool(x).view(b, c)
        excitation = self.fc(squeezed).view(b, c, 1)
        return x * excitation

class LSTMSeperatorSingle(nn.Module):
    """
    This model returns only 1 source => shape => (batch, 1, L, 2).
    """
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_sources=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_sources = num_sources  # 1
        self.conv_in = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.lstm_layers = nn.ModuleList()
        self.se_blocks   = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_size if i == 0 else hidden_size*2
            self.lstm_layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True, bidirectional=True))
            self.se_blocks.append(SELayer1D(hidden_size*2))

        self.conv_out = nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_size*2)

        # Output mask => shape => (b, seq_len, hidden_size*2 * num_sources=1)
        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2 * num_sources),
            nn.Sigmoid()
        )
        self.decoder = nn.Conv1d(hidden_size*2, input_size, kernel_size=3, padding=1)

    def forward(self, x):
        # x => (batch, seq_len, 2)
        x = x.permute(0, 2, 1)  # => (b, channels=2, seq_len)
        x = self.conv_in(x)     # => (b, hidden_size, seq_len)
        x = x.permute(0, 2, 1)  # => (b, seq_len, hidden_size)

        lstm_out = x
        for lstm_layer, se_block in zip(self.lstm_layers, self.se_blocks):
            lstm_out, _ = lstm_layer(lstm_out)  # => (b, seq_len, hidden_size*2)
            lstm_out = lstm_out.permute(0, 2, 1)
            lstm_out = se_block(lstm_out)
            lstm_out = lstm_out.permute(0, 2, 1)

        coding_feat = lstm_out.permute(0, 2, 1)
        coding_feat = self.conv_out(coding_feat)
        coding_feat = coding_feat.permute(0, 2, 1)
        coding_feat = self.layer_norm(coding_feat)

        b, seq_len, feat_dim = coding_feat.shape
        masks = self.mask_generator(coding_feat)   # => (b, seq_len, feat_dim*1)
        masks = masks.view(b, seq_len, feat_dim, self.num_sources)  # => (b, seq_len, feat_dim, 1)

        coding_feat_expanded = coding_feat.unsqueeze(-1)  # => (b, seq_len, feat_dim, 1)
        masked_features = coding_feat_expanded * masks    # => (b, seq_len, feat_dim, 1)

        # decode => shape => (b, 2, seq_len)
        out_sources = []
        src_feat = masked_features[..., 0]  # => (b, seq_len, feat_dim)
        src_feat = src_feat.permute(0, 2, 1)  # => (b, feat_dim, seq_len)
        decoded = self.decoder(src_feat)      # => (b, 2, seq_len)
        decoded = decoded.permute(0, 2, 1)    # => (b, seq_len, 2)
        out_sources.append(decoded)

        return torch.stack(out_sources, dim=1)  # => (b, 1, seq_len, 2)

########################################
# 4) Loss Functions
########################################
def si_snr_single(est, ref, eps=1e-8):
    """
    negative single SI-SNR => shape => (b, seq_len, 2)
    """
    B, T, F = est.shape
    est = est.view(B, -1)
    ref = ref.view(B, -1)
    est_zm = est - est.mean(dim=1, keepdim=True)
    ref_zm = ref - ref.mean(dim=1, keepdim=True)
    dot = torch.sum(est_zm*ref_zm, dim=1, keepdim=True)
    norm_ref = torch.sum(ref_zm**2, dim=1, keepdim=True)+eps
    proj = dot/norm_ref*ref_zm
    e = est_zm - proj
    si_snr_value = 10*torch.log10(torch.sum(proj**2, dim=1)/(torch.sum(e**2,dim=1)+eps))
    return -si_snr_value.mean()

def mse_loss_db(pred, target, eps=1e-8):
    """
    For demonstration: standard MSE, convert to dB scale.
    This returns a single scalar in dB. We'll treat that as our 'loss.'
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    mse_db = 10*torch.log10(mse + eps)
    #  # 3) Apply the smoothed logistic function
    # M = 0.5 * (L + U)  # midpoint
    # # L + (U-L) / [1 + exp(-alpha*(mse_dB - M))]
    # loss = L + (U - L) / (1.0 + torch.exp(-alpha * (mse_db - M)))
    # 3) Apply the smoothed logistic function
    L = -100
    U = 50
    alpha = 0.1
    M = 0.5 * (L + U)  # midpoint
    # L + (U-L) / [1 + exp(-alpha*(mse_dB - M))]
    loss = L + (U - L) / (1.0 + torch.exp(-alpha * (mse_db - M)))

    # # 4) (Optional) clamp the final output
    # #  -- Typically you'd saturate to [L, U], but often the sigmoid formula
    # #     already does this in practice. If you want a strict clamp:
    # # loss = torch.clamp(loss, min=L, max=U)

    return loss

########################################
# 5) Helper: Time & Frequency Plots
########################################
def plot_time_and_freq(mix_ri, so1_ri, est_ri, so2_ri=None,
                       plot_dir="./results", prefix="sample0"):
    """
    mix_ri, so1_ri, est_ri => shape (T,2) real+imag
    so2_ri => shape (T,2) if available
    Saves time-domain and freq-domain plots to plot_dir with names:
       prefix_time.png
       prefix_fft.png
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Time-domain: real+imag
    fig, axes = plt.subplots(3 if so2_ri is None else 4, 2, figsize=(10,10), sharex=True)
    fig.suptitle(f"Time-Domain: {prefix}")
    row = 0
    for name, data in [("Mixture", mix_ri), ("SOI (True)", so1_ri), ("Estimate", est_ri)]:
        axes[row,0].plot(data[:,0], color='b', label="Real")
        axes[row,0].set_ylabel(name+" Real")
        axes[row,1].plot(data[:,1], color='r', label="Imag")
        axes[row,1].set_ylabel(name+" Imag")
        row+=1
    if so2_ri is not None:
        # Interference row
        axes[row,0].plot(so2_ri[:,0], color='b')
        axes[row,0].set_ylabel("Interf Real")
        axes[row,1].plot(so2_ri[:,1], color='r')
        axes[row,1].set_ylabel("Interf Imag")

    plt.tight_layout()
    out_path1 = os.path.join(plot_dir, f"{prefix}_time.png")
    plt.savefig(out_path1, dpi=150)
    plt.close()

    # Frequency-domain
    # We'll do magnitude spectrum for brevity, but you can do real/imag or log-scale, etc.
    def get_magnitude_spectrum(sig_ri):
        # sig_ri => (T,2)
        # Convert to complex => shape (T,)
        cplx = sig_ri[:,0] + 1j*sig_ri[:,1]
        spec = np.fft.fftshift(np.fft.fft(cplx))
        mag  = np.abs(spec)
        return mag

    mix_mag = get_magnitude_spectrum(mix_ri)
    so1_mag = get_magnitude_spectrum(so1_ri)
    est_mag = get_magnitude_spectrum(est_ri)
    so2_mag = get_magnitude_spectrum(so2_ri) if so2_ri is not None else None
    freqs   = np.arange(len(mix_mag)) - (len(mix_mag)//2)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(freqs, mix_mag, label="Mixture")
    ax2.plot(freqs, so1_mag, label="SOI True")
    ax2.plot(freqs, est_mag, label="SOI Est", linestyle="--")
    if so2_mag is not None:
        ax2.plot(freqs, so2_mag, label="Interference", linestyle=":")
    ax2.set_title(f"FFT Magnitude: {prefix}")
    ax2.legend()
    ax2.set_xlabel("FFT Bins (shifted)")
    out_path2 = os.path.join(plot_dir, f"{prefix}_fft.png")
    plt.savefig(out_path2, dpi=150)
    plt.close()

########################################
# 6) Main
########################################
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If the user gave a sinr_db, we can incorporate it into naming.
    sinr_tag = f"_SINR{args.sinr_db}dB" if args.sinr_db is not None else ""

    print("Loading dataset from:", args.pkl_file)
    ds = SingleSourceDataset(args.pkl_file)
    total_len = len(ds)

    train_len = int(args.train_split*total_len)
    val_len   = int(args.val_split*total_len)
    test_len  = total_len - train_len - val_len
    print(f"Total samples={total_len}, => train={train_len}, val={val_len}, test={test_len}")

    train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len], 
                                             generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Single-Source model => num_sources=1
    model = LSTMSeperatorSingle(num_sources=1, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    # Make subdirectory for results of this run
    run_plot_dir = os.path.join(args.output_dir, f"plots{sinr_tag}")
    os.makedirs(run_plot_dir, exist_ok=True)

    # Train
    for epoch in range(args.epochs):
        model.train()
        running_loss=0.
        for mix, so1, so2 in train_loader:
            mix, so1 = mix.to(device), so1.to(device)
            optimizer.zero_grad()
            est = model(mix)  # => (b,1,T,2)
            est_s1 = est[:,0]  # => (b,T,2)

            # Use MSE in dB (example) or SI-SNR:
            loss = mse_loss_db(est_s1, so1)  
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss/len(train_loader)

        # val
        model.eval()
        val_loss_sum=0.
        with torch.no_grad():
            for mix, so1, so2 in val_loader:
                mix, so1 = mix.to(device), so1.to(device)
                est = model(mix)
                est_s1 = est[:,0]
                val_loss = mse_loss_db(est_s1, so1)
                val_loss_sum+= val_loss.item()
        epoch_val_loss = val_loss_sum/len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}: train={epoch_train_loss:.4f}, val={epoch_val_loss:.4f}")
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            # Save model with a name that includes the sinr tag
            model_save_path = os.path.join(args.output_dir, f"best_model{sinr_tag}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} => {model_save_path}")

    # test
    model_save_path = os.path.join(args.output_dir, f"best_model{sinr_tag}.pth")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    test_loss_sum=0.
    with torch.no_grad():
        for mix, so1, so2 in test_loader:
            mix, so1 = mix.to(device), so1.to(device)
            est = model(mix)
            est_s1 = est[:,0]
            test_loss = mse_loss_db(est_s1, so1)
            test_loss_sum+=test_loss.item()
    final_test_loss=test_loss_sum/len(test_loader)
    print(f"Final test loss (SI-SNR): {final_test_loss:.4f}")

    # Print final train/val from last epoch
    final_train_loss = epoch_train_loss
    final_val_loss   = epoch_val_loss
    print(f"FINAL_TRAIN_LOSS={final_train_loss:.4f}")
    print(f"FINAL_VAL_LOSS={final_val_loss:.4f}")
    print(f"FINAL_TEST_LOSS={final_test_loss:.4f}")

    # Plot a sample from test set
    mix_batch, so1_batch, so2_batch = next(iter(test_loader))
    mix_batch, so1_batch, so2_batch = mix_batch.to(device), so1_batch.to(device), so2_batch.to(device)
    with torch.no_grad():
        est_out = model(mix_batch)  # => (b,1,T,2)
    # pick item 0
    mixture_0 = mix_batch[0].cpu().numpy()  # shape => (T,2)
    so1_0     = so1_batch[0].cpu().numpy()  # shape => (T,2)
    so2_0     = so2_batch[0].cpu().numpy()  # shape => (T,2) only if it was real interference
    est_s1_0  = est_out[0,0].cpu().numpy()  # (T,2)

    # Generate time & frequency plots => store in run_plot_dir
    sample_prefix = "testSample0" + sinr_tag
    plot_time_and_freq(mixture_0, so1_0, est_s1_0, so2_0, 
                       plot_dir=run_plot_dir, prefix=sample_prefix)

if __name__=="__main__":
    args = parse_args()
    main()


