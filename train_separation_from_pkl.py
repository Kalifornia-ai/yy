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

########################################
# 1) Arg Parsing
########################################
def parse_args():
    parser = argparse.ArgumentParser("Single-Source LSTM separation from PKL dataset.")
    parser.add_argument("--pkl_file", type=str,
                        default="./dataset/Training_Dataset_BASEBAND_CW_SOI_QPSK.pkl",
                        help="Path to the .pkl dataset (single source + mixture).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./results/",
                        help="Output directory for saving best_model.pth")
    parser.add_argument("--train_split", type=float, default=0.64,
                        help="Fraction of dataset for training (remaining is split into val/test)")
    parser.add_argument("--val_split", type=float, default=0.16,
                        help="Fraction of dataset for validation (rest is test)")
    return parser.parse_args()

########################################
# 2) PKL Dataset: single source
########################################
class SingleSourceDataset(Dataset):
    """
    Expects a .pkl with either:
      (all_sig_mixture, all_sig1, bits, meta_data)
      or
      (all_sig_mixture, all_sig1, all_sig2, bits, meta_data)
    But we only care about so1 as the single source.

    We'll store them as real+imag => shape (N, L, 2).
    """
    def __init__(self, pkl_file):
        super().__init__()
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        if len(data) == 5:
            self.all_sig_mixture, self.all_sig1, _, self.all_bits1, self.meta_data = data
        elif len(data) == 4:
            self.all_sig_mixture, self.all_sig1, self.all_bits1, self.meta_data = data
        else:
            raise ValueError("Expected 4 or 5 items in PKL: (mixture, so1, so2, bits, meta).")

        self.x_mixture = self._complex2twochannel(self.all_sig_mixture)
        self.x_so1     = self._complex2twochannel(self.all_sig1)

        self.N = self.x_mixture.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mix_ri = self.x_mixture[idx]  # shape => (L, 2)
        so1_ri = self.x_so1[idx]      # (L, 2)

        # Convert to Torch
        mix_t = torch.from_numpy(mix_ri)  # (L,2)
        so1_t = torch.from_numpy(so1_ri)
        return (mix_t, so1_t)

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
# 5) Single-Source Loss (SI-SNR or MSE)
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

########################################
# 6) Main Script
########################################
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset from:", args.pkl_file)
    ds = SingleSourceDataset(args.pkl_file)
    total_len = len(ds)

    train_len = int(args.train_split*total_len)
    val_len   = int(args.val_split*total_len)
    test_len  = total_len - train_len - val_len
    print(f"Total samples={total_len}, => train={train_len}, val={val_len}, test={test_len}")

    train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Single-Source model => num_sources=1
    model = LSTMSeperatorSingle(num_sources=1, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    # Train
    for epoch in range(args.epochs):
        model.train()
        running_loss=0.
        for mix, so1 in train_loader:
            mix, so1 = mix.to(device), so1.to(device)
            optimizer.zero_grad()
            est = model(mix)  # => (b,1,T,2)
            # shape => (b,T,2)
            est_s1 = est[:,0]
            # compute loss => si_snr_single
            loss = si_snr_single(est_s1, so1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss/len(train_loader)

        # val
        model.eval()
        val_loss_sum=0.
        with torch.no_grad():
            for mix, so1 in val_loader:
                mix, so1 = mix.to(device), so1.to(device)
                est = model(mix)
                est_s1 = est[:,0]
                val_loss = si_snr_single(est_s1, so1)
                val_loss_sum+= val_loss.item()
        epoch_val_loss = val_loss_sum/len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}: train={epoch_train_loss:.4f}, val={epoch_val_loss:.4f}")
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir,"best_model.pth"))
            print(f"Saved best model at epoch {epoch+1}")

    # test
    model.load_state_dict(torch.load(os.path.join(args.output_dir,"best_model.pth"),map_location=device))
    model.eval()
    test_loss_sum=0.
    with torch.no_grad():
        for mix, so1 in test_loader:
            mix, so1 = mix.to(device), so1.to(device)
            est = model(mix)
            est_s1 = est[:,0]
            test_loss = si_snr_single(est_s1, so1)
            test_loss_sum+=test_loss.item()
    final_test_loss=test_loss_sum/len(test_loader)
    print(f"Final test loss: {final_test_loss:.4f}")

    # Plot first item from first test batch
    mix_batch, so1_batch = next(iter(test_loader))
    mix_batch, so1_batch = mix_batch.to(device), so1_batch.to(device)
    with torch.no_grad():
        est_out = model(mix_batch)  # => (b,1,T,2)
    # pick item 0
    mixture_0 = mix_batch[0].cpu().numpy()  # shape => (T,2)
    so1_0     = so1_batch[0].cpu().numpy()  # shape => (T,2)
    est_s1_0  = est_out[0,0].cpu().numpy()  # (T,2)

    # We do 2 rows => mixture row, so1 row (GT vs. Est)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
    t_axis = np.arange(mixture_0.shape[0])

    # Row 1 => mixture
    axes[0,0].plot(t_axis, mixture_0[:,0], label="Mix Real")
    axes[0,0].set_title("Mixture Real")
    axes[0,1].plot(t_axis, mixture_0[:,1], label="Mix Imag")
    axes[0,1].set_title("Mixture Imag")

    # Row 2 => so1
    axes[1,0].plot(t_axis, so1_0[:,0], label="GT Real")
    axes[1,0].plot(t_axis, est_s1_0[:,0], label="Est Real", linestyle="--")
    axes[1,0].set_title("SOI Real")
    axes[1,0].legend()

    axes[1,1].plot(t_axis, so1_0[:,1], label="GT Imag")
    axes[1,1].plot(t_axis, est_s1_0[:,1], label="Est Imag", linestyle="--")
    axes[1,1].set_title("SOI Imag")
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    args = parse_args()
    main()

