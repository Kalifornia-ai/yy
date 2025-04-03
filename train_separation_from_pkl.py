import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

#############################
# 1) Command-line arguments #
#############################
def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM separation from PKL dataset.")
    parser.add_argument("--pkl_file", type=str, default="./dataset/Training_Dataset_BASEBAND_CW_SOI_BASEBAND_CW_INTERFERENCE.pkl", required=False,
                        help="Path to the .pkl file (e.g. dataset/Training_Dataset_...).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--train_split", type=float, default=0.64,
                        help="Fraction of dataset for training (remaining is split into val/test)")
    parser.add_argument("--val_split", type=float, default=0.16,
                        help="Fraction of dataset for validation (rest is for test)")
    return parser.parse_args()


################################
# 2) Dataset class for the PKL #
################################
class PklSeparationDataset(Dataset):
    """
    Loads a previously generated .pkl file that contains:
      all_sig_mixture : shape [N, sig_len] (complex)
      all_sig1        : shape [N, sig_len] (complex) [the SOI, for example]
      all_sig2        : shape [N, sig_len] (complex) [the Interference], optional
      meta_data       : shape [N, 5] or something

    If 'all_sig2' doesn't exist, you can do so2 = mixture - so1 on-the-fly.
    The dataset yields (mixture, [so1, so2]) as Tensors for training.
    """

    def __init__(self, pkl_file, has_src2=False):
        """
        Args:
          pkl_file: Path to the .pkl
          has_src2: If True, the pkl includes an explicit 'all_sig2'.
                    Otherwise, we'll reconstruct it as mixture - so1.
        """
        super().__init__()
        with open(pkl_file, "rb") as f:
            loaded_data = pickle.load(f)

        if len(loaded_data) == 4:
            all_sig_mixture, all_sig1, all_bits1, meta_data = loaded_data
            # If your pkl stores only one clean source (SOI), we assume the "interference" is mixture - soi.
            # Otherwise, adapt as needed if you have a second array.
            self.all_sig_mixture = all_sig_mixture
            self.all_sig1        = all_sig1
            self.meta_data       = meta_data
            # For demonstration, ignoring 'all_bits1'.
            self.all_sig2 = None
            self.has_src2 = has_src2
        elif len(loaded_data) == 5:
            # e.g. if your pkl actually had (all_sig_mixture, all_sig1, all_sig2, all_bits1, meta_data)
            all_sig_mixture, all_sig1, all_sig2, all_bits1, meta_data = loaded_data
            self.all_sig_mixture = all_sig_mixture
            self.all_sig1        = all_sig1
            self.all_sig2        = all_sig2
            self.meta_data       = meta_data
            self.has_src2        = True
        else:
            raise ValueError("Unexpected pickle format. Please adapt accordingly.")

        # Convert all from complex64 => separate real+imag if needed.
        # We'll store them as np.float32 with shape (N, sig_len, 2).
        self.x_mixture = self._complex2twochannel(self.all_sig_mixture)  # shape (N, sig_len, 2)
        self.x_src1    = self._complex2twochannel(self.all_sig1)
        if self.has_src2:
            self.x_src2 = self._complex2twochannel(self.all_sig2)
        else:
            # We'll do mixture - src1 as the second source
            self.x_src2 = self.x_mixture - self.x_src1

        self.length = self.x_mixture.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mixture = self.x_mixture[idx]  # shape (sig_len, 2)
        so1     = self.x_src1[idx]
        so2     = self.x_src2[idx]
        # Convert to torch
        mixture_t = torch.from_numpy(mixture)  # (sig_len, 2)
        so1_t     = torch.from_numpy(so1)
        so2_t     = torch.from_numpy(so2)
        return mixture_t, so1_t, so2_t

    @staticmethod
    def _complex2twochannel(arr_cx):
        """
        arr_cx: shape (N, sig_len) with complex dtype
        returns float32 array shape (N, sig_len, 2)
        """
        # ensure complex64 => real+imag
        arr_cx = np.asarray(arr_cx)  # shape (N, sig_len)
        arr_out = np.stack([arr_cx.real, arr_cx.imag], axis=-1).astype(np.float32)
        # shape => (N, sig_len, 2)
        return arr_out


##############################################
# 3) The LSTM-based Separation Model (same)  #
##############################################
class SELayer1D(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        if channels % reduction_ratio != 0:
            raise ValueError('channels must be divisible by reduction_ratio')
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        squeezed = self.avg_pool(x).view(b, c)
        excitation = self.fc(squeezed).view(b, c, 1)
        return x * excitation

class LSTMSeperator(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_sources=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_sources = num_sources

        self.conv_in = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.lstm_layers = nn.ModuleList()
        self.se_blocks   = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_size if i == 0 else hidden_size*2
            self.lstm_layers.append(
                nn.LSTM(in_dim, hidden_size, batch_first=True, bidirectional=True)
            )
            self.se_blocks.append(SELayer1D(hidden_size * 2))

        self.conv_out = nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2 * num_sources),
            nn.Sigmoid()
        )

        self.decoder = nn.Conv1d(hidden_size*2, input_size, kernel_size=3, padding=1)

    def forward(self, x):
        # x => (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)   # => (b, in_channels, seq_len)
        x = self.conv_in(x)      # => (b, hidden_size, seq_len)
        x = x.permute(0, 2, 1)   # => (b, seq_len, hidden_size)

        lstm_out = x
        for lstm_layer, se_block in zip(self.lstm_layers, self.se_blocks):
            lstm_out, _ = lstm_layer(lstm_out)  # => (b, seq_len, hidden_size*2)
            lstm_out = lstm_out.permute(0, 2, 1)
            lstm_out = se_block(lstm_out)
            lstm_out = lstm_out.permute(0, 2, 1)

        coding_feat = lstm_out.permute(0, 2, 1)         # => (b, hidden_size*2, seq_len)
        coding_feat = self.conv_out(coding_feat)        # => (b, hidden_size*2, seq_len)
        coding_feat = coding_feat.permute(0, 2, 1)      # => (b, seq_len, hidden_size*2)
        coding_feat = self.layer_norm(coding_feat)

        masks = self.mask_generator(coding_feat)        # => (b, seq_len, hidden_size*2*num_sources)
        b, seq_len, hf2_ns = masks.shape
        masks = masks.view(b, seq_len, self.hidden_size*2, self.num_sources)

        coding_feat_expanded = coding_feat.unsqueeze(-1)  # (b, seq_len, hidden_size*2, 1)
        masked_features = coding_feat_expanded * masks     # => (b, seq_len, hidden_size*2, num_sources)

        decoded_sources = []
        for i in range(self.num_sources):
            source_feat = masked_features[..., i]         # (b, seq_len, hidden_size*2)
            source_feat = source_feat.permute(0, 2, 1)    # (b, hidden_size*2, seq_len)
            decoded = self.decoder(source_feat)           # (b, input_size, seq_len)
            decoded = decoded.permute(0, 2, 1)            # (b, seq_len, input_size)
            decoded_sources.append(decoded)

        return torch.stack(decoded_sources, dim=1)       # (b, num_sources, seq_len, input_size)


#######################
# 4) Loss Functions   #
#######################
def si_snr_single(est, ref, eps=1e-8):
    """Compute negative SI-SNR for one estimated source vs. reference."""
    if est.dim() == 3:
        # flatten freq dimension if shape (B, T, F)
        B, T, F = est.shape
        est = est.view(B, -1)
        ref = ref.view(B, -1)
    est_zm = est - est.mean(dim=1, keepdim=True)
    ref_zm = ref - ref.mean(dim=1, keepdim=True)

    dot = torch.sum(est_zm * ref_zm, dim=1, keepdim=True)
    norm_ref = torch.sum(ref_zm**2, dim=1, keepdim=True) + eps
    proj = dot / norm_ref * ref_zm
    e = est_zm - proj
    si_snr_value = 10 * torch.log10(torch.sum(proj**2, dim=1) / (torch.sum(e**2, dim=1)+eps))
    return -si_snr_value.mean()

def si_snr_loss_2src(estimate, reference, eps=1e-8):
    """
    estimate, reference => (B, 2, T, F) or (B, 2, T).
    Non-PIT version: assume estimate[:,0] matches reference[:,0], etc.
    """
    src1_est = estimate[:, 0]
    src1_ref = reference[:, 0]
    src2_est = estimate[:, 1]
    src2_ref = reference[:, 1]

    loss1 = si_snr_single(src1_est, src1_ref, eps=eps)
    loss2 = si_snr_single(src2_est, src2_ref, eps=eps)
    return (loss1 + loss2)*0.5

def si_snr_loss_2src_pit(estimate, reference):
    """
    Permutation-invariant 2-source SI-SNR
    """
    est0_ref0 = si_snr_single(estimate[:,0], reference[:,0])
    est1_ref1 = si_snr_single(estimate[:,1], reference[:,1])
    pair1_loss = (est0_ref0 + est1_ref1)*0.5

    est0_ref1 = si_snr_single(estimate[:,0], reference[:,1])
    est1_ref0 = si_snr_single(estimate[:,1], reference[:,0])
    pair2_loss = (est0_ref1 + est1_ref0)*0.5

    return torch.min(pair1_loss, pair2_loss)


##############################
# 5) Main Training Script   #
##############################
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading dataset from: {args.pkl_file}")

    # 5.1) Load the dataset
    # set has_src2=True if your pkl contains a second source
    # otherwise the dataset will do so2 = mixture - so1
    dataset = PklSeparationDataset(args.pkl_file, has_src2=False)

    total_len = len(dataset)
    train_len = int(args.train_split * total_len)
    val_len   = int(args.val_split * total_len)
    test_len  = total_len - train_len - val_len

    print(f"Total samples: {total_len} => train={train_len}, val={val_len}, test={test_len}")
    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    # 5.2) Instantiate the model
    model = LSTMSeperator(
        input_size=2,    # real+imag
        hidden_size=128,
        num_layers=2,
        num_sources=2,   # separate [so1, so2]
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_val_loss = float('inf')

    epochs = args.epochs
    for epoch in range(epochs):
        ###################
        # ---- TRAIN ---- #
        ###################
        model.train()
        running_loss = 0.0
        for mixture, so1, so2 in train_loader:
            mixture, so1, so2 = mixture.to(device), so1.to(device), so2.to(device)

            optimizer.zero_grad()
            est = model(mixture)  # shape (B, 2, seq_len, 2)
            # stack references => shape (B, 2, seq_len, 2)
            refs = torch.stack([so1, so2], dim=1)
            loss = si_snr_loss_2src_pit(est, refs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        ######################
        # ---- VALIDATE ---- #
        ######################
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for mixture, so1, so2 in val_loader:
                mixture, so1, so2 = mixture.to(device), so1.to(device), so2.to(device)
                est = model(mixture)
                refs = torch.stack([so1, so2], dim=1)
                val_loss = si_snr_loss_2src_pit(est, refs)
                val_loss_sum += val_loss.item()

        epoch_val_loss = val_loss_sum / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} => Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"Best model updated at epoch {epoch+1}, val_loss={best_val_loss:.4f}")

    ###################
    # ---- TEST ----  #
    ###################
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for mixture, so1, so2 in test_loader:
            mixture, so1, so2 = mixture.to(device), so1.to(device), so2.to(device)
            est = model(mixture)
            refs = torch.stack([so1, so2], dim=1)
            test_loss = si_snr_loss_2src_pit(est, refs)
            test_loss_sum += test_loss.item()

    final_test_loss = test_loss_sum / len(test_loader)
    print(f"Final Test Loss (SI-SNR, negative): {final_test_loss:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
