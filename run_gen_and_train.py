#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
import re
import matplotlib.pyplot as plt

def parse_losses_from_output(output_str):
    """
    Given the captured stdout from your training script,
    extract final train/val/test losses (floats) from lines like:
      FINAL_TRAIN_LOSS=0.2345
      FINAL_VAL_LOSS=0.3456
      FINAL_TEST_LOSS=0.4567
    Returns (train_loss, val_loss, test_loss) as floats.
    If something not found, returns None for that field.
    """
    train_loss = None
    val_loss   = None
    test_loss  = None

    # Regex to find lines like "FINAL_TRAIN_LOSS=0.2345"
    train_match = re.search(r"FINAL_TRAIN_LOSS\s*=\s*([0-9.Ee+\-]+)", output_str)
    val_match   = re.search(r"FINAL_VAL_LOSS\s*=\s*([0-9.Ee+\-]+)", output_str)
    test_match  = re.search(r"FINAL_TEST_LOSS\s*=\s*([0-9.Ee+\-]+)", output_str)

    if train_match:
        train_loss = float(train_match.group(1))
    if val_match:
        val_loss = float(val_match.group(1))
    if test_match:
        test_loss = float(test_match.group(1))

    return train_loss, val_loss, test_loss

def main():
    parser = argparse.ArgumentParser(
        description="Sweep SINR from -10 to 25 dB, parse final train/val/test losses, and plot them."
    )
    # Arguments for data generation
    parser.add_argument("--gen_script", type=str, default="./dataset_utils/generate_demod_trainmixture.py",
                        help="Data generation script.")
    parser.add_argument("--soi_sig_type", type=str, required=True)
    parser.add_argument("--interference_sig_type", type=str, required=True)
    parser.add_argument("--n_per_batch", type=int, default=100)
    
    # Arguments for training
    parser.add_argument("--train_script", type=str, default="train_separation_from_pkl.py",
                        help="Training script path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--train_split", type=float, default=0.64)
    parser.add_argument("--val_split", type=float, default=0.16)

    # Optionally skip training
    parser.add_argument("--skip_train", action="store_true")

    args = parser.parse_args()

    # Our chosen SINR sweep: -10, -5, 0, 5, 10, 15, 20, 25
    sinr_list = list(range(-10, 21, 5))
    train_losses = []
    val_losses   = []
    test_losses  = []

    for sinr_db in sinr_list:
        print("=====================================")
        print(f"Processing SINR={sinr_db} dB")
        print("=====================================")

        # Construct the dataset filename
        pkl_name = f"Training_Dataset_{args.soi_sig_type}_{args.interference_sig_type}_SINR{float(sinr_db)}dB.pkl"
        out_path = os.path.join("./dataset", pkl_name)

        # 1) Generate dataset
        gen_cmd = [
            sys.executable,
            args.gen_script,
            "--soi_sig_type", args.soi_sig_type,
            "--interference_sig_type", args.interference_sig_type,
            "--sinr_db", str(sinr_db),
            "--n_per_batch", str(args.n_per_batch)
        ]
        print("Running data gen:", " ".join(gen_cmd))
        subprocess.run(gen_cmd, check=True)

        if not os.path.isfile(out_path):
            print(f"ERROR: {out_path} not found. Skipping training.")
            train_losses.append(None)
            val_losses.append(None)
            test_losses.append(None)
            continue

        if args.skip_train:
            # If skipping training, just store None as placeholders
            train_losses.append(None)
            val_losses.append(None)
            test_losses.append(None)
            continue

        # 2) Train
        train_cmd = [
            sys.executable,
            args.train_script,
            "--pkl_file", out_path,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--output_dir", args.output_dir,
            "--train_split", str(args.train_split),
            "--val_split", str(args.val_split),
        ]
        print("Running training:", " ".join(train_cmd))
        completed_proc = subprocess.run(train_cmd, check=True, capture_output=True, text=True)

        # 3) Parse final losses from the training script’s stdout
        stdout_str = completed_proc.stdout
        train_loss, val_loss, test_loss = parse_losses_from_output(stdout_str)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    # Print summary
    print("\n=== Final losses per SINR ===")
    for sinr, trl, vl, tl in zip(sinr_list, train_losses, val_losses, test_losses):
        print(f"SINR={sinr} => train={trl}, val={vl}, test={tl}")

    # 4) Plot
    #   You can do (A) single plot with 3 lines
    #   or (B) 3 separate subplots. We'll do 3 subplots for clarity.

    # Filter out any None if you used --skip_train or if any run failed
    # but for a normal run, we'll assume we have numeric values.
    sinr_array = []
    tr_arr, vl_arr, te_arr = [], [], []
    for sdb, tr, va, te in zip(sinr_list, train_losses, val_losses, test_losses):
        if (tr is not None) and (va is not None) and (te is not None):
            sinr_array.append(sdb)
            tr_arr.append(tr)
            vl_arr.append(va)
            te_arr.append(te)

    plt.figure(figsize=(8,8))

    # Train
    plt.subplot(3,1,1)
    plt.plot(sinr_array, tr_arr, marker='o')
    plt.title("Train Loss vs. SINR")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Train Loss")

    # Val
    plt.subplot(3,1,2)
    plt.plot(sinr_array, vl_arr, marker='s', color='orange')
    plt.title("Val Loss vs. SINR")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Val Loss")

    # Test
    plt.subplot(3,1,3)
    plt.plot(sinr_array, te_arr, marker='^', color='green')
    plt.title("Test Loss vs. SINR")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Test Loss")

    plt.tight_layout()
    plt.savefig("loss_curves_vs_sinr.png", dpi=150)
    plt.show()

    print("All done. Plots saved => loss_curves_vs_sinr.png")

if __name__ == "__main__":
    main()
