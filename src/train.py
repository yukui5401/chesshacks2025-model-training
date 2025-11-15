import sys
import torch
import torch.nn
import pandas as pd
import modal

sys.path.append("/root/src")
from latent_model import LatentEncoder, fen_to_tensor

vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)

app = modal.App("chessbot")

# GPU-safe batch size for SupCon
BATCH_SIZE = 1024  # conv model uses more memory; adjust if needed
CHECKPOINT_INTERVAL = 50  # steps


@app.function(
    image=modal.Image.debian_slim()
    .pip_install(["torch", "numpy", "pandas", "python-chess"])
    .add_local_dir(
        "/Users/brookeyang/Projects/hackathons/chesshacks/training/src",
        remote_path="/root/src",
    ),
    gpu="A100",
    timeout=60 * 60 * 4,
    volumes={"/checkpoints": vol},
)
def run_training():
    csv_path = "/root/src/stockfish_evaluations.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Model + optimizer
    # -------------------------
    encoder = LatentEncoder(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    step = 0  # global step counter

    for chunk in pd.read_csv(csv_path, chunksize=BATCH_SIZE):
        inputs = []
        combined_labels = []

        # Map moves to integer IDs
        move_to_id = {move: idx for idx, move in enumerate(chunk["best_move"].unique())}

        for _, row in chunk.iterrows():
            # --- FEN → tensor ---
            x_tensor = fen_to_tensor(row["fen"]).float().to(device)
            inputs.append(x_tensor)

            # --- Evaluation label ---
            eval_ = row["evaluation"]
            if isinstance(eval_, str):
                if eval_.startswith("M"):
                    eval_label = -1
                else:
                    try:
                        eval_label = int(round(float(eval_)))
                    except ValueError:
                        eval_label = 0
            else:
                eval_label = int(round(eval_))

            # --- Move label ---
            move_label = move_to_id[row["best_move"]]

            # --- Combine labels into single integer ---
            # Simple approach: combine as tuple (eval_label, move_label) → unique integer
            # Here we shift eval_label by max_move_id + 1 to ensure uniqueness
            combined_label = eval_label * (len(move_to_id) + 1) + move_label
            combined_labels.append(combined_label)

        # -------------------------
        # Stack tensors and labels
        # -------------------------
        x = torch.stack(inputs)  # [batch, 14, 8, 8]
        y = torch.tensor(combined_labels, dtype=torch.long, device=device)

        # -------------------------
        # Forward + loss
        # -------------------------
        z = encoder(x)
        loss = LatentEncoder.supervised_contrastive_loss(z, y)

        # -------------------------
        # Backprop
        # -------------------------
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        print(f"Step {step} | Batch loss: {loss.item():.4f}")

        # -------------------------
        # Checkpoint
        # -------------------------
        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"/checkpoints/encoder_checkpoint_step{step}.pt"
            torch.save(
                {
                    "model_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint at step {step} -> {checkpoint_path}")

    # -------------------------
    # Save final model
    # -------------------------
    final_path = "/checkpoints/encoder_final.pt"
    torch.save(encoder.state_dict(), final_path)
    print(f"Saved final model -> {final_path}")


@app.local_entrypoint()
def main():
    run_training.remote()
