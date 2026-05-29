"""
Training script for long-context sequence modeling on WikiText-2.
Supports various attention mechanisms, positional encodings, and hybrid architectures.
"""
import time
import argparse
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_dataloaders
from transformer import SequenceModel

def calculate_perplexity(loss):
    return torch.exp(loss).item()

def main():
    parser = argparse.ArgumentParser(description="Train SequenceModel on WikiText-2 with modular options.")
    parser.add_argument("--attn_type", type=str, default="standard", choices=["standard", "mqa", "sliding_window", "linear", "conv_before", "gated_conv_ffn"], help="Type of attention block")
    parser.add_argument("--pos_enc", type=str, default="sinusoidal", choices=["sinusoidal", "rope", "alibi", "relative"], help="Type of positional encoding")
    parser.add_argument("--context_length", type=int, default=512, help="Context length for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps per epoch/evaluation for quick dry runs")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: attn_type={args.attn_type}, pos_enc={args.pos_enc}, context_length={args.context_length}, batch_size={args.batch_size}, epochs={args.epochs}, max_steps={args.max_steps}")

    # Hyperparameters
    batch_size = args.batch_size
    context_length = args.context_length
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    epochs = args.epochs
    
    if args.wandb:
        import wandb
        wandb.init(
            project="saidl-core-ml",
            config={
                "attn_type": args.attn_type,
                "pos_enc": args.pos_enc,
                "context_length": context_length,
                "batch_size": batch_size,
                "epochs": epochs,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers
            }
        )
    
    # 1. Load Data for Training
    train_loader, val_loader, vocab_size = get_dataloaders(batch_size=batch_size, context_length=context_length)
    
    # 2. Initialize Model
    model = SequenceModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attn_type=args.attn_type,
        pos_enc=args.pos_enc
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        tokens_processed = 0
        steps_run = 0
        
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Loss computation: (B, T, V) -> (B*T, V) for cross entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            tokens_processed += x.numel()
            steps_run += 1
            
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}", flush=True)
                if args.wandb:
                    wandb.log({"train_step_loss": loss.item()})
                
            if args.max_steps is not None and steps_run >= args.max_steps:
                break
                
        epoch_time = time.time() - start_time
        throughput = tokens_processed / epoch_time
        avg_train_loss = total_loss / steps_run
        
        # 4. Validation Loop
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
                val_steps += 1
                if args.max_steps is not None and val_steps >= args.max_steps:
                    break
                
        avg_val_loss = val_loss / val_steps
        val_perplexity = calculate_perplexity(torch.tensor(avg_val_loss))
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"Peak GPU Memory: {peak_memory:.2f} MB", flush=True)
            
        print(f"--- Epoch {epoch} Summary ---", flush=True)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)
        print(f"Val Perplexity: {val_perplexity:.2f}", flush=True)
        print(f"Throughput: {throughput:.2f} tokens/sec\n", flush=True)

        if args.wandb:
            wandb_log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_perplexity": val_perplexity,
                "throughput": throughput
            }
            if torch.cuda.is_available():
                wandb_log_dict["peak_memory_mb"] = peak_memory
            wandb.log(wandb_log_dict)

    # 5. Extrapolation Test
    print("\n" + "="*50)
    print("RUNNING EXTRAPOLATION TEST")
    print("="*50)
    test_lengths = [512, 1024, 2048]
    extrap_results = {}
    
    model.eval()
    with torch.no_grad():
        for length in test_lengths:
            print(f"Evaluating on L_test = {length}...")
            try:
                # Load validation data for this specific sequence length
                # Keep batch size same
                from dataset import WikiTextDataset
                from torch.utils.data import DataLoader
                test_val_dataset = WikiTextDataset(split="validation", context_length=length)
                test_val_loader = DataLoader(test_val_dataset, batch_size=batch_size, shuffle=False)
                
                test_loss = 0
                test_steps = 0
                for x, y in test_val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    test_loss += loss.item()
                    test_steps += 1
                    if args.max_steps is not None and test_steps >= args.max_steps:
                        break
                    
                avg_test_loss = test_loss / test_steps
                perp = calculate_perplexity(torch.tensor(avg_test_loss))
                extrap_results[length] = (avg_test_loss, perp)
                print(f"L_test = {length} | Loss: {avg_test_loss:.4f} | Perplexity: {perp:.2f}")
            except Exception as e:
                print(f"Failed to evaluate on L_test = {length}: {e}")
                extrap_results[length] = (float('nan'), float('nan'))
                
    print("\n" + "="*50)
    print("FINAL EXTRAPOLATION SUMMARY")
    print("="*50)
    print(f"{'Context Length':<15} | {'Val Loss':<10} | {'Val Perplexity':<15}")
    print("-"*50)
    for length, (loss, perp) in extrap_results.items():
        print(f"{length:<15} | {loss:<10.4f} | {perp:<15.2f}")
        if args.wandb and not math.isnan(loss):
            wandb.log({f"extrap_val_loss_{length}": loss, f"extrap_val_perp_{length}": perp})
    print("="*50)
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
