import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DiTTransformer2DModel, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from dataset import get_dataloader

def train_baseline():
    """
    Trains a baseline DiT (Diffusion Transformer) model on landscape images.
    Uses a frozen VAE to encode images to latents, then trains DiT-B/2 with DDPM noise scheduling.
    """
    torch.manual_seed(42)
    accelerator = Accelerator(mixed_precision="fp16")
    
    # 1. Load VAE and freeze it
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    
    # 2. Initialize DiT-B/8
    # 256x256 image -> 32x32 latent with patch_size=2 -> 16x16 patches.
    model = DiTTransformer2DModel(
        sample_size=32,
        num_layers=12,         # DiT-B
        patch_size=2,          # DiT-B/2 (closer to standard than 8 for 32x32 latent)
        attention_head_dim=64, # DiT-B
        num_attention_heads=12,# DiT-B
        in_channels=4,
        out_channels=8,        # outputs mean and variance
        upcast_attention=True,
    )
    
    # 3. Scheduler & Optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 4. Dataset
    dataloader = get_dataloader(batch_size=8, num_workers=0)
    
    # 5. Prepare with accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    vae.to(accelerator.device)
    
    # Calculate total steps
    num_epochs = 10
    total_steps = len(dataloader) * num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    # 6. Training Loop
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for batch in progress_bar:
            # batch is images [-1, 1]
            images = batch.to(accelerator.device)
            
            with torch.no_grad():
                # VAE encodes to latents, scale by 0.18215 (standard SD scaling)
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            # DiT takes class_labels, but assignment says "no class conditioning".
            # We pass a dummy tensor or None if supported. Diffusers DiT requires class_labels by default.
            # We will use a dummy class label 0 for unconditional.
            class_labels = torch.zeros(bsz, dtype=torch.long, device=latents.device)
            model_pred = model(noisy_latents, timesteps, class_labels=class_labels).sample
            
            # DiT predicts both noise and variance (out_channels=8). We only need noise (first 4 channels) for loss
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)
            
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
            
    # Save base model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs("models", exist_ok=True)
        model = accelerator.unwrap_model(model)
        model.save_pretrained("models/dit-baseline")
        print("Baseline DiT saved successfully!")

if __name__ == "__main__":
    train_baseline()
