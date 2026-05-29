import torch
from diffusers import DiTTransformer2DModel, AutoencoderKL, DDIMScheduler, DDPMScheduler
from PIL import Image
import numpy as np
import time
import os

@torch.no_grad()
def sample_with_global_refinement(model=None, vae=None, scheduler=None, refine_t=200, device="cuda"):
    """
    Implements Global Cyclic Refinement sampling. Generates an initial image, 
    re-injects noise up to refine_t, and denoises back to correct residual errors.
    """
    if model is None or vae is None or scheduler is None:
        model = DiTTransformer2DModel.from_pretrained("models/dit-baseline").to(device)
        model.eval()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.eval()
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    
    model.requires_grad_(False)
    
    scheduler.set_timesteps(50) # 50 steps for fast generation
    
    bsz = 1
    latents = torch.randn((bsz, 4, 32, 32), device=device)
    class_labels = torch.zeros(bsz, dtype=torch.long, device=device)
    
    start_time = time.time()
    
    # 1. Standard Sampling Pass (t=T to t=0)
    for t in scheduler.timesteps:
        # Predict noise
        t_batch = torch.tensor([t.item() if torch.is_tensor(t) else t], dtype=torch.long, device=device)
        model_output = model(latents, t_batch, class_labels=class_labels).sample
        noise_pred, _ = torch.chunk(model_output, 2, dim=1)
        
        # Step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    intermediate_latents = latents.clone()
    
    # 2. Cyclic Refinement
    # Inject noise back to equivalent of t=200 in original 1000-step scale.
    # In DDIM 50-step scale, t=200 corresponds to index 10 (since 1000/50 = 20, 200/20 = 10 steps from end)
    # Let's find the exact timestep in the scheduler
    refine_t = 200
    refine_t_tensor = torch.tensor([refine_t], device=device)
    noise = torch.randn_like(latents)
    
    noisy_latents = scheduler.add_noise(latents, noise, refine_t_tensor)
    
    # Denoise back from t=200 to 0
    # Create a new timestep array for the refinement phase
    refine_timesteps = [t for t in scheduler.timesteps if t <= refine_t]
    
    latents = noisy_latents
    for t in refine_timesteps:
        t_batch = torch.tensor([t.item() if torch.is_tensor(t) else t], dtype=torch.long, device=device)
        model_output = model(latents, t_batch, class_labels=class_labels).sample
        noise_pred, _ = torch.chunk(model_output, 2, dim=1)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    generation_time = time.time() - start_time
    
    # Decode final latents
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image), generation_time

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sampling on {device}...")
    try:
        img, gen_time = sample_with_global_refinement(device=device)
        os.makedirs("results/images", exist_ok=True)
        img.save("results/images/sample_global_refinement.png")
        print(f"Generated globally refined image in {gen_time:.2f} seconds.")
    except Exception as e:
        print(f"Error (likely model not trained yet): {e}")
