import torch
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel, AutoencoderKL, DDIMScheduler
from predictor import DifficultyPredictor
from PIL import Image
import numpy as np
import time
import os

@torch.no_grad()
def sample_racd(model=None, vae=None, scheduler=None, predictor=None, tau=0.5, refine_t=200, device="cuda"):
    """
    Implements Regionally-Adaptive Cyclic Diffusion (RACD).
    Uses a spatial difficulty predictor to identify hard patches and only applies 
    cyclic refinement (noise re-injection) to those specific regions, saving compute.
    """
    if model is None or vae is None or scheduler is None or predictor is None:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.requires_grad_(False)
        
        model = DiTTransformer2DModel.from_pretrained("models/dit-baseline").to(device)
        model.requires_grad_(False)
        
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        
        predictor = DifficultyPredictor().to(device)
        predictor.load_state_dict(torch.load("models/predictor/weights.pt", map_location=device, weights_only=True))
        predictor.requires_grad_(False)
    predictor.eval()
    
    scheduler.set_timesteps(50) # 50 steps for fast generation
    
    bsz = 1
    latents = torch.randn((bsz, 4, 32, 32), device=device)
    class_labels = torch.zeros(bsz, dtype=torch.long, device=device)
    
    # Hook for predictor
    intermediate_features = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            intermediate_features.append(output[0])
        else:
            intermediate_features.append(output)
    
    start_time = time.time()
    
    # 2. Standard Sampling Pass (t=T to t=0)
    capture_t_index = len(scheduler.timesteps) // 2 # capture at halfway point
    
    for i, t in enumerate(scheduler.timesteps):
        if i == capture_t_index:
            hook = model.transformer_blocks[6].register_forward_hook(hook_fn)
            
        t_batch = torch.tensor([t.item() if torch.is_tensor(t) else t], dtype=torch.long, device=device)
        model_output = model(latents, t_batch, class_labels=class_labels).sample
        
        if i == capture_t_index:
            hook.remove()
            
        noise_pred, _ = torch.chunk(model_output, 2, dim=1)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 3. Predict Difficulty Mask
    h = intermediate_features[0] # (1, 256, 768)
    difficulty_map = predictor(h, spatial_dim=16) # (1, 1, 16, 16)
    
    # Threshold the map
    binary_mask = (difficulty_map > tau).float()
    
    # Upsample mask to latent resolution (32x32)
    binary_mask_latent = F.interpolate(binary_mask, size=(32, 32), mode="nearest") # (1, 1, 32, 32)
    
    # 4. Regionally-Adaptive Cyclic Refinement
    refine_t = 200
    refine_t_tensor = torch.tensor([refine_t], device=device)
    noise = torch.randn_like(latents)
    
    # Add noise ONLY to the regions where mask == 1
    global_noisy_latents = scheduler.add_noise(latents, noise, refine_t_tensor)
    latents = latents * (1 - binary_mask_latent) + global_noisy_latents * binary_mask_latent
    
    # Denoise back from t=200 to 0
    refine_timesteps = [t for t in scheduler.timesteps if t <= refine_t]
    
    for t in refine_timesteps:
        t_batch = torch.tensor([t.item() if torch.is_tensor(t) else t], dtype=torch.long, device=device)
        model_output = model(latents, t_batch, class_labels=class_labels).sample
        noise_pred, _ = torch.chunk(model_output, 2, dim=1)
        # We step the whole latent, but since clean areas had no noise, they stay mostly clean 
        # (while the noisy areas get refined). In a strict implementation, we could also force 
        # the clean areas to stay perfectly static, but blending through the network is smoother.
        next_latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents * (1 - binary_mask_latent) + next_latents * binary_mask_latent

    generation_time = time.time() - start_time
    
    # 5. Decode
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    
    mask_visual = (binary_mask_latent[0, 0].cpu().numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(image), Image.fromarray(mask_visual), generation_time

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sampling RACD on {device}...")
    try:
        img, mask_img, gen_time = sample_racd(tau=0.5, device=device)
        os.makedirs("results/images", exist_ok=True)
        img.save("results/images/sample_racd.png")
        mask_img.save("results/images/sample_racd_mask.png")
        print(f"Generated RACD image in {gen_time:.2f} seconds.")
    except Exception as e:
        print(f"Error: {e}")
