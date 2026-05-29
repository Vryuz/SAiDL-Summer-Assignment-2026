import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel, AutoencoderKL, DDPMScheduler
from dataset import get_dataloader
from tqdm.auto import tqdm

class DifficultyPredictor(nn.Module):
    """
    A lightweight spatial predictor network that takes intermediate DiT features
    (e.g., from the middle layer) and outputs a spatial difficulty mask M in [0, 1].
    """
    def __init__(self, in_channels, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        # Simple ConvNet over the un-patched feature map
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # M in [0, 1]
        )

    def forward(self, hidden_states, spatial_dim):
        # hidden_states: (B, N, C) where N = (H/p * W/p)
        # Reshape to (B, C, H/p, W/p)
        B, N, C = hidden_states.shape
        H = W = spatial_dim
        x = hidden_states.transpose(1, 2).view(B, C, H, W)
        
        mask = self.net(x) # (B, 1, H, W)
        return mask

def train_predictor():
    """
    Trains a lightweight DifficultyPredictor network to predict a spatial difficulty mask.
    The predictor is supervised using the discrepancy between an intermediate x0 prediction 
    and the final ground-truth x0 as a proxy for "how hard" each region is to denoise.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load frozen DiT and VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).to(device)
    vae.requires_grad_(False)
    
    model = DiTTransformer2DModel.from_pretrained("models/dit-baseline").to(device)
    model.requires_grad_(False)
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 2. Init Predictor
    # DiT-B hidden size is 768
    predictor = DifficultyPredictor(in_channels=768, patch_size=2).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    dataloader = get_dataloader(batch_size=8, num_workers=0)
    
    # 3. Training Loop
    # We simulate an intermediate timestep t=500. We predict x0 from t=500.
    # We compare predicted x0 to ground truth x0 to get the difficulty mask.
    # We extract hidden states from DiT at t=500 to feed the predictor.
    
    predictor.train()
    epochs = 5
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader)
        progress_bar.set_description(f"Predictor Epoch {epoch}")
        
        for batch in progress_bar:
            images = batch.to(device)
            
            with torch.no_grad():
                # GT x0 latents
                x0 = vae.encode(images).latent_dist.sample() * 0.18215
                bsz = x0.shape[0]
                
                # Sample noise and add up to t=500
                noise = torch.randn_like(x0)
                t_mid = torch.full((bsz,), 500, device=device, dtype=torch.long)
                xt = scheduler.add_noise(x0, noise, t_mid)
                class_labels = torch.zeros(bsz, dtype=torch.long, device=device)
                
                # Get DiT predictions and intermediate hidden states
                # Diffusers DiT doesn't easily return intermediate activations without hooks.
                # Since we just need ANY intermediate feature, we can use the final output feature before projection.
                # Actually, DiTTransformer2DModel outputs the un-projected transformer hidden states if return_dict=True (but it's buried).
                # We will register a forward hook to get the output of the 6th block.
                
                intermediate_features = []
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        intermediate_features.append(output[0])
                    else:
                        intermediate_features.append(output)
                    
                hook = model.transformer_blocks[6].register_forward_hook(hook_fn)
                
                model_output = model(xt, t_mid, class_labels=class_labels).sample
                noise_pred, _ = torch.chunk(model_output, 2, dim=1)
                
                hook.remove()
                
                # Compute predicted x0 (using DDPM step formulation or simply x0 = (xt - sqrt(1-a) * noise_pred) / sqrt(a))
                alpha_prod_t = scheduler.alphas_cumprod[t_mid].view(-1, 1, 1, 1)
                beta_prod_t = 1 - alpha_prod_t
                pred_x0 = (xt - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                
                # Calculate GT Error map per patch
                # x0 is (B, 4, 32, 32). Patch size is 2, so we pool the error map to 16x16
                error_map = F.mse_loss(pred_x0, x0, reduction='none').mean(dim=1, keepdim=True) # (B, 1, 32, 32)
                # Max pool to get patch-level difficulty
                patch_error = F.max_pool2d(error_map, kernel_size=2, stride=2) # (B, 1, 16, 16)
                
                # Normalize error map to [0, 1] per batch for supervision
                b_min = patch_error.view(bsz, -1).min(dim=1)[0].view(bsz, 1, 1, 1)
                b_max = patch_error.view(bsz, -1).max(dim=1)[0].view(bsz, 1, 1, 1)
                target_mask = (patch_error - b_min) / (b_max - b_min + 1e-8)
            
            # Forward Predictor
            h = intermediate_features[0] # (B, 256, 768)
            predicted_mask = predictor(h, spatial_dim=16) # (B, 1, 16, 16)
            
            loss = F.mse_loss(predicted_mask, target_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"pred_loss": loss.item()})
            
    # Save Predictor
    os.makedirs("models/predictor", exist_ok=True)
    torch.save(predictor.state_dict(), "models/predictor/weights.pt")

if __name__ == "__main__":
    train_predictor()
