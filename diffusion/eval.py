import os
import torch
import subprocess
from PIL import Image
from diffusers import AutoencoderKL
from dataset import get_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as TF
from tqdm.auto import tqdm

# Import sampling functions 
from global_refine import sample_with_global_refinement
from racd import sample_racd

def generate_samples(method_name, model=None, vae=None, scheduler=None, predictor=None, num_samples=100, tau=0.5, device="cuda"):
    """
    Generates num_samples images using the specified method and saves them to a temp folder.
    """
    save_dir = f"results/eval_samples/{method_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    total_time = 0
    print(f"Generating {num_samples} samples using {method_name}...")
    
    for i in tqdm(range(num_samples)):
        if method_name == "global_refine":
            img, gen_time = sample_with_global_refinement(model=model, vae=vae, scheduler=scheduler, device=device)
        elif method_name == "racd":
            img, mask, gen_time = sample_racd(model=model, vae=vae, scheduler=scheduler, predictor=predictor, tau=tau, device=device)
        else:
            raise ValueError("Method not supported here.")
            
        img.save(f"{save_dir}/{i}.png")
        total_time += gen_time
        
    avg_time = total_time / num_samples
    return save_dir, avg_time

def evaluate_metrics_standalone(real_dir, gen_dir, device="cuda"):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    transform = TF.Compose([TF.Resize(256), TF.CenterCrop(256), TF.ToTensor()])
    
    files_real = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.png')]
    files_gen = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith('.png')]
    
    def process(files, is_real):
        batch = []
        for f in files:
            batch.append(transform(Image.open(f).convert("RGB")))
            if len(batch) >= 16:
                fid.update(torch.stack(batch).to(device), real=is_real)
                batch = []
        if batch: 
            fid.update(torch.stack(batch).to(device), real=is_real)
            
    process(files_real, True)
    process(files_gen, False)
    
    result = subprocess.run(f"clip-mmd {real_dir} {gen_dir}", shell=True, capture_output=True, text=True)
    cmmd_out = result.stdout
    if result.returncode != 0:
        print(f"Warning: clip-mmd failed. Stderr: {result.stderr}")
        cmmd_out = "CMMD failed to compute."
    return fid.compute().item(), cmmd_out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Preparing Real Images for CMMD Comparison...")
    real_images_dir = "results/eval_samples/real"
    os.makedirs(real_images_dir, exist_ok=True)

    dataloader = get_dataloader(batch_size=1)
    for i, batch in enumerate(dataloader):
        if i >= 100: break
        image = batch.to(device)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        Image.fromarray((image * 255).astype("uint8")).save(f"{real_images_dir}/{i}.png")

    print("\nLoading models for evaluation...")
    from diffusers import DiTTransformer2DModel, AutoencoderKL, DDIMScheduler
    from predictor import DifficultyPredictor
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    model = DiTTransformer2DModel.from_pretrained("models/dit-baseline").to(device)
    model.eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    predictor = DifficultyPredictor().to(device)
    if os.path.exists("models/predictor/weights.pt"):
        predictor.load_state_dict(torch.load("models/predictor/weights.pt", map_location=device, weights_only=True))
    predictor.eval()

    print("\n2. Generating RACD Samples...")
    racd_dir, racd_time = generate_samples("racd", model=model, vae=vae, scheduler=scheduler, predictor=predictor, num_samples=100, tau=0.5, device=device)

    print("\n3. Generating Global Refinement Samples...")
    global_dir, global_time = generate_samples("global_refine", model=model, vae=vae, scheduler=scheduler, predictor=predictor, num_samples=100, device=device)

    print("\n================ FINAL EVALUATION ================")
    print(f"Average Generation Time (RACD)          : {racd_time:.3f} seconds/image")
    print(f"Average Generation Time (Global Refine) : {global_time:.3f} seconds/image")
    print(f"Compute Time Saved by RACD              : {((global_time - racd_time)/global_time)*100:.1f}%\n")

    global_fid, global_cmmd = evaluate_metrics_standalone(real_images_dir, global_dir, device=device)
    print(f"Global Refine FID: {global_fid:.2f}\nGlobal Refine CMMD:\n{global_cmmd}")

    racd_fid, racd_cmmd = evaluate_metrics_standalone(real_images_dir, racd_dir, device=device)
    print(f"RACD FID: {racd_fid:.2f}\nRACD CMMD:\n{racd_cmmd}")
    print("==================================================")
