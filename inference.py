"""
jitstandalone/inference.py

End-to-end inference script for the JIT pipeline.
- Loads the pre-configured schedulers.
- Tokenizes a sample prompt.
- Generates a latent noise tensor.
- Executes the T5, CLIP, FLUX, and VAE models in sequence.
- Saves the final image to disk.
"""
import torch
import numpy as np
from PIL import Image
from transformers import T5Tokenizer, CLIPTokenizer

from model_loader import load_pipeline

def run_inference(prompt: str, output_path: str = "jitstandalone/output.png", quant_config: str = None, cpu_pool_gb: int = 4, gpu_pool_gb: int = 6):
    """
    Runs the full JIT inference pipeline with sequential text encoders.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load all model schedulers and allocator
    cpu_pool_size = cpu_pool_gb * 1024 * 1024 * 1024 
    gpu_pool_size = gpu_pool_gb * 1024 * 1024 * 1024 
    schedulers, allocator = load_pipeline(device, quant_config=quant_config, cpu_pool_size=cpu_pool_size, total_vram_limit=gpu_pool_size)
    t5_scheduler = schedulers["t5"]
    clip_scheduler = schedulers["clip"]
    flux_scheduler = schedulers["flux"]
    vae_scheduler = schedulers["vae"]



    t5_tokenizer = T5Tokenizer.from_pretrained("../../ComfyUI/jitloader/t5_tokenizer")
    clip_tokenizer = CLIPTokenizer.from_pretrained("../../ComfyUI/jitloader/clip_tokenizer")
    
    t5_tokens = t5_tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(device)
    clip_tokens = clip_tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(device)

    del t5_tokenizer
    del clip_tokenizer

    # 3. Run Text Encoders in sequence
    print("Running T5 encoder...")
    t5_embeddings = t5_scheduler.run_encoder_inference(t5_tokens)
    print("T5 encoder finished.")
    print(t5_embeddings)
    return

    print("Running CLIP encoder...")
    clip_embeddings = clip_scheduler.run_encoder_inference(clip_tokens)
    print("CLIP encoder finished.")

    # 4. Prepare inputs for FLUX
    context = torch.cat([t5_embeddings, clip_embeddings], dim=1)
    # Create a dummy pooled vector for now
    y_shape = (1, t5_scheduler.blueprint.config.d_model + clip_scheduler.blueprint.config.hidden_size)
    y_size_bytes = y_shape[0] * y_shape[1] * 4
    y_buffer = allocator.allocate(y_size_bytes, device)
    y = y_buffer.view(torch.float32).reshape(y_shape)
    y.normal_()

    # Latent noise
    height, width = 1024, 1024
    patch_size = flux_scheduler.blueprint.patch_size
    latent_height = height // patch_size
    latent_width = width // patch_size
    latent_channels = flux_scheduler.blueprint.in_channels
    
    latent_shape = (1, latent_channels, latent_height, latent_width)
    n_elements = 1 * latent_channels * latent_height * latent_width
    latent_size_bytes = n_elements * 4
    latents_buffer = allocator.allocate(latent_size_bytes, device)
    latents = latents_buffer.view(torch.float32).reshape(latent_shape)
    latents.normal_()

    timestep = torch.tensor([999], device=device)

    # 5. Run the FLUX model
    print("Running FLUX model...")
    output_latents = flux_scheduler.run_inference(latents, timestep, context, y)
    print("FLUX model finished.")

    # 6. Decode the latents with the VAE
    print("Running VAE decoder...")
    image_tensor = vae_scheduler.run_decoder_inference(output_latents)
    print("VAE decoder finished.")

    # 7. Save the output image
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = (image_tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    image = Image.fromarray(image_tensor)
    image.save(output_path)
    print(f"Inference complete. Output saved to {output_path}")

if __name__ == "__main__":
    sample_prompt = "A photograph of a majestic lion in the savanna."
    run_inference(sample_prompt)