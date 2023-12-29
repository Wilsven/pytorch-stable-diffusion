from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from ddpm import DDPMSampler


HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = HEIGHT // 8
LATENT_WIDTH = WIDTH // 8


def generate(
    prompt: str,
    uncond_prompt: str,  # also known as the negative prompt (can be empty string)
    input_image: Optional[Image.Image] = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models: dict = {},
    seed: Optional[int] = None,
    device: Optional[str] = None,
    idle_device: Optional[str] = None,
    tokenizer: Optional[CLIPTokenizer] = None,
):
    with torch.no_grad():
        if not (0.0 < strength < 1.0):
            raise ValueError("strength must be between 0 and 1.")
        if idle_device is not None:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Set seed for reproducibility
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]  # load CLIP pre-trained model
        clip.to(device)

        # Classifier Free Guidance (combine output)
        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # Convert token input_ids into tensors -> (b, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Convert tensors into embeddings: (b, seq_len) -> (b, seq_len, n_embed)
            cond_context = clip(cond_tokens)

            # Convert the unconditioned prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # Convert unconditioned token input_ids into tensors -> (b, seq_len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # Convert unconditioned tensors into embeddings: (b, seq_len) -> (b, seq_len, n_embed)
            uncond_context = clip(uncond_tokens)

            # (b, seq_len, n_embed) + (b, seq_len, n_embed) -> (2 * b, seq_len, n_embed) -> (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        # Use prompt only if decide not to use CFG
        else:
            # Convert the prompt into tokens using the tokenizer
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # Convert token input_ids into tensors -> (b, seq_len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # Convert tensors into embeddings: (b, seq_len) -> (b, seq_len, n_embed) -> (1, 77, 768)
            context = clip(tokens)

        # Useful if you have limited GPU so you can offload model onto CPU after using them
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name} not found.")

        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)  # (1, 4, 64, 64)

        # If image is provided, we are doing image-to-image generation
        if input_image is not None:
            encoder = models["encoder"]
            encoder.to(device)

            input_image = input_image.resize(WIDTH, HEIGHT)
            input_image_array = np.array(input_image)
            input_image_tensor = torch.tensor(
                input_image_array, dtype=torch.float32
            )  # (h, w, c)

            # Rescale pixels from between 0 and 255 to -1 and 1
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (h, w, c) -> (1, h, w, c)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (1, h, w, c) -> (1, c, h, w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latent_shape, generator=generator, device=device
            )
            # Run image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            # `strength` parameter is how much weight we want to give to
            # the input image when doing image-to-image generation. The larger
            # the strength, the less noised the image is and the less freedom
            # the model has to alter the input image
            sampler.set_strength(strength=strength)
            timestep = sampler.timesteps[0]
            latents = sampler.add_noise(latents, timestep)

            to_idle(encoder)

        else:
            # Start with noise, N(0, I) if we are doing text-to-image
            noise = torch.randn(latent_shape, generator=generator, device=device)
            latents = noise

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for _, ts in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(ts).to(device)

            # (b, 4, h / 8, w / 8) -> (b, 4, 64, 64)
            model_input = latents

            if do_cfg:
                # (b, 4, 64, 64) -> (b * 2, 4, 64, 64)
                model_input = model_input.repeat(2, 1, 1, 1)

            # `model_output` is the predicted noise by the U-net
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # (b * 2, 4, h / 8, w / 8) -> (b * 2, 4, 64, 64) -> 2 x (b, 4, 64, 64)
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                # Combine output according to Classifier Free Guidance formula
                model_output = cfg_scale * (output_cond - output_uncond) + output_cond

            # Remove noise predicted by the U-net
            latents = sampler.step(ts, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (b, 3, h, w) -> (b, h, w, 3)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]


def rescale(
    x: torch.Tensor, old_range: tuple[int], new_range: tuple[int], clamp: bool = False
) -> torch.Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep: int) -> torch.Tensor:
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160) because of unsqueeze operation
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # (1, 160) -> (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
