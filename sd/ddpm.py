import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        n_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                n_training_steps,
                dtype=torch.float32,
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_hat = torch.cumprod(
            self.alphas, dim=0
        )  # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.n_training_steps = n_training_steps
        self.timesteps = torch.from_numpy(
            np.arange(0, n_training_steps)[::-1].copy()
        )  # [999, 998, 997, ..., 3, 2, 1]

    def set_inference_steps(self, n_inference_steps: int = 50) -> None:
        self.n_inference_steps = n_inference_steps

        self.step_ratio = self.n_training_steps // n_inference_steps  # 1000 // 50 = 20
        self.timesteps = torch.from_numpy(
            (np.arange(0, n_inference_steps) * self.step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )  # [999, 979, 959, ..., 60, 40, 20, 0]

    def step(
        self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_hat_t = self.alphas_hat[t]
        alpha_hat_t_prev = self.alphas_hat[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_hat_t
        beta_prod_t_prev = 1 - alpha_hat_t_prev

        curr_alpha_t = alpha_hat_t / alpha_hat_t_prev
        curr_beta_t = 1 - curr_alpha_t

        # Compute the predicted original sample, $\hat{x_{0}} ≈ x_{0}$,
        # using formula (15) from DDPM paper
        pred_orig_sample = (
            latents - beta_prod_t**0.5 * model_output
        ) / alpha_hat_t**0.5

        # Now that we have the predicted original sample, $\hat{x_{0}} ≈ x_{0}$,
        # we can compute the coefficients of the mean (see formula (7) of DDPM paper)
        # for `pred_orig_sample`, $x_{0}$ and current sample `x_t`, $x_{t}$
        pred_orig_sample_coef = (alpha_hat_t_prev**0.5 * curr_beta_t) / beta_prod_t
        curr_sample_coef = (curr_alpha_t**0.5 * beta_prod_t_prev) / beta_prod_t

        # Compute the predicted previous sample mean according to forumla (7) of DDPM paper
        pred_prev_sample_mean = (
            pred_orig_sample_coef * pred_orig_sample + curr_sample_coef * latents
        )

        std = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            std = self._get_variance(t) ** 0.5  # sqrt of variance = standard deviation

            # N(0, 1) -> N(mu, sigma ** 2)
            # X = mu + sigma * Z where Z ~ N(0, 1)
            pred_prev_sample = pred_prev_sample_mean + std * noise
        else:
            pred_prev_sample = pred_prev_sample_mean

        return pred_prev_sample

    def add_noise(
        self, latents: torch.FloatTensor, timestep: torch.IntTensor
    ) -> torch.FloatTensor:
        alphas_hat = self.alphas_hat.to(device=latents.device, dtype=latents.dtype)
        timestep = timestep.to(device=latents.device)

        sqrt_alpha_hat = alphas_hat[timestep] ** 0.5  # this is the mean
        sqrt_alpha_hat = sqrt_alpha_hat.flatten()

        while len(sqrt_alpha_hat.shape) < len(latents.shape):
            sqrt_alpha_hat = sqrt_alpha_hat.unsqueeze(-1)

        sqrt_one_minus_alpha_hat = (
            1 - alphas_hat[timestep]
        ) ** 0.5  # sqrt because we want the standard deviation, not the variance
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.flatten()

        while len(sqrt_one_minus_alpha_hat.shape) < len(latents.shape):
            sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.unsqueeze(-1)

        # According to equation (4) of DDPM paper
        # Z = N(0, 1) -> X = N(mean, variance)
        # X = mean + standard deviation * Z
        noise = torch.randn(
            latents.shape,
            generator=self.generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = (sqrt_alpha_hat * latents) + (sqrt_one_minus_alpha_hat * noise)

        return latents

    def set_strength(self, strength: float = 1.0) -> None:
        assert strength >= 0.0 and strength <= 1.0  # stength must be between 0 and 1
        # If strength is set to 0.8, it means we skip 20% of the inference steps (i.e. 10 steps)
        # and that the model will have less freedom to change the image because it is now
        # starting at step 10 instead of step 0 (`start_step` = 50 - 40 = 10). In summary,
        # the higher the strength, the more freedom the model has to change the image.
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.step_ratio

        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_hat_t = self.alphas_hat[timestep]
        alpha_hat_t_prev = self.alphas_hat[prev_t] if prev_t >= 0 else self.one

        curr_alpha_t = alpha_hat_t / alpha_hat_t_prev
        curr_beta_t = 1 - curr_alpha_t

        # Computed variance according to formula (7) of DDPM paper
        var = (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * curr_beta_t
        var = torch.clamp(var, min=1e-20)

        return var
