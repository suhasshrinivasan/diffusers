# %%
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import KarrasDiffusionSchedulers

import torch
from torch import nn
from ...models import AutoencoderKL, UNet2DConditionModel
from .safety_checker import StableDiffusionSafetyChecker

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker

#%%
class ConditionEncoder(nn.Linear):  # keep it linear for now
    def forward(self, x):
        return super().forward(x).unsqueeze(1) # add a new dimension mimicking seq len of 1

# %%
class ConditionalLatentDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        conditioning_encoder: Optional[ConditionEncoder] = None,
    ):
        super().__init__()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.register_modules(
            vae=vae,
            conditioning_encoder=conditioning_encoder,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(
            vae_scale_factor=self.vae_scale_factor,
            safety_checker=None,
            requires_safety_checker=False,
        )

    def encode_condition(self, cond: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode the condition (e.g., text) to latents.
        """
        if self.conditioning_encoder is not None:
            cond = self.conditioning_encoder(cond)
        return cond

    def decode_latents(self, latents: torch.FloatTensor) -> torch.FloatTensor:
        """
        Decode the latents back to image space.
        """
        latents /= self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        # TODO: what is the purpose of this?
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # TODO: need to figure out what's generator
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # TODO: FIGURE properties
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps


    @torch.no_grad()
    def __call__():
        raise NotImplementedError
# %%



