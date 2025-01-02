from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import Field
import torch
import pygame
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from PIL import Image
from huggingface_hub import snapshot_download as hf_snapshot_download

from .models.diffusion.diffusion_sampler import build_sigmas
from .agent import Agent
from .game.play_env import PlayEnv
from .data import collate_segments_to_batch
from .data.dataset import Dataset
from .data.batch_sampler import BatchSampler
from .envs.world_model_env import WorldModelEnv
from .csgo.action_processing import CSGOAction
from .csgo.keymap import CSGO_KEYMAP

from dweam import Game, get_cache_dir


OmegaConf.register_new_resolver("eval", eval, replace=True)


def snapshot_download(**kwargs) -> Path:
    base_cache_dir = get_cache_dir()
    cache_dir = base_cache_dir / 'huggingface-data'
    path = hf_snapshot_download(cache_dir=str(cache_dir), **kwargs)
    return Path(path)


def prepare_env(cfg: DictConfig, device: torch.device, model_name: str) -> tuple[PlayEnv, dict]:
    path_hf = snapshot_download(repo_id="DereWah/diamond-mariokart64", allow_patterns=f"{model_name}/*")
    
    path_ckpt = path_hf / f"{model_name}/model/{model_name}.pt"
    spawn_dir = path_hf / f"{model_name}/spawn"

    assert cfg.env.train.id == model_name
    num_actions = cfg.env.num_actions

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)
    
    # World model environment
    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser, 
        agent.upsampler,
        agent.rew_end_model,
        spawn_dir,
        1,
        sl,
        wm_env_cfg,
        return_denoising_trajectory=True
    )

    # TODO do we do this?
    # if device.type == "cuda":  # and args.compile:
    #     print("Compiling models...")
    #     wm_env.predict_next_obs = torch.compile(wm_env.predict_next_obs, mode="reduce-overhead")
    #     wm_env.upsample_next_obs = torch.compile(wm_env.upsample_next_obs, mode="reduce-overhead")


    play_env = PlayEnv(
        agent,
        wm_env,
        recording_mode=False,
        store_denoising_trajectory=False,
        store_original_obs=False,
    )

    return play_env, CSGO_KEYMAP


class MarioKartGame(Game):
    class Params(Game.Params):
        denoising_steps: int = Field(default=10, description="Less steps means faster generation, but less accurate")
        context_window: int = Field(default=4, le=4, ge=1, description="Number of frames to remember when generating the next frame")

    def on_params_update(self, new_params: Params) -> None:
        super().on_params_update(new_params)

        diffusion_sampler = self.env.env.sampler_next_obs
        cfg = diffusion_sampler.cfg
        cfg.num_steps_denoising = new_params.denoising_steps
        diffusion_sampler.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, diffusion_sampler.denoiser.device)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        with initialize(version_base="1.3", config_path="config"):
            cfg = compose(config_name="trainer")

        device_id = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"Using device", device_id=device_id)
        self.device = torch.device(device_id)

        self.env, self.keymap = prepare_env(
            cfg=cfg,
            device=self.device,
            model_name="lr",
        )
        self.env.reset()

        # Set size
        self.height, self.width = (cfg.env.train.size,) * 2 if isinstance(cfg.env.train.size, int) else cfg.env.train.size

    def draw_game(self, obs) -> pygame.Surface:
        assert obs.ndim == 4 and obs.size(0) == 1
        img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        pygame_image = np.array(img.resize((self.height, self.width), resample=Image.NEAREST)).transpose((1, 0, 2))
        return pygame.surfarray.make_surface(pygame_image)

    def step(self) -> pygame.Surface:
        if pygame.K_LEFT in self.keys_pressed:
            steering_value = -1.0
        elif pygame.K_RIGHT in self.keys_pressed:
            steering_value = 1.0
        else:
            steering_value = 0.0

        action = CSGOAction(
            keys=[k for k in self.keys_pressed if k in CSGO_KEYMAP],
            steering_value=steering_value,
        )

        # Apply context window by zeroing out old observations/actions
        wm_env = self.env.env
        context_size = wm_env.sampler_next_obs.denoiser.cfg.inner_model.num_steps_conditioning
        window_size = min(self.params.context_window, context_size)
        
        # Zero out observations/actions beyond the context window
        if window_size < context_size:
            wm_env.obs_buffer[:, :context_size-window_size] = 0
            wm_env.act_buffer[:, :context_size-window_size] = 0
            if wm_env.obs_full_res_buffer is not None:
                wm_env.obs_full_res_buffer[:, :context_size-window_size] = 0

        # Step the environment with CSGO action
        next_obs, rew, end, trunc, info = self.env.step(action)

        # Reset if episode ended
        if end or trunc:
            self.env.reset()

        return self.draw_game(next_obs)

    def on_key_down(self, key: int) -> None:
        # Handle special keys
        if key == pygame.K_RETURN:
            self.env.reset()
        elif key == pygame.K_PERIOD:
            self.paused = not self.paused
        elif key == pygame.K_e and self.paused:
            self.do_one_step()

    def stop(self) -> None:
        super().stop()
        # TODO deload the model from GPU memory
