import distutils
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import isaacgym
import numpy as np
import torch
import yaml

try:
    import gym
    from gym import spaces
except ModuleNotFoundError:
    import gymnasium as gym
    from gymnasium import spaces

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments
from rl_games.common import env_configurations, vecenv

try:
    from aerial_gym.rl_training.rl_games.recording_types import (
        RECORD_TARGET_XYZ,
        THIRD_PERSON_PRESET,
    )
    from aerial_gym.rl_training.rl_games.recording_env_manager import RecordingEnvManager
    from aerial_gym.rl_training.rl_games.recording_warp_sensor import RecordingWarpSensor
except ModuleNotFoundError:
    # Support direct script execution from this directory when rl_training is not a package.
    from recording_types import (
        RECORD_TARGET_XYZ,
        THIRD_PERSON_PRESET,
    )
    from recording_env_manager import RecordingEnvManager
    from recording_warp_sensor import RecordingWarpSensor

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class RecordingComplete(Exception):
    pass


class EpisodeGifRecorder:
    SEGMENT_PALETTE = np.array(
        [
            [230, 57, 70],
            [29, 53, 87],
            [69, 123, 157],
            [168, 218, 220],
            [42, 157, 143],
            [233, 196, 106],
            [244, 162, 97],
            [231, 111, 81],
            [67, 170, 139],
            [112, 193, 179],
            [38, 70, 83],
            [152, 193, 217],
        ],
        dtype=np.uint8,
    )
    ROBOT_SEGMENT_COLORS = {
        25: np.array([255, 80, 80], dtype=np.uint8),   # body
        26: np.array([255, 180, 70], dtype=np.uint8),  # arm
        27: np.array([90, 190, 255], dtype=np.uint8),  # motor
    }

    def __init__(
        self,
        output_dir: Path,
        target_episodes: int,
        fps: int,
        record_env_id: int,
        include_segmentation: bool = True,
        max_record_steps: Optional[int] = None,
        gif_mode: str = "depth_seg_color",
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_episodes = target_episodes
        self.record_env_id = record_env_id
        self.include_segmentation = include_segmentation
        self.gif_mode = self._resolve_gif_mode(gif_mode, include_segmentation)
        self.max_record_steps = max_record_steps if (max_record_steps is not None and max_record_steps > 0) else None
        self.frame_duration_ms = max(1, int(1000 / max(1, fps)))
        self.episodes_saved = 0
        self.frames = []
        self.current_episode_steps = 0

    def capture(self, task_env, terminated, truncated):
        tensor_dict = self._get_tensor_dict(task_env)
        depth_tensor = tensor_dict.get("depth_range_pixels")
        if depth_tensor is None:
            raise RuntimeError(
                "No camera tensor found: 'depth_range_pixels' is missing. "
                "Please use a task/robot with camera enabled or use --camera_preset=third_person. "
                "If you run headless with Warp, avoid empty_env (no warp meshes) and use env_with_obstacles."
            )

        num_envs = int(depth_tensor.shape[0])
        if self.record_env_id < 0 or self.record_env_id >= num_envs:
            raise RuntimeError(
                f"--record_env_id={self.record_env_id} is out of range for num_envs={num_envs}."
            )

        frame = self._build_frame(tensor_dict)
        self.frames.append(frame)
        self.current_episode_steps += 1

        done = bool(
            (terminated[self.record_env_id] | truncated[self.record_env_id]).item()
        )
        reached_max_steps = (
            self.max_record_steps is not None and self.current_episode_steps >= self.max_record_steps
        )
        if done or reached_max_steps:
            reason = "done" if done else f"maxsteps{self.max_record_steps}"
            self._flush_episode(reason=reason)
            self.current_episode_steps = 0
            if self.episodes_saved >= self.target_episodes:
                raise RecordingComplete(
                    f"Saved {self.episodes_saved} episode(s) to {self.output_dir}."
                )

    def _flush_episode(self, reason: str = "done"):
        if not self.frames:
            return
        out_file = self.output_dir / f"episode_{self.episodes_saved:03d}_{reason}.gif"
        self.frames[0].save(
            out_file,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.frame_duration_ms,
            loop=0,
        )
        print(f"[play_record] Saved GIF: {out_file}")
        self.episodes_saved += 1
        self.frames = []

    def _get_tensor_dict(self, task_env) -> Dict[str, torch.Tensor]:
        if hasattr(task_env, "obs_dict"):
            return task_env.obs_dict
        if hasattr(task_env, "sim_env") and hasattr(task_env.sim_env, "global_tensor_dict"):
            return task_env.sim_env.global_tensor_dict
        raise RuntimeError(
            "Failed to locate observation tensors for recording in task environment."
        )

    def _resolve_gif_mode(self, gif_mode: str, include_segmentation: bool) -> str:
        mode = str(gif_mode).strip().lower()
        if mode in {"auto", ""}:
            mode = "depth_seg_color"
        allowed_modes = {"depth_seg_color", "depth_only", "seg_only"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Unsupported --gif_mode '{gif_mode}'. Supported values: {sorted(allowed_modes)}"
            )
        if not include_segmentation and mode == "depth_seg_color":
            return "depth_only"
        if not include_segmentation and mode == "seg_only":
            raise ValueError(
                "--gif_mode=seg_only requires segmentation. Remove --include_segmentation=False."
            )
        return mode

    def _requires_segmentation(self) -> bool:
        return self.gif_mode in {"depth_seg_color", "seg_only"}

    def _to_u8(self, image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        vmin = float(np.min(image))
        vmax = float(np.max(image))
        if vmax - vmin < 1e-8:
            return np.zeros_like(image, dtype=np.uint8)
        if vmin >= 0.0 and vmax <= 1.0:
            scaled = image * 255.0
        elif vmin >= -1.0 and vmax <= 1.0:
            scaled = (image + 1.0) * 127.5
        else:
            scaled = (image - vmin) / (vmax - vmin) * 255.0
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def _segmentation_to_rgb(self, segmentation_image: np.ndarray) -> np.ndarray:
        seg = np.nan_to_num(segmentation_image, nan=0.0, posinf=0.0, neginf=0.0)
        seg = np.rint(seg).astype(np.int32)
        seg_rgb = np.zeros(seg.shape + (3,), dtype=np.uint8)

        unique_labels = np.unique(seg)
        for label in unique_labels:
            if label in self.ROBOT_SEGMENT_COLORS:
                color = self.ROBOT_SEGMENT_COLORS[label]
            elif label <= 0:
                color = np.array([0, 0, 0], dtype=np.uint8)
            else:
                color = self.SEGMENT_PALETTE[int(label) % len(self.SEGMENT_PALETTE)]
            seg_rgb[seg == label] = color
        return seg_rgb

    def _build_frame(self, tensor_dict: Dict[str, torch.Tensor]):
        from PIL import Image

        env_idx = self.record_env_id
        depth = tensor_dict["depth_range_pixels"][env_idx, 0].detach().cpu().numpy()
        depth_u8 = self._to_u8(depth)
        depth_rgb = np.stack([depth_u8, depth_u8, depth_u8], axis=-1)

        seg_rgb = None
        if self._requires_segmentation():
            segmentation_tensor = tensor_dict.get("segmentation_pixels")
            if segmentation_tensor is None:
                raise RuntimeError(
                    f"GIF mode '{self.gif_mode}' requires segmentation tensor 'segmentation_pixels', "
                    "but it is missing. Use --gif_mode=depth_only or enable segmentation camera."
                )
            seg = segmentation_tensor[env_idx, 0].detach().cpu().numpy()
            seg_rgb = self._segmentation_to_rgb(seg)

        if self.gif_mode == "depth_only":
            frame = depth_rgb
        elif self.gif_mode == "seg_only":
            frame = seg_rgb
        else:
            frame = np.concatenate([depth_rgb, seg_rgb], axis=0)
        return Image.fromarray(frame.astype(np.uint8))


class RecordingTaskWrapper:
    def __init__(self, env, recorder: EpisodeGifRecorder):
        self.env = env
        self.recorder = recorder

    def _inject_record_target(self):
        target_position = getattr(self.env, "target_position", None)
        if not isinstance(target_position, torch.Tensor):
            return False
        if target_position.ndim < 2 or target_position.shape[-1] < 3:
            return False

        target_position[:, 0:3] = target_position.new_tensor(RECORD_TARGET_XYZ)
        if hasattr(self.env, "process_obs_for_task"):
            self.env.process_obs_for_task()
        return True

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._inject_record_target()
        return result

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        self._inject_record_target()
        self.recorder.capture(self.env, terminated, truncated)
        return observations, rewards, terminated, truncated, infos

    def __getattr__(self, name):
        return getattr(self.env, name)


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )
        return observations["observations"], rewards, dones, infos


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.Inf,
            np.ones(self.env.task_config.observation_space_dim) * np.Inf,
        )
        print(info["action_space"], info["observation_space"])
        return info


def get_args():
    def parse_bool_or_auto(x):
        if isinstance(x, bool):
            return x
        val = str(x).strip().lower()
        if val in {"auto", "none"}:
            return None
        return bool(distutils.util.strtobool(val))

    def parse_optional_int(x):
        val = str(x).strip().lower()
        if val in {"none", "null", "auto"}:
            return None
        ival = int(val)
        if ival <= 0:
            return None
        return ival

    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": True,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 1,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--task",
            "type": str,
            "default": None,
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": parse_bool_or_auto,
            "default": "auto",
            "help": "Choose whether to use warp or Isaac Gym rendering pipeline. Use 'auto' to follow task config.",
        },
        {
            "name": "--record_dir",
            "type": str,
            "default": "./recordings",
            "help": "Directory where GIF files are stored.",
        },
        {
            "name": "--record_episodes",
            "type": int,
            "default": 1,
            "help": "Number of episodes to record as GIF.",
        },
        {
            "name": "--gif_fps",
            "type": int,
            "default": 12,
            "help": "GIF frame-rate (frames per second).",
        },
        {
            "name": "--record_env_id",
            "type": int,
            "default": 0,
            "help": "Which vectorized environment index to record.",
        },
        {
            "name": "--camera_preset",
            "type": str,
            "default": "third_person",
            "help": "Camera preset used when task has no camera. Supported: third_person",
        },
        {
            "name": "--record_env_name",
            "type": str,
            "default": "auto",
            "help": "Temporary env override for recording task. Use 'auto' to switch empty_env -> env_with_obstacles in Warp mode.",
        },
        {
            "name": "--include_segmentation",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Concatenate segmentation view below depth image when available.",
        },
        {
            "name": "--gif_mode",
            "type": str,
            "default": "depth_seg_color",
            "help": "GIF view mode: depth_seg_color, depth_only, seg_only.",
        },
        {
            "name": "--max_record_steps",
            "type": parse_optional_int,
            "default": 180,
            "help": "Optional max recorded steps per GIF. Use none/null/auto to disable truncation.",
        },
    ]
    args = parse_arguments(description="RL Policy Playback + GIF Recorder", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):
    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]

    config["params"]["config"]["env_config"]["headless"] = True
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["use_warp"] is not None:
        config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]

    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {
        "use_vecenv": True,
        "render": False,
        "deterministic": True,
        "games_num": args["record_episodes"],
    }
    return config


def _register_temp_task_with_camera(task_name: str, preset: str) -> Tuple[str, bool]:
    task_cfg = task_registry.get_task_config(task_name)
    task_cls = task_registry.get_task_class(task_name)
    robot_name = task_cfg.robot_name
    robot_cfg = robot_registry.get_robot_config(robot_name)

    if preset != "third_person":
        raise ValueError(
            f"Unsupported camera preset '{preset}'. Supported presets: third_person"
        )

    class ThirdPersonDepthCameraConfig(BaseDepthCameraConfig):
        height = THIRD_PERSON_PRESET.height
        width = THIRD_PERSON_PRESET.width
        max_range = THIRD_PERSON_PRESET.max_range
        min_range = THIRD_PERSON_PRESET.min_range
        segmentation_camera = True
        # WarpSensor applies min/max placement only when randomize_placement=True.
        randomize_placement = True
        min_translation = list(THIRD_PERSON_PRESET.translation_xyz)
        max_translation = list(THIRD_PERSON_PRESET.translation_xyz)
        min_euler_rotation_deg = list(THIRD_PERSON_PRESET.euler_deg)
        max_euler_rotation_deg = list(THIRD_PERSON_PRESET.euler_deg)
        nominal_position = list(THIRD_PERSON_PRESET.translation_xyz)

    class RecordingRobotConfig(robot_cfg):
        class sensor_config(robot_cfg.sensor_config):
            enable_camera = True
            camera_config = ThirdPersonDepthCameraConfig
            enable_lidar = False

    record_robot_name = f"{robot_name}_recording_{preset}"
    robot_registry.register(record_robot_name, robot_registry.get_robot_class(robot_name), RecordingRobotConfig)

    class RecordingTaskConfig(task_cfg):
        pass

    RecordingTaskConfig.robot_name = record_robot_name
    RecordingTaskConfig.headless = True
    # Keep Warp by default for headless server rendering.
    RecordingTaskConfig.use_warp = True
    base_args = getattr(task_cfg, "args", {}) or {}
    RecordingTaskConfig.args = dict(base_args)
    RecordingTaskConfig.args["include_robot_in_warp"] = True

    record_task_name = f"{task_name}_recording_{preset}"
    task_registry.register_task(record_task_name, task_cls, RecordingTaskConfig)
    return record_task_name, True


def _register_temp_task_env_override(task_name: str, new_env_name: str, suffix: str) -> str:
    task_cfg = task_registry.get_task_config(task_name)
    task_cls = task_registry.get_task_class(task_name)

    class RecordingEnvOverrideTaskConfig(task_cfg):
        pass

    RecordingEnvOverrideTaskConfig.env_name = new_env_name
    RecordingEnvOverrideTaskConfig.headless = True
    RecordingEnvOverrideTaskConfig.use_warp = True

    record_task_name = f"{task_name}_{suffix}"
    task_registry.register_task(record_task_name, task_cls, RecordingEnvOverrideTaskConfig)
    return record_task_name


def _register_rlgames_env(task_name: str, recorder: EpisodeGifRecorder):
    env_configurations.configurations[task_name] = {
        "env_creator": lambda **kwargs: RecordingTaskWrapper(
            task_registry.make_task(task_name, **kwargs), recorder
        ),
        "vecenv_type": "AERIAL-RLGPU-RECORD",
    }
    try:
        vecenv.register(
            "AERIAL-RLGPU-RECORD",
            lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
        )
    except Exception:
        # rl_games may already have this vecenv type registered in long-lived Python sessions.
        pass


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def main():
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())
    args["play"] = True
    args["train"] = False

    cwd = Path.cwd()
    checkpoint_path = _resolve_path(args["checkpoint"], cwd)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please provide a valid --checkpoint path."
        )
    args["checkpoint"] = str(checkpoint_path)

    config_path = _resolve_path(args["file"], cwd)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please provide a valid --file path."
        )

    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    base_task = args["task"] or config["params"]["config"]["env_name"]
    effective_task, created_temp_task = _register_temp_task_with_camera(
        base_task, args["camera_preset"]
    )

    # Resolve use_warp with an "auto" mode. If we created a temporary third-person camera task,
    # follow the effective task config.
    if args["use_warp"] is None:
        args["use_warp"] = bool(task_registry.get_task_config(effective_task).use_warp)

    # Preserve source env by default. Users can still override manually via --record_env_name.
    effective_task_cfg = task_registry.get_task_config(effective_task)
    chosen_env_name = effective_task_cfg.env_name
    if str(args["record_env_name"]).lower() != "auto":
        chosen_env_name = args["record_env_name"]

    env_overridden = False
    if chosen_env_name != effective_task_cfg.env_name:
        effective_task = _register_temp_task_env_override(
            effective_task, chosen_env_name, "recording_env_override"
        )
        env_overridden = True

    args["task"] = effective_task

    record_dir = _resolve_path(args["record_dir"], cwd)
    recorder = EpisodeGifRecorder(
        output_dir=record_dir,
        target_episodes=max(1, int(args["record_episodes"])),
        fps=max(1, int(args["gif_fps"])),
        record_env_id=int(args["record_env_id"]),
        include_segmentation=bool(args["include_segmentation"]),
        max_record_steps=args["max_record_steps"],
        gif_mode=args["gif_mode"],
    )
    _register_rlgames_env(effective_task, recorder)

    config = update_config(config, args)

    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(config)

    print(f"[play_record] Config file: {config_path}")
    print(f"[play_record] Base task: {base_task}")
    print(f"[play_record] Effective task: {effective_task}")
    if created_temp_task:
        print("[play_record] Camera was not enabled in the base task. A temporary camera task was created.")
    if env_overridden:
        print(f"[play_record] Recording env overridden to: {chosen_env_name}")
    print(f"[play_record] use_warp: {args['use_warp']}")
    print(f"[play_record] include_robot_in_warp: True")
    print(f"[play_record] record_target_xyz: {RECORD_TARGET_XYZ}")
    print(f"[play_record] camera_translation_xyz: {THIRD_PERSON_PRESET.translation_xyz}")
    print(f"[play_record] gif_mode: {recorder.gif_mode}")
    print(f"[play_record] max_record_steps: {args['max_record_steps']}")
    print(f"[play_record] Checkpoint: {checkpoint_path}")
    print(f"[play_record] Output dir: {record_dir}")
    print(f"[play_record] Target episodes: {args['record_episodes']}")

    import aerial_gym.sim.sim_builder as sim_builder_module
    import aerial_gym.robots.robot_manager as robot_manager_module

    original_env_manager_class = sim_builder_module.EnvManager
    original_warp_sensor_class = robot_manager_module.WarpSensor
    sim_builder_module.EnvManager = RecordingEnvManager
    robot_manager_module.WarpSensor = RecordingWarpSensor
    print("[play_record] Patched SimBuilder.EnvManager -> RecordingEnvManager")
    print("[play_record] Patched RobotManager.WarpSensor -> RecordingWarpSensor (yaw-follow stabilization)")
    try:
        runner.run(args)
    except RecordingComplete as exc:
        print(f"[play_record] {exc}")
    finally:
        sim_builder_module.EnvManager = original_env_manager_class
        robot_manager_module.WarpSensor = original_warp_sensor_class
        print("[play_record] Restored SimBuilder.EnvManager")
        print("[play_record] Restored RobotManager.WarpSensor")
        print(f"[play_record] Finished. Saved {recorder.episodes_saved} episode GIF(s).")


if __name__ == "__main__":
    main()
