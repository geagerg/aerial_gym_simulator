import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh as tm

from aerial_gym.env_manager.env_manager import EnvManager
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("recording_env_manager")


class _ProxyWarpAsset:
    def __init__(self, mesh: tm.Trimesh, segmentation_values: np.ndarray):
        self.asset_unified_mesh = mesh
        self.asset_vertex_segmentation_value = segmentation_values.astype(np.int32)
        self.variable_segmentation_mask = np.zeros_like(
            self.asset_vertex_segmentation_value, dtype=np.int32
        )


class RecordingEnvManager(EnvManager):
    """
    Recording-only environment manager.
    This class is intentionally isolated from core env_manager.py and is used via
    runtime monkey-patch in play_record.py.
    """

    def __init__(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        self.include_robot_in_warp = False
        if isinstance(args, dict):
            self.include_robot_in_warp = bool(args.get("include_robot_in_warp", False))
        super().__init__(
            sim_name=sim_name,
            env_name=env_name,
            robot_name=robot_name,
            controller_name=controller_name,
            device=device,
            args=args,
            num_envs=num_envs,
            use_warp=use_warp,
            headless=headless,
        )

    def _resolve_robot_urdf_path(self, robot_asset_cfg) -> str:
        filepath = os.path.join(robot_asset_cfg.asset_folder, robot_asset_cfg.file)
        return filepath

    def _infer_segmentation_id(self, link_name: str) -> int:
        lname = str(link_name).lower()
        if "motor" in lname or "prop" in lname:
            return 27
        if "arm" in lname:
            return 26
        return 25

    def _build_mesh_from_geometry(self, geometry) -> Optional[tm.Trimesh]:
        sphere = getattr(geometry, "sphere", None)
        if sphere is not None:
            radius = float(getattr(sphere, "radius", 0.05))
            return tm.creation.icosphere(subdivisions=2, radius=max(1e-4, radius))

        cylinder = getattr(geometry, "cylinder", None)
        if cylinder is not None:
            radius = float(getattr(cylinder, "radius", 0.01))
            length = float(getattr(cylinder, "length", 0.05))
            return tm.creation.cylinder(
                radius=max(1e-4, radius), height=max(1e-4, length), sections=18
            )

        box = getattr(geometry, "box", None)
        if box is not None:
            extents = np.asarray(getattr(box, "size", [0.05, 0.05, 0.05]), dtype=np.float64).reshape(-1)
            if extents.size < 3:
                extents = np.array([0.05, 0.05, 0.05], dtype=np.float64)
            return tm.creation.box(extents=np.maximum(extents[:3], 1e-4))
        return None

    def _build_robot_primitive_warp_asset_dict(self, robot_asset_cfg) -> Dict[str, Any]:
        try:
            from urdfpy import URDF
        except Exception as exc:
            raise RuntimeError(
                "urdfpy is required for primitive reconstruction fallback."
            ) from exc

        robot_urdf_path = self._resolve_robot_urdf_path(robot_asset_cfg)
        urdf_asset = URDF.load(robot_urdf_path)
        link_fk_map = urdf_asset.link_fk()

        primitive_meshes: List[tm.Trimesh] = []
        segmentation_values: List[np.ndarray] = []

        for link in urdf_asset.links:
            visuals = getattr(link, "visuals", None) or []
            if len(visuals) == 0:
                continue

            link_tf = np.asarray(link_fk_map.get(link, np.eye(4)), dtype=np.float64)
            link_segmentation = self._infer_segmentation_id(link.name)
            for visual in visuals:
                geometry = getattr(visual, "geometry", None)
                if geometry is None:
                    continue
                primitive_mesh = self._build_mesh_from_geometry(geometry)
                if primitive_mesh is None:
                    continue

                visual_tf = getattr(visual, "origin", None)
                if visual_tf is None:
                    visual_tf = np.eye(4, dtype=np.float64)
                else:
                    visual_tf = np.asarray(visual_tf, dtype=np.float64)
                full_tf = np.matmul(link_tf, visual_tf)

                mesh_tf = primitive_mesh.copy()
                mesh_tf.apply_transform(full_tf)
                primitive_meshes.append(mesh_tf)
                segmentation_values.append(
                    np.full(len(mesh_tf.vertices), link_segmentation, dtype=np.int32)
                )

        if len(primitive_meshes) == 0:
            raise RuntimeError(
                f"URDF primitive reconstruction found no sphere/cylinder/box visuals in {robot_urdf_path}"
            )

        combined_mesh = tm.util.concatenate(primitive_meshes)
        combined_segmentation = np.concatenate(segmentation_values, axis=0)
        proxy_warp_asset = _ProxyWarpAsset(combined_mesh, combined_segmentation)

        logger.warning(
            "Using URDF primitive reconstruction for Warp robot rendering (recording-only)."
        )
        return {
            "filename": "recording_robot_proxy_primitives",
            "warp_asset": proxy_warp_asset,
        }

    def _build_robot_sphere_fallback_asset_dict(self) -> Dict[str, Any]:
        proxy_mesh = tm.creation.icosphere(subdivisions=2, radius=0.18)
        segmentation_values = np.full(len(proxy_mesh.vertices), 25, dtype=np.int32)
        return {
            "filename": "recording_robot_proxy_sphere",
            "warp_asset": _ProxyWarpAsset(proxy_mesh, segmentation_values),
        }

    def _build_robot_warp_asset_dict(self) -> Dict[str, Any]:
        robot_asset_cfg = self.robot_manager.cfg.robot_asset
        # Try loading a warp-compatible copy of the robot asset first.
        try:
            return self.asset_loader.load_selected_file_from_config(
                "recording_robot_warp",
                robot_asset_cfg,
                robot_asset_cfg.file,
                is_robot=False,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to build robot warp asset from URDF ({exc}). "
                "Trying URDF primitive reconstruction."
            )
        try:
            return self._build_robot_primitive_warp_asset_dict(robot_asset_cfg)
        except Exception as exc:
            logger.warning(
                f"URDF primitive reconstruction failed ({exc}). "
                "Falling back to proxy sphere mesh for recording."
            )
            return self._build_robot_sphere_fallback_asset_dict()

    def populate_env(self, env_cfg, sim_cfg):
        self.create_sim(env_cfg, sim_cfg)

        self.robot_manager.create_robot(self.asset_loader)
        robot_warp_asset_dict = None
        if self.cfg.env.use_warp and self.include_robot_in_warp:
            robot_warp_asset_dict = self._build_robot_warp_asset_dict()
            logger.warning("Recording mode: robot mesh will be included in Warp rendering.")

        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()

        if self.keep_in_env is None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise Exception(
                "Inconsistent number of assets kept in the environment. "
                "The number of keep_in_env assets must be equal for all environments. Check."
            )

        segmentation_ctr = 100
        self.global_asset_counter = 0
        self.step_counter = 0
        self.asset_min_state_ratio = None
        self.asset_max_state_ratio = None

        self.global_tensor_dict["crashes"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

        if self.cfg.env.create_ground_plane:
            logger.info("Creating ground plane in Isaac Gym Simulation.")
            self.IGE_env.create_ground_plane()
            logger.info("[DONE] Creating ground plane in Isaac Gym Simulation")

        for i in range(self.cfg.env.num_envs):
            logger.debug(f"Populating environment {i}")
            if i % 1000 == 0:
                logger.info(f"Populating environment {i}")

            env_handle = self.IGE_env.create_env(i)
            if self.cfg.env.use_warp:
                self.warp_env.create_env(i)

            self.robot_manager.add_robot_to_env(
                self.IGE_env, env_handle, self.global_asset_counter, i, segmentation_ctr
            )
            if self.cfg.env.use_warp and self.include_robot_in_warp and robot_warp_asset_dict is not None:
                _, warp_seg_ctr = self.warp_env.add_asset_to_env(
                    robot_warp_asset_dict,
                    i,
                    self.global_asset_counter,
                    segmentation_ctr,
                )
                segmentation_ctr += warp_seg_ctr
            self.global_asset_counter += 1

            self.num_obs_in_env = 0
            for asset_info_dict in self.global_asset_dicts[i]:
                asset_handle, ige_seg_ctr = self.IGE_env.add_asset_to_env(
                    asset_info_dict,
                    env_handle,
                    i,
                    self.global_asset_counter,
                    segmentation_ctr,
                )
                self.num_obs_in_env += 1
                warp_segmentation_ctr = 0
                if self.cfg.env.use_warp:
                    _, warp_segmentation_ctr = self.warp_env.add_asset_to_env(
                        asset_info_dict,
                        i,
                        self.global_asset_counter,
                        segmentation_ctr,
                    )
                self.global_asset_counter += 1
                segmentation_ctr += max(ige_seg_ctr, warp_segmentation_ctr)
                if self.asset_min_state_ratio is None or self.asset_max_state_ratio is None:
                    self.asset_min_state_ratio = torch.tensor(
                        asset_info_dict["min_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                    self.asset_max_state_ratio = torch.tensor(
                        asset_info_dict["max_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                else:
                    self.asset_min_state_ratio = torch.vstack(
                        (
                            self.asset_min_state_ratio,
                            torch.tensor(asset_info_dict["min_state_ratio"], requires_grad=False),
                        )
                    )
                    self.asset_max_state_ratio = torch.vstack(
                        (
                            self.asset_max_state_ratio,
                            torch.tensor(asset_info_dict["max_state_ratio"], requires_grad=False),
                        )
                    )

        if self.asset_min_state_ratio is not None:
            self.asset_min_state_ratio = self.asset_min_state_ratio.to(self.device)
            self.asset_max_state_ratio = self.asset_max_state_ratio.to(self.device)
            self.global_tensor_dict["asset_min_state_ratio"] = self.asset_min_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
            self.global_tensor_dict["asset_max_state_ratio"] = self.asset_max_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
        else:
            self.global_tensor_dict["asset_min_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )
            self.global_tensor_dict["asset_max_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )

        self.global_tensor_dict["num_obstacles_in_env"] = self.num_obs_in_env
