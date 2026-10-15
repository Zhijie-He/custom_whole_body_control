from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
from pathlib import Path

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, 
                 motion_file: str | None, 
                 body_indexes: Sequence[int] | torch.Tensor, 
                 device: str = "cpu", 
                 motion_dir:str | None = None):
        self.joint_pos: torch.Tensor
        self.joint_vel: torch.Tensor
        self._body_pos_w: torch.Tensor
        self._body_quat_w: torch.Tensor
        self._body_lin_vel_w: torch.Tensor
        self._body_ang_vel_w: torch.Tensor

        if(motion_file is None) == (motion_dir is None):
            raise ValueError("Motion file and motion dir cannot be none simultaneously!")

        if motion_dir is not None:
            directory = Path(motion_dir)
            if not directory.is_dir():
                raise ValueError("Motion directory not exists!")
            
            files = sorted(directory.glob("*.npz"))
        else:
            file = Path(motion_file)
            if not file.is_file():
                raise ValueError("Motion file not exists!")
            
            files = [file]

        if len(files) == 0:
            raise ValueError("No NPZ files found!")

        self.names = [path.stem for path in files]
        self.num_motions = len(files)
        self._body_indexes = torch.as_tensor(
            body_indexes,
            dtype=torch.long,
            device=device,
        )

        keys = [
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]

        chunks = {key: [] for key in keys}
        lengths = []
        fps_values = []
        
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                for key in [*keys, "fps"]:
                    if key not in data:
                        raise ValueError(f"{path}: missing field {key}")
                
                length = data["joint_pos"].shape[0]
                if length < 1:
                    raise ValueError(f"Empty motion: {path}")

                for key in keys:
                    array = data[key]

                    if array.shape[0] != length:
                        raise ValueError(f"{path}: {key} num of frames mismatch")

                    if not np.isfinite(array).all():
                        raise ValueError(f"{path}: {key} contains NaN or infinity")

                    chunks[key].append(array.copy())

                lengths.append(length)
                fps_values.append(float(np.array(data["fps"]).reshape(-1)[0]))

        if not np.allclose(fps_values, fps_values[0]):
            raise ValueError("All motion must use the same FPS")

        self.fps = fps_values[0]
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device), 
            self.lengths.cumsum(dim=0)[:-1]
        ])

        for key in keys:
            expected_shpae = chunks[key][0].shape[1:]
            if any(
                array.shape[1:] != expected_shpae
                for array in chunks[key]
            ):
                raise ValueError(
                    f"{key}: non-time dimensions differ between motions"
                )

            tensor = torch.as_tensor(
                np.concatenate(chunks[key], axis=0),
                dtype=torch.float32,
                device=device,
            )

            name = f"_{key}" if key.startswith("body_") else key
            setattr(self, name, tensor)

        if self._body_indexes.ndim != 1:
            raise ValueError("body_indexes must be one-dimension")

        if torch.any(self._body_indexes < 0) or torch.any(
            self._body_indexes >= self._body_pos_w.shape[1]
        ):
            raise ValueError("body_indexes contains an out-of-range index")

        self.total_frames = self.joint_pos.shape[0]
        print(f"[INFO] load {len(files)} motions!")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    def get_body_pos_w_by_frame_id(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return self._body_pos_w[
            frame_ids[:, None],
            self._body_indexes,
        ]

    def get_body_quat_w_by_frame_id(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return self._body_quat_w[
            frame_ids[:, None],
            self._body_indexes,
        ]
    
    def get_body_lin_vel_w_by_frame_id(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return self._body_lin_vel_w[
            frame_ids[:, None],
            self._body_indexes,
        ]
    
    def get_body_ang_vel_w_by_frame_id(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return self._body_ang_vel_w[
            frame_ids[:, None],
            self._body_indexes,
        ]

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            motion_file=self.cfg.motion_file, 
            motion_dir=self.cfg.motion_dir,
            body_indexes=self.body_indexes, 
            device=self.device,
        )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.has_motion = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motion_weights = torch.ones(self.motion.num_motions, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_frames = max(1, round(float(self.motion.fps)))
        self.bin_counts = (
            self.motion.lengths + self.bin_frames - 1
        ) // self.bin_frames

        max_bins = int(self.bin_counts.max().item())
        self.bin_failed_count = torch.zeros(
            self.motion.num_motions,
            max_bins,
            device=self.device,
        )
        self._current_bin_failed = torch.zeros_like(
            self.bin_failed_count
        )

        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], 
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def frame_ids(self) -> torch.Tensor:
        return self.motion.offsets[self.motion_ids] + self.time_steps
    
    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.frame_ids]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.frame_ids]

    @property
    def body_pos_w(self) -> torch.Tensor:
        positions = self.motion.get_body_pos_w_by_frame_id(self.frame_ids)
        return positions + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.get_body_quat_w_by_frame_id(self.frame_ids)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.get_body_lin_vel_w_by_frame_id(self.frame_ids)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.get_body_ang_vel_w_by_frame_id(self.frame_ids)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        delta = self.anchor_pos_w - self.robot_anchor_pos_w
        self.metrics["error_anchor_pos"] = torch.norm(delta, dim=-1)
        self.metrics["error_anchor_pos_xy"] = torch.norm(delta[:, :2], dim=-1)
        self.metrics["error_anchor_pos_z"] = delta[:, 2].abs()
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)

        if num_envs == 0:
            return

        # 1. 为每个环境选择动作
        sampled_motion_ids = torch.multinomial(
            self.motion_weights,
            num_samples=num_envs,
            replacement=True,
        )

        self.motion_ids[env_ids] = sampled_motion_ids

        # 2. 分别为每个动作选择起始帧
        for motion_id in sampled_motion_ids.unique().tolist():

            # 找到本次分到这个动作的环境
            selected = sampled_motion_ids == motion_id
            selected_env_ids = env_ids[selected]

            # 当前动作实际有多少个 bin
            bin_count = int(self.bin_counts[motion_id].item())

            # 只读取当前动作的有效 bin
            scores = self.bin_failed_count[
                motion_id, :bin_count
            ].clone()

            # 保证每个 bin 都有机会被选中
            scores += self.cfg.adaptive_uniform_ratio / bin_count

            # 保留原来的平滑逻辑
            # 平滑只在当前动作内部进行
            if self.cfg.adaptive_kernel_size > 1:
                padded_scores = torch.nn.functional.pad(
                    scores[None, None, :],
                    (0, self.cfg.adaptive_kernel_size - 1),
                    mode="replicate",
                )

                scores = torch.nn.functional.conv1d(
                    padded_scores,
                    self.kernel[None, None, :],
                ).flatten()

            # 转换为概率
            probabilities = scores / scores.sum()

            # 3. 选择 bin
            sampled_bins = torch.multinomial(
                probabilities,
                num_samples=len(selected_env_ids),
                replacement=True,
            )

            # 4. 在 bin 内选择具体帧
            start_frames = sampled_bins * self.bin_frames

            motion_length = self.motion.lengths[motion_id]

            # 最后一个 bin 可能没有满一整秒
            available_frames = torch.clamp(
                motion_length - start_frames,
                max=self.bin_frames,
            )

            random_offsets = (
                torch.rand(
                    len(selected_env_ids),
                    device=self.device,
                )
                * available_frames
            ).long()

            self.time_steps[selected_env_ids] = (
                start_frames + random_offsets
            )

            # 5. 更新采样指标
            entropy = -(
                probabilities
                * probabilities.clamp_min(1e-12).log()
            ).sum()

            if bin_count > 1:
                entropy = entropy / math.log(bin_count)
            else:
                entropy = torch.zeros_like(entropy)

            top_probability, top_bin = probabilities.max(dim=0)

            self.metrics["sampling_entropy"][selected_env_ids] = entropy

            self.metrics["sampling_top1_prob"][selected_env_ids] = (
                top_probability
            )

            self.metrics["sampling_top1_bin"][selected_env_ids] = (
                top_bin.float() / bin_count
            )

    def _record_failures(self, env_ids: torch.Tensor):
        failed = self._env.termination_manager.terminated[env_ids]

        initialized = self.has_motion[env_ids]

        failed_env_ids = env_ids[failed & initialized]

        if len(failed_env_ids) == 0:
            return 

        motion_ids = self.motion_ids[failed_env_ids]

        bin_ids = self.time_steps[failed_env_ids] // self.bin_frames

        bin_ids = torch.minimum(
            bin_ids,
            self.bin_counts[motion_ids] - 1,
        )

        counts = torch.ones(
            len(failed_env_ids),
            device=self.device
        )

        self._current_bin_failed.index_put_(
            (motion_ids, bin_ids),
            counts,
            accumulate=True
        )

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor, record_failures:bool = True):
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        )   

        if record_failures:
            self._record_failures(env_ids_tensor)

        self._adaptive_sampling(env_ids_tensor)
        self.has_motion[env_ids_tensor] = True

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        current_lengths = self.motion.lengths[self.motion_ids]
        finished_env_ids = torch.where(
            self.time_steps >= current_lengths
        )[0]

        self._resample_command(
            finished_env_ids,
            record_failures=False
        )

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | None = None
    motion_dir: str | None = None

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


