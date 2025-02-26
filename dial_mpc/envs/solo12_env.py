from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path

#import dial_mpc.envs as dial_envs


""" ███████╗ ██████╗ ██╗      ██████╗    ██╗██████╗ 
    ██╔════╝██╔═══██╗██║     ██╔═══██╗  ███║╚════██╗
    ███████╗██║   ██║██║     ██║   ██║  ╚██║ █████╔╝
    ╚════██║██║   ██║██║     ██║   ██║   ██║██╔═══╝ 
    ███████║╚██████╔╝███████╗╚██████╔╝   ██║███████╗
    ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝    ╚═╝╚══════╝ """
                                                                                                                                              


@dataclass
class Solo12EnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = 4.0
    kd: Union[float, jax.Array] = 0.2
    default_vx: float = 0.5
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "trot"


class Solo12Env(BaseEnv):
    def __init__(self, config: Solo12EnvConfig):
        super().__init__(config)

        self._foot_radius = 0.01 # Updated foot radius

        self._gait = config.gait
        # FR leg, FL leg, RR leg, RL leg
        self._gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            "trot": jnp.array([0.45, 2, 0.06]),
            "canter": jnp.array([0.4, 4, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }

        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base_link"
        )

        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]

        self.joint_range = jnp.array(
            [
                [-0.3, 0.3],
                [0.2, 1.0],
                [-1.4, -0.6],
                [-0.3, 0.3],
                [0.2, 1.0],
                [-1.4, -0.6],
                [-0.3, 0.3],
                [0.2, 1.0],
                [-1.4, -0.6],
                [-0.3, 0.3],
                [0.2, 1.0],
                [-1.4, -0.6],
            ]
        )
        """ self.joint_range = jnp.array(
            [
                [-0.3, 0.3],
                [0.0, 0.8],
                [-1.2, -0.4],
                [-0.3, 0.3],
                [0.0, 0.8],
                [-1.2, -0.4],
                [-0.3, 0.3],
                [0.0, 0.8],
                [-1.2, -0.4],
                [-0.3, 0.3],
                [0.0, 0.8],
                [-1.2, -0.4],
            ]
        ) """
        """ self.joint_range = jnp.array(
            [
                [-0.5, 0.5],
                [-1, 1],
                [-2, -0.5],
                [-0.5, 0.5],
                [-1, 1],
                [-2, -0.5],
                [-0.5, 0.5],
                [-1, 1],
                [-2, -0.5],
                [-0.5, 0.5],
                [-1, 1],
                [-2, -0.5],
            ]
        ) """
        """ self.joint_range = jnp.array(
            [
                [-0.9, 0.9],
                [-1.45, 1.45],
                [-2.8, 2.8],
                [-0.9, 0.9],
                [-1.45, 1.45],
                [-2.8, 2.8],
                [-0.9, 0.9],
                [-1.45, 1.45],
                [-2.8, 2.8],
                [-0.9, 0.9],
                [-1.45, 1.45],
                [-2.8, 2.8],
            ]
        ) """
        self.joint_velocity_range = jnp.array(
            [
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
                [-20, 20],
            ]
        )
        self.joint_torque_range = 3/4*jnp.array(
            [
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-4, 4],
            ]
        )
        feet_site = [
            "FL_foot",
            "FR_foot",
            "HL_foot",
            "HR_foot",
        ]
        feet_site_id = [
            mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = jnp.array(feet_site_id)

    def make_system(self, config: Solo12EnvConfig) -> System:
        model_path = get_model_path("solo12", "solo12_gen.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, 0.3]), # Central point of the torso, originally the head of the robot
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4)
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:

        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # switch to new target if randomize_target is True
        def dont_randomize():
            return (
                jnp.array([self._config.default_vx, self._config.default_vy, 0.0]),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        vel_tar, ang_vel_tar = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )
        state.info["vel_tar"] = jnp.minimum(
            vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time, vel_tar
        )
        state.info["ang_vel_tar"] = jnp.minimum(
            ang_vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time,
            ang_vel_tar,
        )

        
        """ ██████╗ ███████╗██╗    ██╗ █████╗ ██████╗ ██████╗ ███████╗
            ██╔══██╗██╔════╝██║    ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝
            ██████╔╝█████╗  ██║ █╗ ██║███████║██████╔╝██║  ██║███████╗
            ██╔══██╗██╔══╝  ██║███╗██║██╔══██║██╔══██╗██║  ██║╚════██║
            ██║  ██║███████╗╚███╔███╔╝██║  ██║██║  ██║██████╔╝███████║
            ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝
                                                                    """

        """ # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)

        # Foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  
        foot_contact_z = foot_pos[:, 2] - self._foot_radius

        contact = foot_contact_z < 1e-3  # A mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # Yaw orientation reward
        yaw_tar = state.info["yaw_tar"] + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar

        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

        # Velocity reward
        vb = global_to_body_velocity(xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1])
        ab = global_to_body_velocity(xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1])

        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)

        # Stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

        # Final reward computation
        reward = (
            reward_yaw * 0.3
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_upright * 0.5
            + reward_gaits * 0.1
        ) """

        # Foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  
        foot_contact_z = foot_pos[:, 2] - self._foot_radius

        contact = foot_contact_z < 1e-3  # A mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # Yaw orientation reward
        yaw_tar = state.info["yaw_tar"] + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar

        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

        # Velocity reward
        vb = global_to_body_velocity(xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1])
        ab = global_to_body_velocity(xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1])

        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)

        # Stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)

        # Final reward computation
        reward = (
            reward_yaw * 0.3
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_upright * 0.5
            + reward_gaits * 0.1
        )



        """  ██████╗ ██████╗ ███╗   ██╗███████╗████████╗██████╗  █████╗ ██╗███╗   ██╗████████╗███████╗
            ██╔════╝██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║╚══██╔══╝██╔════╝
            ██║     ██║   ██║██╔██╗ ██║███████╗   ██║   ██████╔╝███████║██║██╔██╗ ██║   ██║   ███████╗
            ██║     ██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║   ██║   ╚════██║
            ╚██████╗╚██████╔╝██║ ╚████║███████║   ██║   ██║  ██║██║  ██║██║██║ ╚████║   ██║   ███████║
             ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
                                                                                                    """
        """ # Torque constraint violation
        cstr_torque = 0.0 * (jnp.abs(ctrl) - self.joint_torque_range[:,1]) / self.joint_torque_range[:,1]

        # Velocity constraint violation
        cstr_velocity = (jnp.abs(pipeline_state.qvel[6:]) - self.joint_velocity_range[:,1]) / self.joint_velocity_range[:,1]

        # HAA (Hip Adduction/Abduction) angle constraint violation
        cstr_HAA = (jnp.abs(pipeline_state.q[jnp.array([7, 10, 13, 16])]) - 0.2) / 0.2

        # Air time constraint (normalized by foot contact)
        cstr_air_time_lower = (0.2 - state.info["feet_air_time"]) / 0.2 * first_contact
        cstr_air_time_upper = 0.0 * (state.info["feet_air_time"] - 0.4) / 0.4 * first_contact
        cstr_air_time = 0.0 * jnp.maximum(cstr_air_time_lower, cstr_air_time_upper)

        # Orientation constraint (limit on the magnitude of the orientation vector)
        qw, qx, qy, qz = x.rot[self._torso_idx - 1]
        roll_limit = jnp.radians(20)
        pitch_limit = jnp.radians(20)
        roll = 0.0 * jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = 0.0 * jnp.arcsin(2 * (qw * qy - qz * qx))
        cstr_roll = jnp.expand_dims((roll - roll_limit)/roll_limit, axis=0)
        cstr_pitch = jnp.expand_dims((pitch - pitch_limit)/pitch_limit, axis=0)

        # Foot contact constraints (ensuring only one pair of feet are in contact)
        foot_contact = foot_contact_z < 1e-3  # True if foot is in contact with terrain
        pair1 = foot_contact[0] & foot_contact[3]
        pair2 = foot_contact[1] & foot_contact[2]
        cstr_feet_contact = 1.0 - (pair1 + pair2)  # At least one pair in contact
        # cstr_feet_contact = 1.0 - (pair1 ^ pair2)  # Only one pair in contact (XOR)
        # cstr_feet_contact = 1.0 - ((pair1 ^ pair2) | (pair1 & pair2))  # Only one pair in contact or all
        cstr_feet_contact = jnp.expand_dims(cstr_feet_contact, axis=0)

        # Body height constraint (lower bound on torso height)
        cstr_height = jnp.expand_dims((0.22 - x.pos[self._torso_idx - 1, 2])/0.22, axis=0)

        # Feet height constraint
        cstr_feet_height = 0.0 * (pipeline_state.site_xpos[self._feet_site_id][:, 2] - 0.05)/0.05

        # Joint limits constraint violation
        joint_lower_violation = (self.joint_range[:, 0] - pipeline_state.q[7:]) / jnp.abs(self.joint_range[:, 0])
        joint_upper_violation = (pipeline_state.q[7:] - self.joint_range[:, 1]) / jnp.abs(self.joint_range[:, 1])

        # Prendiamo la maggiore violazione tra lower e upper limit
        cstr_joint_limits = jnp.maximum(joint_lower_violation, joint_upper_violation)

        # Concatenate all the constraints into a single array
        constr_raw = jnp.concatenate([
            cstr_torque, 
            cstr_velocity, 
            cstr_HAA, 
            cstr_feet_contact, 
            cstr_air_time, 
            cstr_roll,
            cstr_pitch,
            cstr_height,
            cstr_feet_height,
            cstr_joint_limits
        ])

        # Apply the maximum constraint violation (clamping)
        constr = jnp.maximum(0.0, constr_raw) """

        # Limits for various physical quantities
        torque_limit =   jnp.array([35.0, 35.0, 45.0, 
                                    35.0, 35.0, 45.0,
                                    35.0, 35.0, 45.0,
                                    35.0, 35.0, 45.0])
        velocity_limit = jnp.array([20.0, 20.0, 16.0,
                                    20.0, 20.0, 16.0, 
                                    20.0, 20.0, 16.0, 
                                    20.0, 20.0, 16.0])

        # Torque constraint violation
        cstr_torque = (jnp.abs(ctrl) - self.joint_torque_range[:,1]) / self.joint_torque_range[:,1]

        # Velocity constraint violation
        cstr_velocity = (jnp.abs(pipeline_state.qvel[6:]) - velocity_limit) / velocity_limit

        # HAA (Hip Adduction/Abduction) angle constraint violation
        cstr_HAA = (jnp.abs(pipeline_state.q[jnp.array([7, 10, 13, 16])]) - 0.2) / 0.2

        # Air time constraint (normalized by foot contact)
        cstr_air_time_lower = (0.1 - state.info["feet_air_time"]) / 0.1 * first_contact
        cstr_air_time_upper = (state.info["feet_air_time"] - 0.15) / 0.15 * first_contact
        cstr_air_time = 0.0 * jnp.maximum(cstr_air_time_lower, cstr_air_time_upper)

        # Orientation constraint (limit on the magnitude of the orientation vector)
        qw, qx, qy, qz = x.rot[self._torso_idx - 1]
        roll_limit = jnp.radians(20)
        pitch_limit = jnp.radians(20)
        roll = jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = jnp.arcsin(2 * (qw * qy - qz * qx))
        cstr_roll = 0.0 * jnp.expand_dims((roll - roll_limit)/roll_limit, axis=0)
        cstr_pitch = 0.0 * jnp.expand_dims((pitch - pitch_limit)/pitch_limit, axis=0)

        # Foot contact constraints (ensuring only one pair of feet are in contact)
        foot_contact = foot_contact_z < 1e-3  # True if foot is in contact with terrain
        pair1 = foot_contact[0] & foot_contact[3]
        pair2 = foot_contact[1] & foot_contact[2]
        # cstr_feet_contact = 1.0 - (pair1 + pair2)  # At least one pair in contact
        cstr_feet_contact = 1.0 - (pair1 ^ pair2)  # Only one pair in contact (XOR)
        cstr_feet_contact = 0.0 * jnp.expand_dims(cstr_feet_contact, axis=0)

        # Body height constraint (lower bound on torso height)
        cstr_height = jnp.expand_dims(0.25 - x.pos[self._torso_idx - 1, 2], axis=0)

        # Concatenate all the constraints into a single array
        constr_raw = jnp.concatenate([
            cstr_torque, 
            cstr_velocity, 
            cstr_HAA, 
            cstr_feet_contact, 
            cstr_air_time, 
            cstr_roll,
            cstr_pitch,
            cstr_height
        ])

        # Apply the maximum constraint violation (clamping)
        constr = jnp.maximum(0.0, constr_raw)

        # done
        up = jnp.array([0.0, 0.0, 1.0]) 
        joint_angles = pipeline_state.q[7:]
        # done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        threshold = jnp.cos(0.5)  
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < threshold
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.1
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        # state.info["c_max"] = c_max

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state, constr

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb, # Linear velocity
                ab, # Angular velocity
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        lin_vel_x = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_lin_vel_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0])
        new_ang_vel_cmd = jnp.array([0.0, 0.0, ang_vel_yaw[0]])
        return new_lin_vel_cmd, new_ang_vel_cmd



brax_envs.register_environment("solo12_walk", Solo12Env)
#dial_envs.register_config("my_env_name", UnitreeAliengoEnvConfig)
