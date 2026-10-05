"""Unified Ground-and-Flight Environment for Drosophila in MuJoCo.

Seamlessly bridges:
  1. Biological Ground Walking: CPG tripod coordination & 42-DOF leg kinematics
  2. Ground-to-Air Takeoff Jump: Leg catapult reflex launching into free flight
  3. Airborne Flight Dynamics: Aerodynamic lift, thrust, low-Reynolds drag, and banking yaw
  4. Air-to-Ground Landing: Leg extension & touchdown ground contact transition
"""

from __future__ import annotations

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from flygym import Fly, SingleFlySimulation, Camera, YawOnlyCamera
from flygym.arena import FlatTerrain
from flygym.examples.locomotion.turning_fly import HybridTurningFly

from connectome_rl.src.envs.flight_controller import (
    FlightController,
    FlightConfig,
    FlightState,
)


class FlightTrackingCamera(YawOnlyCamera):
    """3D Tracking camera that follows the fly across X, Y and Z (altitude) in flight."""

    def _update_cam_pos(
        self,
        physics,
        floor_height: float,
        yaw_correction,
        fly_pos: list[float],
    ):
        # Follow fly in all 3 dimensions (including altitude Z)
        physics.bind(self._cam).xpos = (
            np.array(fly_pos, dtype=np.float64)
            + (yaw_correction.as_matrix() @ self.camera_base_offset).flatten()
        )


class UnifiedFlyFlightEnv(gym.Env):
    """Multi-modal environment supporting both walking and aerodynamic flight."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        flight_config: FlightConfig | None = None,
        timestep: float = 1e-4,
        physics_steps_per_control: int = 10,
        enable_render: bool = False,
        camera_name: str = "camera_right",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.timestep = timestep
        self.physics_steps_per_control = physics_steps_per_control
        self.enable_render = enable_render
        self.camera_name = camera_name
        self.seed = seed

        self.flight_controller = FlightController(config=flight_config)

        # Contact sensors for 6 legs (required by HybridTurningFly)
        self.contact_sensors = [
            f"{leg}{segment}"
            for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
        ]

        self.fly = HybridTurningFly(
            enable_adhesion=True,
            draw_adhesion=False,
            contact_sensor_placements=self.contact_sensors,
            seed=self.seed,
            timestep=self.timestep,
        )

        cameras = []
        if self.enable_render:
            self.cam = FlightTrackingCamera(
                attachment_point=self.fly.model.worldbody,
                camera_name=self.camera_name,
                targeted_fly_names=[self.fly.name],
                play_speed=0.2,
                fps=30,
            )
            cameras.append(self.cam)
        else:
            self.cam = None

        self.sim = SingleFlySimulation(
            fly=self.fly,
            cameras=cameras,
            timestep=self.timestep,
            arena=FlatTerrain(),
        )

        # ---------------------------------------------------------------------
        # Action Space:
        # [0]: Ground Left Drive / Flight Throttle [-1.0, 1.0]
        # [1]: Ground Right Drive / Flight Steering Yaw [-1.0, 1.0]
        # [2]: Flight Climb / Pitch [-1.0, 1.0]
        # [3]: Flight Mode Trigger (> 0.5 triggers Takeoff / Land toggle)
        # ---------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation Space (16 continuous physical dimensions)
        # 0..2:   Thorax Position [x, y, z] (mm)
        # 3..5:   Thorax Linear Velocity [vx, vy, vz] (mm/s)
        # 6..8:   Thorax Angular Velocity [wx, wy, wz] (rad/s)
        # 9..11:  Thorax Euler Orientation [roll, pitch, yaw] (rad)
        # 12:     Flight State Integer (0=GROUND, 1=TAKEOFF, 2=FLYING, 3=LANDING)
        # 13:     Current Speed (mm/s)
        # 14:     Altitude above floor (mm)
        # 15:     Ground Contact Active (0.0 or 1.0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32,
        )

        self._steps: int = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed

        self.flight_controller.reset()
        self._steps = 0

        obs_dict, info = self.sim.reset(seed=self.seed)
        obs = self._build_observation()
        return obs, info

    def _build_observation(self) -> np.ndarray:
        """Extract compact proprioceptive and state vector from MuJoCo."""
        t_id = self.flight_controller.thorax_body_id or 3
        pos = self.sim.physics.data.xpos[t_id]
        cvel = self.sim.physics.data.cvel[t_id]
        xmat = self.sim.physics.data.xmat[t_id].reshape(3, 3)

        # Pitch and roll from rotation matrix
        pitch = float(-np.arcsin(np.clip(xmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(xmat[2, 1], xmat[2, 2]))
        yaw = float(np.arctan2(xmat[1, 0], xmat[0, 0]))

        state_map = {
            FlightState.GROUND: 0.0,
            FlightState.TAKEOFF: 1.0,
            FlightState.FLYING: 2.0,
            FlightState.LANDING: 3.0,
        }
        state_val = state_map[self.flight_controller.state]
        contact_active = 1.0 if pos[2] < 1.0 else 0.0

        obs = np.array([
            pos[0], pos[1], pos[2],
            cvel[3], cvel[4], cvel[5],
            cvel[0], cvel[1], cvel[2],
            roll, pitch, yaw,
            state_val,
            float(np.linalg.norm(cvel[3:5])),
            pos[2],
            contact_active,
        ], dtype=np.float32)
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance multi-modal simulation by physics_steps_per_control steps."""
        self._steps += 1
        action = np.asarray(action, dtype=np.float64)

        # Parse control inputs
        c1 = float(action[0])
        c2 = float(action[1])
        climb_cmd = float(action[2]) if len(action) > 2 else 0.0
        trigger = float(action[3]) if len(action) > 3 else 0.0

        # Handle flight toggle trigger
        if trigger > 0.5:
            self.flight_controller.toggle_flight()

        telemetry: dict[str, float] = {}

        # Physics sub-stepping
        for _ in range(self.physics_steps_per_control):
            curr_state = self.flight_controller.state

            if curr_state == FlightState.GROUND:
                # Ground locomotion: actions drive CPG oscillators [left_drive, right_drive]
                cpg_action = np.array([c1, c2], dtype=np.float64)
                self.sim.step(cpg_action)
                telemetry = self.flight_controller.update(self.sim, dt=self.timestep)

            elif curr_state == FlightState.TAKEOFF:
                # Catapult jump: legs extend downward to launch
                # Neutral leg extension during takeoff
                cpg_action = np.array([0.0, 0.0], dtype=np.float64)
                self.sim.step(cpg_action)
                telemetry = self.flight_controller.update(self.sim, dt=self.timestep)

            elif curr_state == FlightState.FLYING:
                # Airborne flight: c1=throttle, c2=steer_yaw, climb=climb_cmd
                # Legs tuck during flight (zero leg oscillation)
                cpg_action = np.array([0.0, 0.0], dtype=np.float64)
                self.sim.step(cpg_action)
                telemetry = self.flight_controller.update(
                    self.sim,
                    throttle=c1,
                    steer_yaw=c2,
                    climb=climb_cmd,
                    dt=self.timestep,
                )

            elif curr_state == FlightState.LANDING:
                # Landing gear: legs prepare for touchdown
                cpg_action = np.array([0.2, 0.2], dtype=np.float64)
                self.sim.step(cpg_action)
                telemetry = self.flight_controller.update(
                    self.sim,
                    throttle=0.0,
                    climb=-0.5,
                    dt=self.timestep,
                )

        obs = self._build_observation()
        reward = float(telemetry.get("speed_mm_s", 0.0) * 0.1)
        terminated = False
        truncated = False
        info = {"telemetry": telemetry, "state": self.flight_controller.state.value}

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render current camera viewport."""
        if not self.enable_render or self.cam is None:
            return None
        frame = self.sim.render()[0]
        return frame

    def close(self) -> None:
        """Safely release simulation resources."""
        if hasattr(self, "sim") and self.sim is not None:
            self.sim.close()
