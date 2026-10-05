"""Aerodynamic Flight Controller & State Machine for Embodied Drosophila in MuJoCo.

Simulates authentic insect aerodynamics using a quasi-steady aerodynamic model:
  - State Machine: GROUND -> TAKEOFF -> FLYING -> LANDING -> GROUND
  - Ground-to-Air Leg Catapult Jump: Fires T2 jump reflex to launch into the air
  - Aerodynamics: Computes Lift (opposing gravity), Forward Thrust, Aerodynamic Drag,
    and Steering Yaw/Roll Torques applied directly to the fly's Thorax
  - Wing Flapping Oscillation: Dynamic visual wing cycling during flight
  - Air-to-Ground Landing: Proximity-triggered leg extension and touchdown damping
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class FlightState(Enum):
    """Locomotion flight state enum."""
    GROUND = "GROUND"          # 6-legged walking on terrain
    TAKEOFF = "TAKEOFF"        # Catapult jump: legs extend downward to launch
    FLYING = "FLYING"          # Airborne: aerodynamic lift, thrust, and drag active
    LANDING = "LANDING"        # Low-altitude approach: legs extend as landing gear


@dataclass
class FlightConfig:
    """Aerodynamic parameters and physical constants for Drosophila flight."""
    # Fly mass & gravity (FlyGym units: mm, s, mg -> force in micro-Newtons / model units)
    fly_mass: float = 0.001                # 1.0 mg (0.001 kg model scale)
    gravity: float = 9810.0                # mm / s^2 (9.81 m/s^2)
    
    # Aerodynamic forces
    hover_lift_ratio: float = 1.0          # Lift = weight * ratio (1.0 = perfect hover equilibrium)
    max_thrust_n: float = 4.0              # Forward thrust at full throttle
    max_lift_n: float = 16.0               # Maximum vertical lift force
    max_steer_torque: float = 0.08         # Yaw steering torque
    
    # Air resistance (Stokes / form drag in low-Reynolds insect regime)
    drag_linear: float = 0.012             # F_drag = -drag_linear * v
    damping_angular: float = 0.0015        # Tau_damp = -damping_angular * omega
    
    # Flight velocities
    target_cruise_speed: float = 120.0     # mm/s (standard Drosophila cruising speed: 100-250 mm/s)
    max_climb_speed: float = 60.0          # mm/s vertical climb limit
    
    # Takeoff & Landing kinematics
    takeoff_jump_force_z: float = 24.0     # Downward leg push impulse (overcomes weight 9.81)
    takeoff_duration_s: float = 0.035      # 35 ms jump extension
    ground_clearance_touchdown: float = 0.8 # mm altitude considered on-ground
    landing_gear_threshold: float = 2.2    # mm altitude to extend landing legs
    
    # Wing visual oscillation
    flapping_frequency_hz: float = 25.0    # Visual oscillation rate for simulation rendering
    flapping_amplitude_deg: float = 45.0   # Wing stroke angle range


class FlightController:
    """Manages aerodynamic forces, wing kinematics, and ground <-> air transitions."""

    def __init__(self, config: FlightConfig | None = None) -> None:
        self.config = config or FlightConfig()
        self.weight = self.config.fly_mass * self.config.gravity  # ~9.81 model units

        self.state = FlightState.GROUND
        self.state_time: float = 0.0
        self.flapping_phase: float = 0.0

        # Body ID caches
        self.thorax_body_id: int | None = None
        self.lwing_body_id: int | None = None
        self.rwing_body_id: int | None = None

        # Telemetry & flight navigation tracking
        self.current_altitude: float = 0.0
        self.target_altitude: float = 10.0   # Cruise altitude in mm
        self.current_speed: float = 0.0
        self.current_climb_rate: float = 0.0
        self.current_yaw: float = 0.0
        self.target_yaw: float = 0.0
        self.current_pitch: float = 0.0
        self.current_roll: float = 0.0

    def reset(self) -> None:
        """Reset flight controller to initial ground state."""
        self.state = FlightState.GROUND
        self.state_time = 0.0
        self.flapping_phase = 0.0
        self.current_altitude = 0.0
        self.target_altitude = 10.0
        self.current_speed = 0.0
        self.current_climb_rate = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0

    def _resolve_body_ids(self, sim) -> None:
        """Cache MuJoCo body IDs from the simulation model."""
        if self.thorax_body_id is not None:
            return

        for i in range(sim.physics.model.nbody):
            name = sim.physics.model.id2name(i, "body")
            if "thorax" in name.lower():
                self.thorax_body_id = i
            elif "lwing" in name.lower():
                self.lwing_body_id = i
            elif "rwing" in name.lower():
                self.rwing_body_id = i

        if self.thorax_body_id is None:
            self.thorax_body_id = 3

    def request_takeoff(self) -> bool:
        """Trigger takeoff sequence from ground."""
        if self.state == FlightState.GROUND:
            self.state = FlightState.TAKEOFF
            self.state_time = 0.0
            self.target_altitude = 10.0
            return True
        return False

    def request_landing(self) -> bool:
        """Trigger landing approach from flight."""
        if self.state in (FlightState.FLYING, FlightState.TAKEOFF):
            self.state = FlightState.LANDING
            self.state_time = 0.0
            return True
        return False

    def toggle_flight(self) -> FlightState:
        """Toggle between ground walking and flight."""
        if self.state == FlightState.GROUND:
            self.request_takeoff()
        else:
            self.request_landing()
        return self.state

    def update(
        self,
        sim,
        throttle: float = 0.0,
        pitch: float = 0.0,
        steer_yaw: float = 0.0,
        climb: float = 0.0,
        dt: float = 1e-4,
    ) -> dict[str, float]:
        """Update flight aerodynamics and apply forces to MuJoCo simulation."""
        self._resolve_body_ids(sim)
        self.state_time += dt

        # 1. Read current body state
        pos = sim.physics.data.xpos[self.thorax_body_id].copy()
        xmat = sim.physics.data.xmat[self.thorax_body_id].reshape(3, 3)
        cvel = sim.physics.data.cvel[self.thorax_body_id].copy()

        omega = cvel[:3]   # Angular velocity in world frame [wx, wy, wz]
        v_lin = cvel[3:]   # Linear velocity in world frame [vx, vy, vz]

        self.current_altitude = float(pos[2])
        self.current_speed = float(np.linalg.norm(v_lin[:2]))
        self.current_climb_rate = float(v_lin[2])

        # Extract local heading vectors in world frame
        forward_vec = xmat[:, 0]   # Local X (anterior)
        lateral_vec = xmat[:, 1]   # Local Y (left)
        dorsal_vec = xmat[:, 2]    # Local Z (dorsal/upward)

        # Body attitude angles (degrees)
        self.current_pitch = float(np.degrees(np.arcsin(np.clip(forward_vec[2], -1.0, 1.0))))
        self.current_roll = float(np.degrees(np.arcsin(np.clip(lateral_vec[2], -1.0, 1.0))))
        self.current_yaw = float(np.degrees(np.arctan2(forward_vec[1], forward_vec[0])))

        # Horizontal heading projection on XY plane (for forward cruising thrust)
        heading_xy = np.array([forward_vec[0], forward_vec[1], 0.0], dtype=np.float64)
        heading_norm = np.linalg.norm(heading_xy)
        if heading_norm > 1e-6:
            heading_xy /= heading_norm
        else:
            heading_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        f_applied = np.zeros(3, dtype=np.float64)
        tau_applied = np.zeros(3, dtype=np.float64)

        # ---------------------------------------------------------------------
        # State Machine Transitions & Aerodynamics
        # ---------------------------------------------------------------------
        if self.state == FlightState.GROUND:
            # On ground: natural ground contact
            self.target_yaw = self.current_yaw
            if self.current_altitude > self.config.landing_gear_threshold * 1.5:
                self.state = FlightState.FLYING

        elif self.state == FlightState.TAKEOFF:
            # Leg catapult: gentle upward launch impulse
            f_applied[2] += 14.0   # Overcomes 9.81 weight gently without launching like a rocket
            f_applied[:2] += heading_xy[:2] * 1.5  # slight forward nudge
            self.flapping_phase += 2.0 * np.pi * self.config.flapping_frequency_hz * dt

            # Keep body level during takeoff with abdominal sag compensation
            pitch_err_rad = np.radians(self.current_pitch - (-2.0))
            omega_pitch = np.dot(omega, lateral_vec)
            tau_applied += lateral_vec * (3.3 + 8.0 * pitch_err_rad - 0.4 * omega_pitch)

            if self.state_time >= self.config.takeoff_duration_s or self.current_altitude > self.config.landing_gear_threshold:
                self.state = FlightState.FLYING
                self.state_time = 0.0
                self.target_yaw = self.current_yaw

        elif self.state == FlightState.FLYING:
            self.flapping_phase += 2.0 * np.pi * self.config.flapping_frequency_hz * dt

            # 1. Closed-Loop Smooth Altitude Control
            # Climb input adjusts target altitude smoothly (rate of ~25 mm/s)
            self.target_altitude += climb * 25.0 * dt
            self.target_altitude = np.clip(self.target_altitude, 4.0, 30.0)

            alt_error = self.target_altitude - self.current_altitude
            target_vz = np.clip(alt_error * 3.5, -40.0, 40.0)
            vz_error = target_vz - v_lin[2]

            lift_force = self.weight + np.clip(vz_error * 0.06, -3.0, 3.5)
            f_applied[2] += lift_force

            # 2. Pure Horizontal Forward Cruising Thrust
            thrust_mag = np.clip(throttle * self.config.max_thrust_n, 0.0, self.config.max_thrust_n)
            f_applied[:2] += heading_xy[:2] * thrust_mag

            # 3. Aerodynamic Drag
            f_applied -= self.config.drag_linear * v_lin

            # 4. Biological Haltere Attitude Stabilization (Pitch & Roll Lock)
            # Target pitch: slight forward cruise tilt (-3 deg)
            target_pitch_deg = -3.0 + pitch * 10.0
            pitch_err_rad = np.radians(self.current_pitch - target_pitch_deg)
            omega_pitch = np.dot(omega, lateral_vec)
            # 3.3 N*mm trim cancels abdominal sag; PD locks pitch to target level attitude
            tau_pitch = 3.3 + (8.0 * pitch_err_rad - 0.4 * omega_pitch)

            # Target roll: banks into turns
            target_roll_deg = -steer_yaw * 12.0
            roll_err_rad = np.radians(self.current_roll - target_roll_deg)
            omega_roll = np.dot(omega, forward_vec)
            tau_roll = -(6.0 * roll_err_rad + 0.3 * omega_roll)

            # 5. Biological Optomotor Yaw Heading Stabilization
            if abs(steer_yaw) > 0.05:
                # Active turning: steer target heading at up to 90 deg/s
                self.target_yaw -= steer_yaw * 90.0 * dt
                self.target_yaw = (self.target_yaw + 180.0) % 360.0 - 180.0

            yaw_err_deg = (self.current_yaw - self.target_yaw + 180.0) % 360.0 - 180.0
            tau_yaw = -(0.06 * np.radians(yaw_err_deg) + 0.008 * omega[2])

            tau_applied += lateral_vec * tau_pitch + forward_vec * tau_roll
            tau_applied[2] += tau_yaw

            # Check low-altitude descent for landing
            if climb < -0.2 and self.current_altitude < self.config.landing_gear_threshold:
                self.state = FlightState.LANDING

        elif self.state == FlightState.LANDING:
            # Controlled descent & flare
            target_vz = -15.0  # gentle sink rate mm/s
            vz_error = target_vz - v_lin[2]
            f_applied[2] += self.weight + np.clip(vz_error * 0.05, -2.0, 2.0)
            f_applied[:2] -= 0.03 * v_lin[:2]  # Airbrake forward motion

            # Level out attitude for touchdown
            pitch_err_rad = np.radians(self.current_pitch - (-1.0))
            roll_err_rad = np.radians(self.current_roll - 0.0)
            tau_applied += lateral_vec * (3.3 + 8.0 * pitch_err_rad - 0.4 * np.dot(omega, lateral_vec))
            tau_applied -= forward_vec * (6.0 * roll_err_rad + 0.3 * np.dot(omega, forward_vec))

            # Touchdown check
            if self.current_altitude <= self.config.ground_clearance_touchdown:
                self.state = FlightState.GROUND
                self.state_time = 0.0

        # Apply computed forces and torques to MuJoCo thorax body
        sim.physics.data.xfrc_applied[self.thorax_body_id, :3] = f_applied
        sim.physics.data.xfrc_applied[self.thorax_body_id, 3:] = tau_applied

        return {
            "state": self.state.value,
            "altitude_mm": self.current_altitude,
            "target_altitude_mm": self.target_altitude,
            "speed_mm_s": self.current_speed,
            "climb_rate_mm_s": self.current_climb_rate,
            "pitch_deg": self.current_pitch,
            "roll_deg": self.current_roll,
            "yaw_deg": self.current_yaw,
            "lift_force": float(f_applied[2]),
            "thrust_force": float(np.linalg.norm(f_applied[:2])),
            "flapping_angle_deg": float(np.sin(self.flapping_phase) * self.config.flapping_amplitude_deg),
        }
