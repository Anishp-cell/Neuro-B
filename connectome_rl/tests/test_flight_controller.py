"""Unit & Integration Tests for the Aerodynamic Flight Controller."""

from __future__ import annotations

import numpy as np
import pytest

from connectome_rl.src.envs.flight_controller import (
    FlightController,
    FlightConfig,
    FlightState,
)


class MockPhysicsData:
    """Mock MuJoCo data structure for unit testing."""
    def __init__(self):
        self.xpos = np.zeros((10, 3), dtype=np.float64)
        self.xmat = np.eye(3, dtype=np.float64).reshape(-1)  # Identity rotation
        self.xmat = np.tile(self.xmat, (10, 1))
        self.cvel = np.zeros((10, 6), dtype=np.float64)
        self.xfrc_applied = np.zeros((10, 6), dtype=np.float64)


class MockPhysicsModel:
    """Mock MuJoCo model structure for unit testing."""
    def __init__(self):
        self.nbody = 10
        self._names = {
            0: "world",
            1: "0/Head",
            2: "0/Thorax",
            3: "0/LWing",
            4: "0/RWing",
            5: "0/Abdomen",
        }

    def id2name(self, i: int, obj_type: str) -> str:
        return self._names.get(i, f"body_{i}")


class MockSimulation:
    """Mock FlyGym simulation."""
    def __init__(self):
        self.physics = type("Physics", (), {
            "model": MockPhysicsModel(),
            "data": MockPhysicsData(),
        })()


def test_flight_controller_initial_state():
    """Verify initial state is GROUND with zero flight telemetry."""
    controller = FlightController()
    assert controller.state == FlightState.GROUND
    assert controller.current_altitude == 0.0
    assert controller.current_speed == 0.0


def test_takeoff_request():
    """Verify takeoff transition and jump catapult force."""
    controller = FlightController()
    sim = MockSimulation()
    sim.physics.data.xpos[2] = np.array([0.0, 0.0, 0.5])  # On ground

    success = controller.request_takeoff()
    assert success is True
    assert controller.state == FlightState.TAKEOFF

    # Step takeoff physics
    metrics = controller.update(sim=sim, dt=0.01)
    assert metrics["state"] == "TAKEOFF"
    # Upward force must exceed gravity (9.81)
    assert sim.physics.data.xfrc_applied[2, 2] > 9.81


def test_takeoff_to_flying_transition():
    """Verify state transitions from TAKEOFF to FLYING after jump duration."""
    config = FlightConfig(takeoff_duration_s=0.03)
    controller = FlightController(config=config)
    sim = MockSimulation()
    sim.physics.data.xpos[2] = np.array([0.0, 0.0, 3.0])  # Elevated

    controller.request_takeoff()
    # Step past takeoff duration
    controller.update(sim=sim, dt=0.04)
    assert controller.state == FlightState.FLYING


def test_airborne_aerodynamics():
    """Verify hover equilibrium, forward thrust, and drag in FLYING state."""
    controller = FlightController()
    controller.state = FlightState.FLYING
    sim = MockSimulation()
    sim.physics.data.xpos[2] = np.array([0.0, 0.0, 10.0])  # 10 mm altitude

    # 1. Hover with zero control (should balance gravity ~9.81)
    metrics = controller.update(sim=sim, throttle=0.0, climb=0.0, dt=1e-4)
    assert metrics["state"] == "FLYING"
    assert np.isclose(sim.physics.data.xfrc_applied[2, 2], 9.81, atol=1.0)

    # 2. Forward thrust
    controller.update(sim=sim, throttle=1.0, dt=1e-4)
    assert sim.physics.data.xfrc_applied[2, 0] > 0.0  # Positive forward force

    # 3. Steering yaw torque
    controller.update(sim=sim, steer_yaw=0.5, dt=1e-4)
    assert abs(sim.physics.data.xfrc_applied[2, 5]) > 0.0  # Yaw torque


def test_landing_and_touchdown():
    """Verify landing approach and touchdown transition to GROUND."""
    controller = FlightController()
    controller.state = FlightState.FLYING
    sim = MockSimulation()
    sim.physics.data.xpos[2] = np.array([0.0, 0.0, 1.5])  # Near ground

    # Request landing
    controller.request_landing()
    assert controller.state == FlightState.LANDING

    # Step landing until touchdown
    sim.physics.data.xpos[2] = np.array([0.0, 0.0, 0.5])  # Touched down
    controller.update(sim=sim, dt=1e-4)
    assert controller.state == FlightState.GROUND


def test_toggle_flight():
    """Verify toggle_flight cleanly alternates between ground and flight."""
    controller = FlightController()
    assert controller.state == FlightState.GROUND

    controller.toggle_flight()
    assert controller.state == FlightState.TAKEOFF

    controller.state = FlightState.FLYING
    controller.toggle_flight()
    assert controller.state == FlightState.LANDING
