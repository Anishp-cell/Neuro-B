"""Integration tests for UnifiedFlyFlightEnv."""

from __future__ import annotations

import numpy as np
import pytest

from connectome_rl.src.envs.flight_env import UnifiedFlyFlightEnv
from connectome_rl.src.envs.flight_controller import FlightState


def test_unified_flight_env_initialization():
    """Verify UnifiedFlyFlightEnv initializes and resets properly."""
    env = UnifiedFlyFlightEnv(enable_render=False, physics_steps_per_control=5)
    obs, info = env.reset()

    assert obs.shape == (16,)
    assert env.observation_space.contains(obs)
    assert env.flight_controller.state == FlightState.GROUND
    env.close()


def test_unified_flight_env_walking_step():
    """Verify ground walking steps update position."""
    env = UnifiedFlyFlightEnv(enable_render=False, physics_steps_per_control=10)
    obs, _ = env.reset()
    start_x = obs[0]

    # Step forward march [1.0, 1.0, 0.0, 0.0]
    for _ in range(10):
        obs, reward, done, trunc, info = env.step(np.array([1.0, 1.0, 0.0, 0.0]))
        assert info["state"] == "GROUND"

    env.close()


def test_unified_flight_env_takeoff_and_flight():
    """Verify trigger launches fly into TAKEOFF and FLYING."""
    env = UnifiedFlyFlightEnv(enable_render=False, physics_steps_per_control=10)
    obs, _ = env.reset()

    # Trigger takeoff (action[3] = 1.0)
    obs, reward, done, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
    assert env.flight_controller.state in (FlightState.TAKEOFF, FlightState.FLYING)

    # Step airborne with positive climb
    for _ in range(50):
        obs, reward, done, trunc, info = env.step(np.array([0.5, 0.0, 1.0, 0.0]))

    assert env.flight_controller.state == FlightState.FLYING
    # Altitude should have increased
    assert obs[2] > 2.0  # z > 2 mm
    env.close()
