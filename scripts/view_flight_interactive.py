"""Launch an Interactive 3D MuJoCo Physics Desktop Window with the Flying Drosophila.

Opens a native OpenGL 3D viewer window directly on your desktop where you can:
  - Watch the fruit fly walk, launch into the air with an explosive leg catapult jump,
    climb, bank through the air with aerodynamic lift and thrust, and smoothly land!
  - Orbit, rotate, and zoom the camera with your mouse in real time.
  - Apply perturbation forces by Ctrl + Right-Click dragging the fly in mid-air!
  - Pause and resume the physics simulation at any moment with Spacebar.

Usage:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/view_flight_interactive.py
"""

from __future__ import annotations

import os

# Explicitly configure WSLg display environment to prevent [WARN: COPY MODE]
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
if "XDG_RUNTIME_DIR" not in os.environ:
    os.environ["XDG_RUNTIME_DIR"] = "/mnt/wslg/runtime-dir"
if "WAYLAND_DISPLAY" in os.environ:
    del os.environ["WAYLAND_DISPLAY"]

# Activate Direct3D 12 hardware acceleration for WSLg using NVIDIA GPU
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import argparse
import sys
import time
import numpy as np

import mujoco
import mujoco.viewer

from flygym import SingleFlySimulation
from flygym.arena import FlatTerrain
from flygym.examples.locomotion.turning_fly import HybridTurningFly

from connectome_rl.src.envs.flight_controller import (
    FlightController,
    FlightConfig,
    FlightState,
)


def launch_flight_viewer(
    pattern: str = "demo",
    cam_dist: float = 22.0,
    cruise_speed: float = 1.0,
) -> None:
    print("=" * 80)
    print("  INTERACTIVE MUJOCO 3D DESKTOP VIEWER: DROSOPHILA FLIGHT ENGINE")
    print(f"  [*] FLIGHT PATTERN:  {pattern.upper()}")
    print(f"  [*] SPEED SCALE:     {cruise_speed}x")
    print(f"  [*] CAMERA DISTANCE: {cam_dist} mm (Thorax Center-of-Mass Tracking)")
    print("=" * 80)
    print("\n[1/2] Initializing HybridTurningFly & Aerodynamic Flight Engine...")

    contact_sensors = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    fly = HybridTurningFly(
        enable_adhesion=True,
        draw_adhesion=False,
        contact_sensor_placements=contact_sensors,
        seed=42,
        timestep=1e-4,
    )

    sim = SingleFlySimulation(
        fly=fly,
        arena=FlatTerrain(),
        timestep=1e-4,
    )

    sim.reset(seed=42)

    flight_ctrl = FlightController()

    m = sim.physics.model.ptr
    d = sim.physics.data.ptr

    # Resolve thorax ID for camera tracking
    thorax_id = 3
    for i in range(sim.physics.model.nbody):
        name = sim.physics.model.id2name(i, "body")
        if "thorax" in name.lower():
            thorax_id = i
            break

    print("[2/2] Launching native 3D OpenGL desktop viewer window...")
    print("\n" + "*" * 75)
    print("  >>> INTERACTIVE 3D WINDOW IS NOW OPEN ON YOUR DESKTOP! <<<")
    print("  * Left-Click + Drag:       Orbit camera around the fly")
    print("  * Right-Click + Drag:      Zoom camera in / out (or Mouse Scroll)")
    print("  * Ctrl + Right-Click Drag: Apply virtual physics force to the fly!")
    print("  * Spacebar:                Pause / Resume simulation")
    print("*" * 75 + "\n")

    # Step simulation with passive viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Camera configuration
        viewer.cam.lookat[:] = d.xpos[thorax_id]
        viewer.cam.distance = cam_dist
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = 135.0
        viewer.sync()

        step_count = 0
        cycle_len = 16000  # Steps per complete flight demo cycle (1.6s physics)
        n_substeps = 15    # Physics steps per viewer frame render (~60 FPS)

        while viewer.is_running():
            for _ in range(n_substeps):
                t_sec = step_count * 1e-4
                cycle_step = step_count % cycle_len

                if pattern == "demo":
                    # Cycle: Walk (0-0.3s) -> Jump (0.3-0.4s) -> Climb (0.4-0.9s) -> Bank (0.9-1.3s) -> Land (1.3-1.6s)
                    if cycle_step < 3000:
                        # Ground walking forward
                        action_cpg = np.array([1.0, 1.0], dtype=np.float64)
                        sim.step(action_cpg)
                        telem = flight_ctrl.update(sim, dt=1e-4)

                    elif cycle_step == 3000:
                        # Takeoff catapult jump trigger
                        flight_ctrl.request_takeoff()
                        action_cpg = np.array([0.0, 0.0], dtype=np.float64)
                        sim.step(action_cpg)
                        telem = flight_ctrl.update(sim, dt=1e-4)

                    elif cycle_step < 9000:
                        # Climb into free flight
                        action_cpg = np.array([0.0, 0.0], dtype=np.float64)
                        sim.step(action_cpg)
                        telem = flight_ctrl.update(sim, throttle=0.9 * cruise_speed, climb=0.8, dt=1e-4)

                    elif cycle_step < 13000:
                        # Aerial banking curve
                        action_cpg = np.array([0.0, 0.0], dtype=np.float64)
                        sim.step(action_cpg)
                        steer_dir = 0.7 if (step_count // cycle_len) % 2 == 0 else -0.7
                        telem = flight_ctrl.update(sim, throttle=0.7 * cruise_speed, steer_yaw=steer_dir, climb=0.05, dt=1e-4)

                    else:
                        # Controlled landing & touchdown
                        action_cpg = np.array([0.2, 0.2], dtype=np.float64)
                        sim.step(action_cpg)
                        telem = flight_ctrl.update(sim, throttle=0.0, climb=-0.6, dt=1e-4)

                elif pattern == "hover":
                    # Continuous hover in mid-air
                    if flight_ctrl.state == FlightState.GROUND:
                        flight_ctrl.request_takeoff()
                    action_cpg = np.array([0.0, 0.0], dtype=np.float64)
                    sim.step(action_cpg)
                    # Maintain ~15 mm altitude
                    curr_z = d.xpos[thorax_id][2]
                    climb_err = np.clip((15.0 - curr_z) * 0.1, -0.8, 0.8)
                    telem = flight_ctrl.update(sim, throttle=0.1, climb=climb_err, dt=1e-4)

                elif pattern == "soar":
                    # Continuous circular soaring flight
                    if flight_ctrl.state == FlightState.GROUND:
                        flight_ctrl.request_takeoff()
                    action_cpg = np.array([0.0, 0.0], dtype=np.float64)
                    sim.step(action_cpg)
                    telem = flight_ctrl.update(sim, throttle=0.8 * cruise_speed, steer_yaw=0.5, climb=0.05, dt=1e-4)

                else:
                    # Default: straight flight
                    action_cpg = np.array([0.0, 0.0], dtype=np.float64)
                    sim.step(action_cpg)
                    telem = flight_ctrl.update(sim, throttle=0.8, climb=0.2, dt=1e-4)

                step_count += 1

            # Camera smoothly tracks Thorax in 3D space
            viewer.cam.lookat[:] = d.xpos[thorax_id]

            # Live terminal telemetry every 600 steps
            if step_count % 600 == 0:
                st = flight_ctrl.state.value
                alt = d.xpos[thorax_id][2]
                spd = telem.get("speed_mm_s", 0.0)
                yaw = telem.get("yaw_deg", 0.0)
                print(f"  [Step {step_count:06d} | Mode: {st:<7s}] Alt: {alt:5.1f} mm | Speed: {spd:5.1f} mm/s | Yaw: {yaw:+5.1f}°")

            viewer.sync()
            time.sleep(0.001)

    sim.close()
    print("\nViewer window closed. Flight demo finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive 3D MuJoCo Flight Simulation Viewer")
    parser.add_argument("--pattern", type=str, default="demo", choices=["demo", "hover", "soar"], help="Flight pattern (demo, hover, soar)")
    parser.add_argument("--cam-dist", type=float, default=22.0, help="Camera distance in mm (default: 22.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Cruise speed scale (default: 1.0)")
    args = parser.parse_args()

    launch_flight_viewer(pattern=args.pattern, cam_dist=args.cam_dist, cruise_speed=args.speed)
