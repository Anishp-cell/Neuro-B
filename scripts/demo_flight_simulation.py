"""Demonstration Script: Drosophila Takeoff, Free Flight, Aerial Banking, and Landing.

Simulates a continuous multi-modal physics trajectory:
  1. Phase 1 (0.0 - 0.3s): Ground walking forward with biological tripod gait
  2. Phase 2 (0.3 - 0.4s): Explosive leg catapult jump (takeoff reflex)
  3. Phase 3 (0.4 - 1.0s): Free flight climb with aerodynamic lift and forward thrust
  4. Phase 4 (1.0 - 1.5s): Aerial banking turn (aerodynamic yaw steering)
  5. Phase 5 (1.5 - 2.0s): Controlled descent, landing gear extension, and smooth touchdown!

Renders synchronized 3D camera frames and outputs:
  - outputs/flight_takeoff_landing_demo.mp4 (High-definition MP4)
  - outputs/flight_takeoff_landing_demo.gif (Web preview GIF)
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set up unbuffered stdout logger (bypasses dm_control log suppression)
logger = logging.getLogger("FlightDemo")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)

from connectome_rl.src.envs.flight_env import UnifiedFlyFlightEnv
from connectome_rl.src.envs.flight_controller import FlightState


def get_font(size: int = 14) -> ImageFont.ImageFont | None:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in font_candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def run_flight_demo(
    out_mp4: str | Path = "outputs/flight_takeoff_landing_demo.mp4",
    out_gif: str | Path = "outputs/flight_takeoff_landing_demo.gif",
    num_steps: int = 20000,    # 20,000 steps * 1e-4s = 2.0s physics
    timestep: float = 1e-4,
    duration_s: float = 10.0,  # 10.0 seconds output video
    playback_fps: int = 25,
) -> tuple[Path, Path]:
    out_mp4 = Path(out_mp4)
    out_gif = Path(out_gif)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    target_total_frames = int(duration_s * playback_fps)  # 250 frames
    render_interval = max(num_steps // target_total_frames, 1)

    print("\n" + "=" * 80, flush=True)
    print("  DROSOPHILA EMBODIED FLIGHT ENGINE: TAKEOFF, AERODYNAMICS & LANDING DEMO", flush=True)
    print(f"  Physics Steps: {num_steps:,} (2.0s physics) | Render Interval: every {render_interval} steps", flush=True)
    print(f"  Target Video: {target_total_frames} frames @ {playback_fps} FPS = {duration_s:.1f}s video", flush=True)
    print("=" * 80 + "\n", flush=True)

    logger.info("Initializing UnifiedFlyFlightEnv with 3D tracking camera...")
    env = UnifiedFlyFlightEnv(
        timestep=timestep,
        physics_steps_per_control=1,
        enable_render=True,
        camera_name="camera_right",
        seed=42,
    )

    obs, _ = env.reset(seed=42)
    logger.info("Fly spawned at Z=%.2f mm. Starting multi-modal physics trajectory...", obs[2])

    raw_frames: list[np.ndarray] = []
    telemetry_log: list[dict[str, float]] = []

    # Timeline in physics steps (timestep = 1e-4s = 0.1ms per step):
    # 0..3000   (0.0 - 0.3s): Ground walking forward
    # 3000..4000 (0.3 - 0.4s): Explosive leg catapult jump (takeoff reflex)
    # 4000..10000 (0.4 - 1.0s): Aerodynamic climb to altitude
    # 10000..15000 (1.0 - 1.5s): Aerial banking turn (yaw steering)
    # 15000..20000 (1.5 - 2.0s): Controlled descent, landing gear extension, touchdown

    takeoff_step = int(0.3 * num_steps)
    climb_end_step = int(0.55 * num_steps)
    turn_end_step = int(0.75 * num_steps)

    for step in range(num_steps):
        t_sec = step * timestep

        if step < takeoff_step:
            # Phase 1: Ground walking forward
            action = np.array([1.0, 1.0, 0.0, 0.0])

        elif step == takeoff_step:
            # Phase 2: Trigger leg catapult takeoff reflex
            action = np.array([0.0, 0.0, 0.0, 1.0])

        elif step < climb_end_step:
            # Phase 3: Free flight climb
            action = np.array([0.85, 0.0, 0.75, 0.0])  # [throttle, steer, climb, trigger]

        elif step < turn_end_step:
            # Phase 4: Elegant S-turn aerial banking maneuver (banks right, then left, then centers)
            phase_prog = (step - climb_end_step) / max(turn_end_step - climb_end_step, 1)
            steer = float(np.sin(2.0 * np.pi * phase_prog) * 0.45)
            action = np.array([0.75, steer, 0.05, 0.0])

        else:
            # Phase 5: Controlled landing approach & touchdown
            action = np.array([0.0, 0.0, -0.6, 0.0])

        obs, reward, done, trunc, info = env.step(action)
        telem = info["telemetry"]
        telem["sim_time_s"] = t_sec
        telem["state_str"] = info["state"]
        telem["altitude"] = float(obs[2])
        telem["forward_x"] = float(obs[0])

        # Capture camera frame at paced interval
        if step % render_interval == 0 and len(raw_frames) < target_total_frames:
            frame = env.render()
            if frame is not None:
                raw_frames.append(frame)
                telemetry_log.append(telem)

        # Live progress bar in terminal
        if step % 1000 == 0 or step == num_steps - 1:
            pct = (step + 1) / num_steps * 100.0
            alt = obs[2]
            spd = telem.get("speed_mm_s", 0.0)
            st = info["state"]
            bar_len = 25
            filled = int(bar_len * (step + 1) // num_steps)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:5.1f}% | Step {step:5d}/{num_steps} | State: {st:<7s} | Alt: {alt:5.1f}mm | Spd: {spd:5.1f}mm/s | Frames: {len(raw_frames)}", end="", flush=True)

    print("\n", flush=True)
    env.close()
    logger.info("Simulation completed. Captured %d rendered frames.", len(raw_frames))

    # -------------------------------------------------------------------------
    # Composite HUD Overlays
    # -------------------------------------------------------------------------
    logger.info("Compositing flight HUD overlays...")
    font_large = get_font(15)
    font_med = get_font(12)
    font_small = get_font(11)

    annotated_bgr: list[np.ndarray] = []
    annotated_pil: list[Image.Image] = []

    h, w, _ = raw_frames[0].shape
    header_h = 36
    footer_h = 34
    total_w = w
    total_h = h + header_h + footer_h

    state_colors = {
        "GROUND": (40, 160, 80),      # Emerald Green
        "TAKEOFF": (230, 140, 20),    # Orange
        "FLYING": (30, 120, 220),     # Sky Blue
        "LANDING": (180, 50, 200),    # Violet
    }

    for i, (frame, telem) in enumerate(zip(raw_frames, telemetry_log)):
        canvas = Image.new("RGB", (total_w, total_h), color=(14, 18, 26))
        canvas.paste(Image.fromarray(frame), (0, header_h))
        draw = ImageDraw.Draw(canvas)

        # 1. Top Header Banner
        curr_state = telem.get("state_str", "UNKNOWN")
        badge_color = state_colors.get(curr_state, (100, 100, 100))

        draw.rectangle([0, 0, total_w, header_h], fill=(18, 24, 34))
        draw.text((14, 8), "Drosophila Embodied Flight Engine | MuJoCo Aerodynamics", fill=(230, 235, 245), font=font_large)

        # State Badge
        badge_text = f"MODE: {curr_state}"
        draw.rectangle([total_w - 180, 6, total_w - 14, header_h - 6], fill=badge_color)
        draw.text((total_w - 165, 10), badge_text, fill=(255, 255, 255), font=font_med)

        # 2. Bottom Telemetry HUD
        footer_y = header_h + h
        draw.rectangle([0, footer_y, total_w, total_h], fill=(10, 14, 20))

        t_sim = telem.get("sim_time_s", 0.0) * 1000.0
        alt = telem.get("altitude", 0.0)
        spd = telem.get("speed_mm_s", 0.0)
        lift = telem.get("lift_force", 0.0)
        thrust = telem.get("thrust_force", 0.0)

        left_stat = f"Alt: {alt:5.1f} mm | Spd: {spd:5.1f} mm/s"
        mid_stat = f"Sim Time: {t_sim:4.0f} ms / 2000 ms"
        right_stat = f"Aerodynamics: Lift={lift:4.1f} N | Thrust={thrust:4.1f} N"

        draw.text((14, footer_y + 9), left_stat, fill=(160, 240, 200), font=font_small)
        draw.text((total_w // 2 - 80, footer_y + 9), mid_stat, fill=(255, 255, 140), font=font_small)
        draw.text((total_w - 300, footer_y + 9), right_stat, fill=(180, 210, 255), font=font_small)

        bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        annotated_bgr.append(bgr)
        annotated_pil.append(canvas)

    # 3. Write MP4 Video
    logger.info("Encoding MP4 video to: %s...", out_mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, playback_fps, (total_w, total_h))
    for f in annotated_bgr:
        writer.write(f)
    writer.release()
    duration_actual = len(annotated_bgr) / playback_fps
    logger.info("SUCCESS: Saved Flight Demo MP4: %s (%d KB, %d frames @ %d FPS, Duration: %.2f s)",
                out_mp4, out_mp4.stat().st_size // 1024, len(annotated_bgr), playback_fps, duration_actual)

    # 4. Write Preview GIF
    logger.info("Encoding GIF preview to: %s...", out_gif)
    gif_frames = annotated_pil[::2]
    gif_fps = max(playback_fps // 2, 1)
    gif_frames[0].save(
        out_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / gif_fps),
        loop=0,
    )
    logger.info("SUCCESS: Saved Flight Demo GIF: %s (%d KB)", out_gif, out_gif.stat().st_size // 1024)

    print("\n" + "=" * 80, flush=True)
    print(f"  FLIGHT DEMO COMPLETE! Video saved to: {out_mp4}", flush=True)
    print("=" * 80 + "\n", flush=True)

    return out_mp4, out_gif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Drosophila Flight Engine Demonstration")
    parser.add_argument("--out-mp4", type=str, default="outputs/flight_takeoff_landing_demo.mp4")
    parser.add_argument("--out-gif", type=str, default="outputs/flight_takeoff_landing_demo.gif")
    parser.add_argument("--steps", type=int, default=20000, help="Physics simulation steps (default: 20000 = 2.0s physics)")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (default: 25)")
    parser.add_argument("--duration", type=float, default=10.0, help="Output video duration in seconds (default: 10.0)")
    args = parser.parse_args()

    run_flight_demo(
        out_mp4=args.out_mp4,
        out_gif=args.out_gif,
        num_steps=args.steps,
        duration_s=args.duration,
        playback_fps=args.fps,
    )
