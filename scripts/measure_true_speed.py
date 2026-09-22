import numpy as np
from connectome_rl.src.envs.cpg_wrapper import CPGLocomotionEnv

def main():
    print("Testing CPGLocomotionEnv physical speed...")
    env = CPGLocomotionEnv(seed=42)
    obs, info = env.reset(seed=42)
    print("Initial pos_mm:", info.get("position_mm"))

    # Test 800 steps (0.08 seconds at dt=1e-4)
    for i in range(800):
        obs, r, d, tr, info = env.step(np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))

    dist_800 = float(info.get("forward_dist_mm", 0.0))
    speed_800 = dist_800 / 0.08
    print(f"At step 800 (t=0.08s):")
    print(f"  position_mm: {info.get('position_mm')}")
    print(f"  forward_dist_mm: {dist_800:.4f} mm")
    print(f"  measured speed (dist / 0.08s): {speed_800:.2f} mm/s")
    print("  info keys:", list(info.keys()))
    if "speed_mm_s" in info and info["speed_mm_s"] is not None:
        print(f"  instantaneous speed_mm_s: {info.get('speed_mm_s'):.2f} mm/s")

    # Continue to 10000 steps (1.0 second at dt=1e-4)
    for i in range(800, 10000):
        obs, r, d, tr, info = env.step(np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))

    dist_10k = float(info.get("forward_dist_mm", 0.0))
    speed_10k = dist_10k / 1.0
    print(f"\nAt step 10000 (t=1.00s):")
    print(f"  position_mm: {info.get('position_mm')}")
    print(f"  forward_dist_mm: {dist_10k:.4f} mm")
    print(f"  measured speed (dist / 1.00s): {speed_10k:.2f} mm/s")
    if "speed_mm_s" in info and info["speed_mm_s"] is not None:
        print(f"  instantaneous speed_mm_s: {info.get('speed_mm_s'):.2f} mm/s")
    print(f"  body lengths/sec (assuming 2.5 mm fly): {speed_10k / 2.5:.2f} BL/s")
    env.close()

if __name__ == "__main__":
    main()
