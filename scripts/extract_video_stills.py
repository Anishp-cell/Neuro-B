from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def extract_stills():
    gif_path = Path("outputs/digital_sphinx_audit.gif")
    out_png = Path("outputs/audit_video_stills.png")

    img = Image.open(gif_path)
    n_frames = img.n_frames
    print(f"Total frames in GIF: {n_frames}")

    # Select 3 representative timestamps: early (15%), mid (50%), late (85%)
    idx_early = int(n_frames * 0.15)
    idx_mid = int(n_frames * 0.50)
    idx_late = int(n_frames * 0.85)
    selected_indices = [idx_early, idx_mid, idx_late]
    selected_frames = []

    for idx in selected_indices:
        target_idx = min(idx, n_frames - 1)
        img.seek(target_idx)
        selected_frames.append(img.copy().convert("RGB"))

    # Compose 3-row figure showing temporal progression
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    timestamps = [
        "A. Initiation (Early Phase: Bilateral Tripod Stance)",
        "B. Divergence (Mid Phase: Asymmetric Turning Emergence)",
        "C. Terminal Kinematics (Late Phase: Stereotyped Steer vs Diffuse Drift)",
    ]

    for i, (frame, title) in enumerate(zip(selected_frames, timestamps)):
        axs[i].imshow(frame)
        axs[i].set_title(f"Synchronized Lesion Response: {title}", fontsize=11, fontweight="bold", pad=6)
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("Saved video stills figure to:", out_png)

if __name__ == "__main__":
    extract_stills()
