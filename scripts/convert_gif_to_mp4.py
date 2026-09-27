from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def convert_gif_to_mp4(
    gif_path: str | Path = "outputs/digital_sphinx_audit.gif",
    mp4_path: str | Path = "paper/Supplementary_Video_1.mp4",
    fps: int = 20,
    repeat_per_frame: int = 3,
):
    gif_file = Path(gif_path)
    mp4_file = Path(mp4_path)
    mp4_file.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(gif_file)
    n_frames = img.n_frames
    frames = []

    for i in range(n_frames):
        img.seek(i)
        frame_rgb = np.array(img.convert("RGB"))
        # cv2 expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for _ in range(repeat_per_frame):
            frames.append(frame_bgr)

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(mp4_file), fourcc, fps, (w, h))

    for f in frames:
        out.write(f)
    out.release()

    duration_s = len(frames) / fps
    print(f"Successfully converted {n_frames} source frames into {len(frames)} playback frames: {mp4_file} (Duration: {duration_s:.2f} s, {mp4_file.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    convert_gif_to_mp4()
