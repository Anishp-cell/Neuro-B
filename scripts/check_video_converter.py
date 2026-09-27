import sys
for mod in ["imageio", "cv2", "PIL", "moviepy"]:
    try:
        __import__(mod)
        print(f"{mod}: available")
    except ImportError:
        print(f"{mod}: NOT available")
