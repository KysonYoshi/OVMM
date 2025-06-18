import cv2
import time
import numpy as np
import argparse

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

def record_video(output_file, fps=10.0):
    """
    Records video from the robot's camera until 'q' is pressed,
    and saves it to disk.
    """
    # ── NEW ── Initialize DDS participant before any clients ──
    ChannelFactoryInitialize(0, "enp46s0")

    # Now it's safe to create and init your VideoClient:
    client = VideoClient()
    client.Init()

    # Grab one frame to determine resolution
    code, jpeg_bytes = client.GetImageSample()
    if code != 0 or jpeg_bytes is None:
        raise RuntimeError("Failed to get initial frame from robot camera.")
    frame = cv2.imdecode(np.frombuffer(bytes(jpeg_bytes), np.uint8), cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"[INFO] Recording to '{output_file}' — press 'q' to stop")

    cv2.namedWindow('Recording', cv2.WINDOW_NORMAL)

    # Capture loop until 'q'
    while True:
        code, jpeg_bytes = client.GetImageSample()
        if code == 0 and jpeg_bytes:
            frame = cv2.imdecode(np.frombuffer(bytes(jpeg_bytes), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                writer.write(frame)
                cv2.imshow('Recording', frame)

        # WaitKey also throttles to approx the target FPS
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved video to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record video until 'q' is pressed.")
    parser.add_argument("output", help="Output video file path (e.g. out.avi)")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate (frames per second)")
    args = parser.parse_args()

    record_video(args.output, args.fps)
