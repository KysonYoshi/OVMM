import time
import numpy as np
import cv2
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

def main():
    iface = "enp46s0"
    ChannelFactoryInitialize(0, iface)

    client = VideoClient()
    client.SetTimeout(3.0)     
    client.Init()
    i = 0
    while i <= 0:
        i += 1
        code, jpeg_bytes = client.GetImageSample()
        if code != 0 or jpeg_bytes is None:
            print(f"GetImageSample, code={code}...")
            time.sleep(0.5)
            continue

        arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print("JPEG failed to decode")
            break

        cv2.imshow("Go2 Front Cam", img)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"target.jpg", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
