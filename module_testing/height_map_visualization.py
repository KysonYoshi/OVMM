import time
import numpy as np
import matplotlib.pyplot as plt
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
plt.ion()

import os

SAVE_DIR = "./saved_heightmaps"
os.makedirs(SAVE_DIR, exist_ok=True)

saved_count = 0  # global counter

def handle_heightmap(msg: HeightMap_):
    global saved_count

    print("⚙️ [HeightMap]")
    print(f"  Stamp: {msg.stamp}")
    print(f"  Frame_id: {msg.frame_id}")
    print(f"  Resolution: {msg.resolution}")
    print(f"  Size: {msg.width} x {msg.height}")
    print(f"  Origin: {msg.origin}")

    # Convert flat sequence to 2D array
    height_map = np.array(msg.data, dtype=np.float32).reshape((msg.height, msg.width))

    # Replace invalid cells with NaN
    height_map[height_map == 1.0e9] = np.nan

    # Plot
    plt.imshow(height_map, cmap='terrain', origin='lower')
    plt.colorbar(label='Height (m)')
    plt.title(f"Height Map @ stamp {msg.stamp}")
    plt.xlabel('X (cells)')
    plt.ylabel('Y (cells)')
    plt.pause(0.01)
    plt.clf()


if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp46s0")
    hs_sub = ChannelSubscriber("rt/utlidar/height_map_array", HeightMap_)
    hs_sub.Init(handle_heightmap)


    while True:
        time.sleep(1)