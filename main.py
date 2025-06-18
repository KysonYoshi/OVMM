import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import time
import sys
import os
import io
import requests
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, HeightMap_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
from unitree_sdk2py.go2.video.video_client import VideoClient
import keyboard
import curses

from local_agent import LocalAgent
from height_map_visualizer import handle_heightmap
import queue


# Global variable for the latest height map
latest_height_map = None
robot_pos = None

update_queue = queue.Queue()
def handle_heightmap_for_nav(msg: HeightMap_):
    global latest_height_map
   
    height_map = np.array(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    height_map[height_map == 1.0e9] = np.nan
    latest_height_map = {
        "map": height_map,
        "origin": np.array(msg.origin, dtype=np.float32),  # [x, y]
        "resolution": msg.resolution,
        "stamp": msg.stamp
    }
    #print("map: ", latest_height_map.keys())
    
    if "first_origin" not in latest_height_map:
        latest_height_map["first_origin"] = msg.origin

def handle_robotstate(msg: SportModeState_):
    global robot_pos
    if robot_pos is None:
        robot_pos = {}
        robot_pos["first_pos"] = {
            "x": msg.position[0],
            "y": msg.position[1],
            "o": msg.imu_state.rpy[2]
        }
    robot_pos["pos"]= {
        "x": msg.position[0] - robot_pos["first_pos"]["x"],
        "y": msg.position[1] - robot_pos["first_pos"]["y"],
        "o": msg.imu_state.rpy[2]
    }

def send_command_to_robot(sport_client, agent, action):
    global latest_height_map, robot_pos
    step = 100   # Step length (meters)
    turn = 10.0  # Turning angle (degrees)

    if action == 1:  # FORWARD
        sport_client.Move(0.3, 0, 0)
        time.sleep(1)
        sport_client.Move(0, 0, 0)
    elif action == 2:  # TURN LEFT
        sport_client.Move(0.0, 0, 0.5)
        time.sleep(1)
        sport_client.Move(0, 0, 0)
    elif action == 3:  # TURN RIGHT
        sport_client.Move(0.0, 0, -0.5)
        time.sleep(1)
        sport_client.Move(0, 0, 0)

    agent.x_gt = robot_pos["pos"]["x"] * 1000 + 6000
    agent.y_gt = robot_pos["pos"]["y"] * 1000 + 6000
    agent.o_gt = robot_pos["pos"]["o"]
    print("current pos: ", agent.x_gt, agent.y_gt, np.rad2deg(agent.o_gt))

def capture_camera_frame(video_client):
    points_1sec = aggregate_point_cloud(1.0)
    visualize_point_cloud(points_1sec)

    code, jpeg_bytes = video_client.GetImageSample()
    if code != 0 or jpeg_bytes is None:
            print(f"GetImageSample 错误, code={code}，重试...")
            time.sleep(0.5)
    
    arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode JPEG from robot")
    cv2.imwrite(f"current.jpg", img)
    return img

def main_demo():
    global latest_height_map, robot_pos

    ChannelFactoryInitialize(0, "enp46s0")
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    video_client = VideoClient()
    video_client.Init()

    # Subscribe to height map messages
    hs_sub = ChannelSubscriber("rt/utlidar/height_map_array", HeightMap_)
    hs_sub.Init(handle_heightmap_for_nav)
    pos_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    pos_sub.Init(handle_robotstate)
    poic_sub = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)
    poic_sub.Init(handle_pointcloud)

    # If you set up the SSH tunnel as above, use localhost
    url = "http://localhost:5000/estimate"

    # Point these paths at your local copies of the images
    path_target  = "/home/chifeng/OVMM/target.jpg"
    cam_bgr = capture_camera_frame(video_client)

    # 1) Encode to JPEG in memory
    success, jpg_buf = cv2.imencode('.jpg', cv2.cvtColor(cam_bgr, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode camera frame")

    # 2) Wrap in a BytesIO
    current_buf = io.BytesIO(jpg_buf.tobytes())

    # 3) Prepare files dict
    files = {
        "target":  ("target.jpg", open(path_target, "rb"), "image/jpeg"),
        "current": ("current.jpg", current_buf, "image/jpeg"),
    }

    resp = requests.post(url, files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print("ρ (rho):", data["rho"])
    print("φ (phi):", data["phi"])
    
       # or whatever your live capture file is
    rho, phi = data["rho"], data["phi"] 
    
    time.sleep(1)
    agent = LocalAgent(
        map_size_mm=12000,
        map_resolution=60,
        robot_pos = robot_pos
    )
    print("Initial pos: ", agent.x_gt, agent.y_gt, np.rad2deg(agent.o_gt))

    print("[INFO] Waiting for first height map...")
    while latest_height_map is None:
        print("wait")
        time.sleep(2.0)
    #print("height_map_keys: ", latest_height_map.keys())
    agent.update_local_map(latest_height_map)
    agent.set_goal(rho, phi)
    plt.ion()  # Enable interactive mode
    plt.clf()  # Clear previous frame
    plt.imshow(agent.wall_map_gt, cmap='gray', origin='lower')
    plt.title("Local Exp Map Visualization")
    plt.scatter(agent.goal[0], agent.goal[1], color='blue', s=50, marker='o', label="Goal Position")

    # Compute arrow start point
    x = (agent.x_gt - 6000) / 60 + 100
    y = (agent.y_gt - 6000) / 60 + 100
    angle = agent.o_gt
    dx = np.cos(angle)
    dy = np.sin(angle)

    # Draw arrow
    plt.quiver(x, y, dx, dy, color='red', scale=10, scale_units='inches', width=0.01, label="Orientation")

    plt.legend()
    plt.colorbar(label="Values")
    plt.pause(1)  # Short pause to update the plot
    time.sleep(1)

    max_steps = 1000
    for step_i in range(max_steps):
        action, terminate_local = agent.navigate_local()
        send_command_to_robot(sport_client, agent, action)
        print("Step %d, action=%d" % (step_i, action))

        plt.ion()  # Enable interactive mode
        plt.clf()  # Clear previous frame
        plt.imshow(agent.wall_map_gt, cmap='gray', origin='lower')
        plt.title("Local Exp Map Visualization")
        plt.scatter(agent.goal[0], agent.goal[1], color='blue', s=50, marker='o', label="Goal Position")

        # Compute arrow start point
        x = (agent.x_gt - 6000) / 60 + 100
        y = (agent.y_gt - 6000) / 60 + 100
        angle = agent.o_gt
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Draw arrow
        plt.quiver(x, y, dx, dy, color='red', scale=10, scale_units='inches', width=0.01, label="Orientation")

        plt.legend()
        plt.colorbar(label="Values")
        plt.pause(0.001)  # Short pause to update the plot
        print("[INFO] Waiting for next height map...")
        while latest_height_map is None:
            time.sleep(0.1)
        #print("height_map_keys: ", latest_height_map.keys())
        agent.update_local_map(latest_height_map)
        time.sleep(0.1)
        if terminate_local == 1:
            print("[INFO] Local navigation done.")
            break

    print("Done. Final agent pose = (x=%.2f, y=%.2f, th=%.2f deg)" % (
        agent.x_gt, agent.y_gt, np.degrees(agent.o_gt)
    ))

if __name__ == "__main__":
    main_demo()
