import numpy as np
import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, SportModeState_
from local_agent import LocalAgent

current_imustate = None
current_sportmodestate = None

def handle_sportmodestate(msg: SportModeState_):
    global current_sportmodestate
    current_sportmodestate = msg
    """print("⚙️ [SportModeState]")
    print(f"  Mode: {msg.mode}, Gait: {msg.gait_type}, Height: {msg.body_height}")
    print(f"  Position: {msg.position}")
    print(f"  Velocity: {msg.velocity}")
    print(f"  Yaw Speed: {msg.yaw_speed}")
    print(f"  Foot Forces: {msg.foot_force}")
    print("")"""

def send_command_to_robot(sport_client, agent, action):
    """
    Simulate moving the robot:
      action: 1 => FORWARD, 2 => LEFT, 3 => RIGHT
    In a real project or simulator, use the actual motion control API.
    """
    step = 100   # Step length (meters)
    turn = 10.0  # Turning angle (degrees)

    if action == 1:  # FORWARD
        agent.x_gt += step * np.cos(agent.o_gt)
        agent.y_gt += step * np.sin(agent.o_gt)
        sport_client.Move(0.3,0,0)
        time.sleep(1)
        sport_client.Move(0,0,0)
    elif action == 2:  # TURN LEFT
        agent.o_gt -= np.radians(turn)
        sport_client.Move(0.0,0,-0.5)
        time.sleep(1)
        sport_client.Move(0,0,0)
    elif action == 3:  # TURN RIGHT
        agent.o_gt += np.radians(turn)
        sport_client.Move(0.0,0,0.5)
        time.sleep(1)
        sport_client.Move(0,0,0)
    print(agent.x_gt, agent.y_gt, agent.o_gt)

    



def run_robot_test():
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    agent = LocalAgent(
        map_size_mm=12000,
        map_resolution=50,
    )

    print("Use keys to control the robot: [W] Forward, [A] Left, [D] Right, [Q] Quit")

    while True:
        key = input("Enter command (w/a/d/q): ").strip().lower()
        if key == 'w':
            print("Forward (command 1)")
            send_command_to_robot(sport_client, agent, 1)
        elif key == 'a':
            print("Left (command 2)")
            send_command_to_robot(sport_client, agent, 2)
        elif key == 'd':
            print("Right (command 3)")
            send_command_to_robot(sport_client, agent, 3)
        elif key == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid key. Use [w/a/d/q].")

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp46s0")
    hs_sportmodestate = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    hs_sportmodestate.Init(handle_sportmodestate)
 
    run_robot_test()