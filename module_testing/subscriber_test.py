import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# 匯入資料型別
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_

def handle_pointcloud(msg: PointCloud2_):
    print("📍 [PointCloud2]")
    print(f"  Header: {msg.header}")
    print(f"  Frame: {msg.header.frame_id}")
    print(f"  Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
    print(f"  Size: {msg.width} x {msg.height}")
    print(f"  Data size: {len(msg.data)} bytes")
    print(f"  Fields: {msg.fields}")
    print(f"  is_bigendian: {msg.is_bigendian}")
    print(f"  Point Step: {msg.point_step}")
    print(f"  Row step: {msg.row_step}")
    print(f"  is_dense: {msg.is_dense}")
    #print(f"  Data: {msg.data}")
    print("")

def handle_sportmodestate(msg: SportModeState_):
    print("⚙️ [SportModeState]")
    print(f"  Mode: {msg.mode}, Gait: {msg.gait_type}, Height: {msg.body_height}")
    print(f"  Position: {msg.position}")
    print(f"  Velocity: {msg.velocity}")
    print(f"  Yaw Speed: {msg.yaw_speed}")
    print(f"  Foot Forces: {msg.foot_force}")
    print(f"  Yaw: {msg.imu_state.rpy[2]}")
    print("")

def handle_heightmap(msg: HeightMap_):
    print("⚙️ [HeightMap]")
    print(f"  Stamp: {msg.stamp}")
    print(f"  Frame_id: {msg.frame_id}")
    print(f"  Resolution: {msg.resolution}")
    print(f"  Size: {msg.width} x {msg.height}")
    print(f"  Origin: {msg.origin}")
    #print(f"  Data: {msg.data}")
    print("")

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp46s0")

    # 訂閱
    #pc_sub = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)
    #pc_sub.Init(handle_pointcloud)
    hs_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    hs_sub.Init(handle_sportmodestate)
    #hs_sub = ChannelSubscriber("rt/utlidar/height_map_array", HeightMap_)
    #hs_sub.Init(handle_heightmap)


    while True:
        time.sleep(1)
