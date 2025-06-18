import time
import numpy as np
import matplotlib.pyplot as plt
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
import queue

# Thread-safe queue for passing point cloud data
update_queue = queue.Queue()

def pointcloud2_to_xyz(msg, scale=1):
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)
    ])
    data_bytes = bytes(msg.data)
    points = np.frombuffer(data_bytes, dtype=dtype)
    points = points.reshape((msg.height, msg.width))
    xyz = np.stack((points['x'], points['y'], points['z']), axis=-1)
    if scale > 1:
        xyz = xyz[::scale, ::scale, :]
    return xyz

# Global plot variables.
fig = None
ax = None
scatter = None

def init_plot():
    global fig, ax, scatter
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Real-Time PointCloud2 Visualization")
    plt.show()

def update_plot(pts):
    print(len(pts))
    global scatter, fig, ax
    scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
    # bounds
    x_min, x_max = pts[:,0].min(), pts[:,0].max()
    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    z_min, z_max = pts[:,2].min(), pts[:,2].max()
    print(f"X range: {x_min:.2f} to {x_max:.2f}")
    print(f"Y range: {y_min:.2f} to {y_max:.2f}")
    print(f"Z range: {z_min:.2f} to {z_max:.2f}")
    ax.relim()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 6)
    fig.canvas.draw_idle()

def handle_pointcloud(msg: PointCloud2_):
    xyz = pointcloud2_to_xyz(msg, scale=1)
    pts = xyz.reshape(-1, 3)
    update_queue.put(pts)

if __name__ == "__main__":
    init_plot()
    ChannelFactoryInitialize(0, "enp46s0")
    pc_sub = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)
    pc_sub.Init(handle_pointcloud)

    last_plot_time = time.time()
    buffer = []

    while True:
        try:
            # Collect everything that arrived
            while not update_queue.empty():
                buffer.append(update_queue.get())

            now = time.time()
            if now - last_plot_time >= 1.0:
                if buffer:
                    # Stack all batches into one big array
                    all_pts = np.vstack(buffer)
                    update_plot(all_pts)
                    buffer.clear()
                last_plot_time = now

            # Keep GUI responsive
            plt.pause(0.1)

        except KeyboardInterrupt:
            break
