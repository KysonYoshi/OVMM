import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import depth_utils as du

def get_depth_img_from_robot(pipeline):
    """
    Retrieves a single depth frame from the RealSense sensor.
    Returns a synthetic depth image (H, W) after thresholding values >3000.
    """
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_image = np.asanyarray(depth_frame.get_data())
    # Set depth values above 3000 to 0 (or any other invalid marker)
    depth_image[depth_image > 3000] = 0
    return depth_image

def visualize_point_cloud_realtime(pipeline, camera_matrix, scale=4):
    """
    Continuously captures depth frames and updates two 3D visualizations in real time:
    1. The original point cloud (colored by geocentric Z).
    2. A 3D occupancy visualization, where each occupied bin is displayed as a point
       with size and color proportional to its occupancy count.
       
    The colorbar for the occupancy visualization is created only once.
    """
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    plt.ion()  # Enable interactive mode
    
    # Create a figure with two 3D subplots
    fig = plt.figure(figsize=(15, 7))
    ax_pc = fig.add_subplot(121, projection='3d')   # Point cloud view
    ax_occ = fig.add_subplot(122, projection='3d')  # Occupancy view

    # Occupancy bin parameters
    occupancy_map_size = 241  # occupancy grid size (cells)
    z_bins = [0, 300, 1000]  # boundaries for z bins (creates len(z_bins)+1 bins)
    occ_xy_resolution = 50     # resolution used for binning in x and y
    
    
    # Wait for the first valid depth image
    depth_image = None
    while depth_image is None:
        depth_image = get_depth_img_from_robot(pipeline)
    
    # Process the initial depth image for both visualizations
    XYZ = du.get_point_cloud_from_z(depth_image, camera_matrix, scale)
    XYZ = du.transform_camera_view(XYZ, 1250, 0)
    geocentric_pc = du.transform_pose(XYZ, (6000, 6000, 0))
    
    # --- Initial Point Cloud Visualization ---
    points = XYZ.reshape(-1, 3)
    geo_colors = geocentric_pc.reshape(-1, 3)[:, 2]
    ax_pc.scatter(points[:, 0], points[:, 1], points[:, 2],
                  s=1, c=geo_colors, cmap='jet')
    ax_pc.set_xlabel('X')
    ax_pc.set_ylabel('Y')
    ax_pc.set_zlabel('Z')
    ax_pc.set_title('Real-time 3D Point Cloud')
    
    occupancy, _ = du.bin_points(geocentric_pc, occupancy_map_size, z_bins, occ_xy_resolution)
    rows, cols, bins_idx = np.nonzero(occupancy)

    ax_occ.cla()  # Clear the occupancy axis
    scatter_occ = ax_occ.scatter(rows, cols, bins_idx,c=bins_idx, cmap='viridis')
    ax_occ.set_xlabel('X Bin Center')
    ax_occ.set_ylabel('Y Bin Center')
    ax_occ.set_zlabel('Z Bin Center')
    ax_occ.set_title('3D Occupancy Visualization')
    
    
    # Create the colorbar only once outside the loop
    cbar = fig.colorbar(scatter_occ, ax=ax_occ, label='Occupancy Count')
    
    plt.tight_layout()
    plt.show()
    
    # --- Main Loop: Update Visualizations ---
    while True:
        depth_image = get_depth_img_from_robot(pipeline)
        if depth_image is None:
            continue

        # Display the depth colormap via OpenCV
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                           cv2.COLORMAP_JET)
        cv2.imshow('RealSense Depth Stream', depth_colormap)
        cv2.waitKey(1)
        
        # Update point cloud data
        XYZ = du.get_point_cloud_from_z(depth_image, camera_matrix, scale)
        XYZ = du.transform_camera_view(XYZ, 1250, 0)
        geocentric_pc = du.transform_pose(XYZ, (6000, 6000, 0))
        #print(geocentric_pc.shape)
        points = geocentric_pc.reshape(-1, 3)
        geo_colors = points[:, 2]
        
        ax_pc.cla()
        ax_pc.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=1, c=geo_colors, cmap='jet')
        ax_pc.set_xlabel('X')
        ax_pc.set_ylabel('Y')
        ax_pc.set_zlabel('Z')
        ax_pc.set_title('Real-time 3D Point Cloud')
        
        # Update occupancy data and 3D occupancy scatter
        occupancy, _ = du.bin_points(geocentric_pc, occupancy_map_size, z_bins, occ_xy_resolution)
        rows, cols, bins_idx = np.nonzero(occupancy)
        ax_occ.cla()  # Clear the occupancy axis
        scatter_occ = ax_occ.scatter(cols, rows, bins_idx, c=bins_idx, cmap='viridis')
        ax_occ.set_xlabel('X Bin Center')
        ax_occ.set_ylabel('Y Bin Center')
        ax_occ.set_zlabel('Z Bin Center')
        ax_occ.set_title('3D Occupancy Visualization')
        
        # Update the existing colorbar without creating a new one
        cbar.update_normal(scatter_occ)
        
        plt.pause(0.001)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    # Configure depth stream settings.
    pipeline = rs.pipeline()
    config = rs.config()
    
    frame_width, frame_height = 640, 480
    # Enable the depth stream.
    config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, 30)
    
    # Start streaming.
    pipeline.start(config)
    
    # Obtain the camera matrix using depth_utils.
    # Make sure your get_camera_matrix returns an object with attributes: xc, zc, and f.
    fov = 87
    camera_matrix = du.get_camera_matrix(frame_width, frame_height, fov)
    
    # Begin real-time visualization of the point cloud.
    visualize_point_cloud_realtime(pipeline, camera_matrix, scale=3)
