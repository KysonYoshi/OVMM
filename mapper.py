import numpy as np

#import depth_utils as du

def build_mapper(camera_height=0.5):
    params = {}
    map_size_mm = 12000
    params["frame_width"] = 640
    params["frame_height"] = 480
    params["fov"] = 87
    params["resolution"] = 60
    params["map_size_mm"] = map_size_mm
    params["agent_min_z"] = 70
    params["agent_medium_z"] = 200
    params["agent_max_z"] = 2000
    params["agent_height"] = camera_height * 1000
    params["agent_view_angle"] = 0
    params["du_scale"] = 3
    params["vision_range"] = 64
    #params["use_mapper"] = 1
    #params["visualize"] = 0
    #params["maze_task"] = 1
    params["obs_threshold"] = 1
    mapper = MapBuilder(params)
    return mapper


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params["frame_width"]
        frame_height = params["frame_height"]
        fov = params["fov"]
        #self.camera_matrix = du.get_camera_matrix(frame_width, frame_height, fov)
        self.vision_range = params["vision_range"]
        self.map_size_mm = params["map_size_mm"]
        self.resolution = params["resolution"]
        agent_min_z = params["agent_min_z"]
        agent_medium_z = params["agent_medium_z"]
        agent_max_z = params["agent_max_z"]
        self.z_bins = [agent_min_z, agent_medium_z, agent_max_z]
        self.du_scale = params["du_scale"]
        #self.use_mapper = params["use_mapper"]
        #self.visualize = params["visualize"]
        #self.maze_task = params["maze_task"]
        self.obs_threshold = params["obs_threshold"]

        self.map = np.zeros(
            (
                self.map_size_mm // self.resolution + 1,
                self.map_size_mm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )
        self.agent_height = params["agent_height"]
        self.agent_view_angle = params["agent_view_angle"]

    def reset_map(self):
        self.map = np.zeros(
            (
                self.map_size_mm // self.resolution + 1,
                self.map_size_mm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )
        

    def update_map(self, latest_height_map, current_pose):

        def bin_height_map_to_occupancy(latest_height_map, z_bins):
            """
            Bins height map values into 3D occupancy map.

            height_map: 2D array of shape (H, W) with np.nan for invalid values
            origin: [origin_x, origin_y], the world coordinates of the map[0,0]
            resolution: meters per cell
            map_size: desired output map size (e.g. 128 for 128x128)
            z_bins: list of z boundaries to bin into

            Returns:
                counts: shape (map_size, map_size, len(z_bins)+1)
                isvalid: shape (H, W, 1)
            """
            # Load height map and rotate it 90° counterclockwise.
            height_map = latest_height_map["map"] * 1000
            map_size = len(self.map[0])
            H, W = height_map.shape
            n_z_bins = len(z_bins) + 1
            counts = np.zeros((map_size, map_size, n_z_bins), dtype=np.int32)
            
            isnotnan = ~np.isnan(height_map)
            
            # Calculate x, y grid coordinate for each cell
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            world_x = (latest_height_map["origin"][0] - latest_height_map["first_origin"][0]) * 1000 + xs * self.resolution + (map_size - H) * self.resolution // 2 + 1
            world_y = (latest_height_map["origin"][1] - latest_height_map["first_origin"][1]) * 1000 + ys * self.resolution + (map_size - H) * self.resolution // 2 + 1
            #print("Cord: ", latest_height_map["origin"][0], latest_height_map["first_origin"][0] )
            
            world_z = height_map

            # Discretize x/y to bins
            X_bin = np.round(world_x / self.resolution).astype(np.int32)
            Y_bin = np.round(world_y / self.resolution).astype(np.int32)
            Z_bin = np.digitize(world_z, bins=z_bins).astype(np.int32)
            
            # Filter out invalid values
            isvalid = (
                (X_bin >= 0) & (X_bin < map_size) &
                (Y_bin >= 0) & (Y_bin < map_size) &
                (Z_bin >= 0) & (Z_bin < n_z_bins) &
                isnotnan
            )

            # Flatten arrays
            flat_X = X_bin[isvalid]
            flat_Y = Y_bin[isvalid]
            flat_Z = Z_bin[isvalid]
            
            for x, y, z in zip(flat_X, flat_Y, flat_Z):
                counts[y, x, z] += 1  # Note: (y, x, z) because of image row-major order

            return counts, isvalid[..., None]

        # Build occupancy grid from the height map
        geocentric_flat, is_valids = bin_height_map_to_occupancy(latest_height_map, self.z_bins)

        self.map = geocentric_flat

        # Create ground truth occupancy maps.
        map_gt = (self.map[:, :, 1] + self.map[:, :, 2]) // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        wall_map_gt = self.map[:, :, 2] // self.obs_threshold
        wall_map_gt[wall_map_gt >= 0.5] = 1.0
        wall_map_gt[wall_map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        # Adjust final maps: First vertically flip, then rotate 90° counterclockwise.
        

        return map_gt, explored_gt, wall_map_gt