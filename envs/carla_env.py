import numpy as np
import gymnasium as gym
from gymnasium import spaces
import carla
import random
import math
import sys
import os
import cv2

# Add agents module from CARLA installation
current_dir = os.path.dirname(os.path.abspath(__file__))
carla_path = os.path.join(current_dir, 'Carla', 'WindowsNoEditor', 'PythonAPI', 'carla')
sys.path.insert(0, carla_path)

from agents.navigation.global_route_planner import GlobalRoutePlanner


class CarlaCNNEnv(gym.Env):
    """CARLA Environment with CNN-friendly observations"""

    def __init__(self, host='localhost', port=2000, num_surrounding_vehicles=30, max_steps=1000, **kwargs):
        super().__init__()

        self.client = carla.Client(host, port)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.camera_image_fv = None
        self.camera_image_bev = None
        self.num_surrounding_vehicles = num_surrounding_vehicles
        self.collision_hist = []
        self.route_completed = False
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize sensor references
        self.collision_sensor = None
        self.camera_fv = None
        self.camera_bev = None
        self.vehicle = None

        # Initialize waypoint tracking
        self.waypoint_queue = None
        self.current_wp_index = 0
        self.destination_list = []  # Store waypoints for path overlay

        # Image dimensions for CNN
        self.image_width = 64
        self.image_height = 64

        # Define observation space as a dictionary
        self.observation_space = spaces.Dict({
            'bev': spaces.Box(low=0, high=255, shape=(3, self.image_height, self.image_width), dtype=np.uint8),
            'fv': spaces.Box(low=0, high=255, shape=(3, self.image_height, self.image_width), dtype=np.uint8),
            'dynamics': spaces.Box(low=-1000, high=1000, shape=(7,), dtype=np.float32)
        })

        # Action space: [steering, acceleration]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # CLEANUP: Destroy previous sensors and vehicle if they exist
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None

        if self.camera_fv is not None:
            try:
                self.camera_fv.stop()
                self.camera_fv.destroy()
            except:
                pass
            self.camera_fv = None

        if self.camera_bev is not None:
            try:
                self.camera_bev.stop()
                self.camera_bev.destroy()
            except:
                pass
            self.camera_bev = None

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass
            self.vehicle = None

        # Reset camera images
        self.camera_image_fv = None
        self.camera_image_bev = None

        # Reset waypoint tracking
        self.waypoint_queue = None
        self.current_wp_index = 0
        self.destination_list = []  # Reset destination list

        # Set Synchronous Mode at Reset
        self.init_settings = self.world.get_settings()
        self.settings = self.world.get_settings()

        # Set synchronous mode with fixed delta seconds
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(self.settings)

        # Initialize traffic manager if not exists
        if not hasattr(self, 'my_tm'):
            self.my_tm = self.client.get_trafficmanager(8000)
        self.my_tm.set_synchronous_mode(True)
        self.world.tick()

        # Vehicle Spawning and Selection
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle_bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')

        # Ego Vehicle Selection
        self.vehicle = None
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, random.choice(spawn_points))
            if self.vehicle is None:
                self.world.tick()

        # Initialize collision sensor
        collision_bp = self.bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
        self.collision_hist = []
        self.route_completed = False

        # Initializing Surrounding Vehicles
        for _ in range(self.num_surrounding_vehicles):
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        # Surrounding Vehicle Control Initialization
        for v in self.world.get_actors().filter('*vehicle*'):
            if v.id == self.vehicle.id:
                continue
            v.set_autopilot(True, self.my_tm.get_port())

        # Update Timestep
        self.world.tick()

        # Ego Vehicle Camera Sensor Initialization - SEMANTIC SEGMENTATION
        self.camera_bp = self.bp_lib.find('sensor.camera.semantic_segmentation')

        # Set camera resolution to match our image dimensions
        self.camera_bp.set_attribute('image_size_x', str(self.image_width))
        self.camera_bp.set_attribute('image_size_y', str(self.image_height))

        self.camera_init_trans_fv = carla.Transform(carla.Location(z=2))
        self.camera_init_trans_bev = carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))

        # Attach Cameras to Vehicle
        self.camera_fv = self.world.spawn_actor(self.camera_bp, self.camera_init_trans_fv, attach_to=self.vehicle)
        self.camera_bev = self.world.spawn_actor(self.camera_bp, self.camera_init_trans_bev, attach_to=self.vehicle)

        # Listen to cameras
        self.camera_fv.listen(lambda data: self.process_image(data, camera_type='fv'))
        self.camera_bev.listen(lambda data: self.process_image(data, camera_type='bev'))

        # Reset previous state variables for reward calculation
        self.previous_x_position = None
        self.previous_y_position = None
        self.previous_steer = 0.0

        # Wait for cameras to be ready
        self.world.tick()
        self.world.tick()

        observation = self.return_state()
        info = {}
        return observation, info

    def step(self, action):
        # Increment step counter
        self.current_step += 1

        # Apply action
        self.return_action(action)
        self.world.tick()

        # Get observation
        observation = self.return_state()

        # Return Reward for Action
        reward = self.return_reward(action)

        # Check termination conditions
        terminated = len(self.collision_hist) > 0 or self.route_completed

        # Check if episode should be truncated (max steps reached)
        truncated = self.current_step >= self.max_steps

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        # Properly cleanup all sensors and vehicle
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass

        if self.camera_fv is not None:
            try:
                self.camera_fv.stop()
                self.camera_fv.destroy()
            except:
                pass

        if self.camera_bev is not None:
            try:
                self.camera_bev.stop()
                self.camera_bev.destroy()
            except:
                pass

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass

        # Always disable sync mode before the script ends
        try:
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)
            self.my_tm.set_synchronous_mode(False)
        except:
            pass

    # Helper Functions
    def process_image(self, data, camera_type):
        """Convert CARLA semantic segmentation to colored RGB and overlay waypoint path"""

        # Convert raw data to numpy array
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))  # BGRA

        # CARLA semantic segmentation uses the RED channel for class labels
        semantic_tags = array[:, :, 2]  # Red channel

        # Create RGB output image - start with black
        height, width = semantic_tags.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define color mapping using CARLA's actual class IDs
        carla_to_color = {
            1: [128, 64, 128],  # Roads -> Purple
            14: [0, 0, 142],  # Car -> Dark blue
            15: [0, 0, 70],  # Truck -> Dark blue (similar)
            16: [0, 60, 100],  # Bus -> Dark cyan
            18: [0, 0, 230],  # Motorcycle -> Blue
            12: [220, 20, 60],  # Pedestrian -> Red
            13: [255, 0, 0],  # Rider -> Red (similar)
            24: [255, 255, 255],  # RoadLine -> White
            2: [244, 35, 232],  # SideWalks -> Pink/Magenta
        }

        # Apply color mapping
        for class_id, color in carla_to_color.items():
            mask = semantic_tags == class_id
            rgb_image[mask] = color

        # Overlay waypoint path on BEV camera only
        if camera_type == 'bev' and hasattr(self, 'destination_list') and len(self.destination_list) > 0:
            try:
                # Get remaining waypoints from current position
                n_waypoints = 100
                start_idx = self.current_wp_index if hasattr(self, 'current_wp_index') else 0
                end_idx = min(start_idx + n_waypoints, len(self.destination_list))
                waypoints = self.destination_list[start_idx:end_idx]
                
                # Only proceed if we have waypoints and vehicle exists
                if len(waypoints) > 0 and self.vehicle is not None and self.camera_bev is not None:
                    # Coordinate conversion setup
                    vehicle_transform = self.vehicle.get_transform()
                    camera_location = self.camera_bev.get_transform().location
                    center_x, center_y = width // 2, height // 2
                    pixels_per_meter = 10
                    vehicle_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
                    
                    # Rotation: 270° counter-clockwise
                    rotation_angle = vehicle_yaw - np.pi / 2
                    
                    # Extract lane center points
                    center_points = []
                    
                    for wp in waypoints:
                        dx = wp.transform.location.x - camera_location.x
                        dy = wp.transform.location.y - camera_location.y
                        
                        # Apply 270° counter-clockwise rotation + vehicle yaw
                        rotated_x = dx * np.cos(rotation_angle) + dy * np.sin(rotation_angle)
                        rotated_y = -dx * np.sin(rotation_angle) + dy * np.cos(rotation_angle)
                        
                        # Mirror horizontally
                        rotated_x = -rotated_x
                        
                        pixel_x = int(center_x + rotated_x * pixels_per_meter)
                        pixel_y = int(center_y - rotated_y * pixels_per_meter)
                        
                        if 0 <= pixel_x < width and 0 <= pixel_y < height:
                            center_points.append((pixel_x, pixel_y))
                    
                    # Draw blue line
                    if len(center_points) > 1:
                        center_array = np.array(center_points, dtype=np.int32)
                        cv2.polylines(rgb_image, [center_array], False, (66, 133, 244), 5, cv2.LINE_AA)
            except Exception as e:
                # Silently fail if path overlay fails - don't crash the environment
                pass

        # Overlay waypoint path on FV camera
        if camera_type == 'fv' and hasattr(self, 'destination_list') and len(self.destination_list) > 0:
            try:
                # Get remaining waypoints from current position
                n_waypoints = 100
                start_idx = self.current_wp_index if hasattr(self, 'current_wp_index') else 0
                end_idx = min(start_idx + n_waypoints, len(self.destination_list))
                waypoints = self.destination_list[start_idx:end_idx]
                
                # Only proceed if we have waypoints and vehicle exists
                if len(waypoints) > 0 and self.vehicle is not None and self.camera_fv is not None:
                    # Camera transform and intrinsics
                    camera_transform = self.camera_fv.get_transform()
                    camera_location = camera_transform.location
                    
                    # Camera intrinsics for perspective projection
                    image_w = width
                    image_h = height
                    fov = 90.0  # Default CARLA FOV
                    
                    # Calculate focal length from FOV
                    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
                    
                    # Camera intrinsic matrix (K matrix)
                    K = np.array([
                        [focal, 0.0, image_w / 2.0],
                        [0.0, focal, image_h / 2.0],
                        [0.0, 0.0, 1.0]
                    ])
                    
                    # Get camera's basis vectors (accounts for pitch, yaw, roll)
                    forward = camera_transform.get_forward_vector()
                    right = camera_transform.get_right_vector()
                    up = camera_transform.get_up_vector()
                    
                    # Extract lane center points
                    center_points = []
                    
                    for wp in waypoints:
                        # Translate world point to camera origin
                        dx = wp.transform.location.x - camera_location.x
                        dy = wp.transform.location.y - camera_location.y
                        dz = wp.transform.location.z - camera_location.z
                        
                        # Transform to camera coordinate system using basis vectors
                        camera_x = dx * right.x + dy * right.y + dz * right.z
                        camera_y = -(dx * up.x + dy * up.y + dz * up.z)  # Negate for screen coords
                        camera_z = dx * forward.x + dy * forward.y + dz * forward.z
                        
                        # Filter points behind camera
                        if camera_z <= 0.1:  # Small threshold
                            continue
                        
                        # Perspective projection using intrinsic matrix
                        point_3d = np.array([camera_x, camera_y, camera_z])
                        point_2d = K @ point_3d
                        
                        # Perspective division (divide by depth)
                        pixel_x = int(point_2d[0] / point_2d[2])
                        pixel_y = int(point_2d[1] / point_2d[2])
                        
                        # Only include points within image bounds
                        if 0 <= pixel_x < width and 0 <= pixel_y < height:
                            center_points.append((pixel_x, pixel_y))
                    
                    # Draw blue line
                    if len(center_points) > 1:
                        center_array = np.array(center_points, dtype=np.int32)
                        cv2.polylines(rgb_image, [center_array], False, (66, 133, 244), 5, cv2.LINE_AA)
            except Exception as e:
                # Silently fail if path overlay fails - don't crash the environment
                pass

        # Store the processed image
        if camera_type == 'fv':
            self.camera_image_fv = rgb_image
        elif camera_type == 'bev':
            self.camera_image_bev = rgb_image

    def return_state(self):
        """Return structured state with images and dynamics"""
        # Get vehicle position
        self.x_position = self.vehicle.get_location().x
        self.y_position = self.vehicle.get_location().y

        # Get velocity
        self.x_velocity = self.vehicle.get_velocity().x
        self.y_velocity = self.vehicle.get_velocity().y
        self.foward_speed = math.sqrt(self.x_velocity ** 2 + self.y_velocity ** 2)

        # Get heading
        self.vehicle_heading = self.vehicle.get_transform().rotation.yaw

        # Get waypoint position
        self.x_waypoint, self.y_waypoint, self.road_heading = self.return_destination_waypoint_coordinates()

        # Calculate relative coordinates
        self.x_sr = self.x_waypoint - self.x_position
        self.y_sr = self.y_waypoint - self.y_position
        self.theta_sr = self.road_heading - self.vehicle_heading

        # Get lane information
        vehicle_position = self.vehicle.get_location()
        map = self.world.get_map()
        current_waypoint = map.get_waypoint(location=vehicle_position)
        left_waypoint = current_waypoint.get_left_lane()
        right_waypoint = current_waypoint.get_right_lane()

        # Check lane existence
        current_lane = 1 if (current_waypoint and current_waypoint.lane_type == carla.libcarla.LaneType.Driving) else 0
        left_lane = 1 if (left_waypoint and left_waypoint.lane_type == carla.libcarla.LaneType.Driving) else 0
        right_lane = 1 if (right_waypoint and right_waypoint.lane_type == carla.libcarla.LaneType.Driving) else 0

        # Construct dynamics state
        lane_encoding = np.array([current_lane, left_lane, right_lane], dtype=np.float32)
        dynamics_state = np.array([self.x_sr, self.y_sr, self.theta_sr, self.foward_speed], dtype=np.float32)
        dynamics = np.concatenate([dynamics_state, lane_encoding])

        # Check if images are ready
        if self.camera_image_bev is None or self.camera_image_fv is None:
            bev = np.zeros((3, self.image_height, self.image_width), dtype=np.uint8)
            fv = np.zeros((3, self.image_height, self.image_width), dtype=np.uint8)
        else:
            # Transpose from (H, W, C) to (C, H, W) for PyTorch
            bev = np.transpose(self.camera_image_bev, (2, 0, 1)).astype(np.uint8)
            fv = np.transpose(self.camera_image_fv, (2, 0, 1)).astype(np.uint8)

        return {
            'bev': bev,
            'fv': fv,
            'dynamics': dynamics
        }

    def return_action(self, model_output):
        """Apply the action to the vehicle"""
        # Handle both 1D and 2D arrays
        if isinstance(model_output, np.ndarray) and model_output.ndim > 1:
            model_output = model_output.squeeze()

        steering_output = float(model_output[0])
        acceleration_output = float(model_output[1])

        # Clip values to valid range
        steering_output = np.clip(steering_output, -1.0, 1.0)
        acceleration_output = np.clip(acceleration_output, -1.0, 1.0)

        # Apply control
        if acceleration_output < 0:
            control = carla.VehicleControl(
                throttle=0.0,
                steer=steering_output,
                brake=abs(acceleration_output),
                reverse=False
            )
        else:
            control = carla.VehicleControl(
                throttle=acceleration_output,
                steer=steering_output,
                brake=0.0,
                reverse=False
            )

        self.vehicle.apply_control(control)

    def return_destination_waypoint_coordinates(self):
        """Get destination waypoint coordinates and populate destination_list"""
        if not hasattr(self, 'waypoint_queue') or not self.waypoint_queue:
            if not hasattr(self, 'planner'):
                self.planner = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2.0)

            spawn_points = self.world.get_map().get_spawn_points()
            destination = random.choice(spawn_points).location
            self.waypoint_queue = self.planner.trace_route(self.vehicle.get_location(), destination)
            self.current_wp_index = 0
            
            # Extract and store waypoints in destination_list
            self.destination_list = [wp for wp, _ in self.waypoint_queue]

        destination_waypoint, _ = self.waypoint_queue[self.current_wp_index]

        # Update waypoint when vehicle approaches
        self.distance_to_waypoint = self.vehicle.get_location().distance(destination_waypoint.transform.location)
        if self.distance_to_waypoint < 5.0:
            self.current_wp_index += 1

            if self.current_wp_index >= len(self.waypoint_queue):
                self.current_wp_index = len(self.waypoint_queue) - 1
                self.route_completed = True
            else:
                self.route_completed = False

            destination_waypoint, _ = self.waypoint_queue[self.current_wp_index]
        else:
            self.route_completed = False

        destination_x_position = destination_waypoint.transform.location.x
        destination_y_position = destination_waypoint.transform.location.y
        destination_yaw = destination_waypoint.transform.rotation.yaw

        return destination_x_position, destination_y_position, destination_yaw

    def return_reward(self, action):
        """Compute reward based on safety, efficiency, and comfort"""
        # Handle both 1D and 2D arrays
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze()

        a_steer = float(action[0])
        a_acc = float(action[1])

        # ==================== SAFETY REWARD ====================
        collision_occurred = len(self.collision_hist) > 0
        r_collision = -100.0 if collision_occurred else 0.0

        d_lateral = np.sqrt(self.x_sr ** 2 + self.y_sr ** 2)
        d_lateral_normalized = np.clip(d_lateral / 3.5, 0.0, 1.0)
        r_deviation = -10.0 * d_lateral_normalized

        theta_error = np.deg2rad(self.theta_sr)
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
        theta_error_normalized = np.abs(theta_error) / np.pi
        r_heading = -5.0 * theta_error_normalized

        v_ego = self.foward_speed
        v_min, v_max = 5.0, 30.0
        if v_ego < v_min:
            r_speed_limit = -2.0 * (v_min - v_ego)
        elif v_ego > v_max:
            r_speed_limit = -5.0 * (v_ego - v_max)
        else:
            r_speed_limit = 0.0

        r_safety = r_collision + r_deviation + r_heading + r_speed_limit

        # ==================== EFFICIENCY REWARD ====================
        v_target = 25.0
        r_speed = 0.5 * (self.foward_speed / v_target)

        if self.previous_x_position is None:
            self.previous_x_position = self.x_position
            self.previous_y_position = self.y_position

        d_delta = np.sqrt((self.x_position - self.previous_x_position) ** 2 +
                          (self.y_position - self.previous_y_position) ** 2)
        r_distance = 0.1 * d_delta
        self.previous_x_position = self.x_position
        self.previous_y_position = self.y_position

        r_efficiency = r_speed + r_distance

        # ==================== COMFORT REWARD ====================
        delta_a_steer = np.abs(a_steer - self.previous_steer)
        r_steering_change = -2.0 * delta_a_steer
        self.previous_steer = a_steer

        a_y_vec = self.vehicle.get_acceleration()
        a_y_lateral = np.sqrt(a_y_vec.x ** 2 + a_y_vec.y ** 2)

        a_y_threshold = 2.0
        r_lateral_acc = -3.0 * max(0.0, a_y_lateral - a_y_threshold)

        r_comfort = r_steering_change + r_lateral_acc

        # ==================== TOTAL REWARD ====================
        w_safety = 1.0
        w_efficiency = 0.3
        w_comfort = 0.2

        r_t = w_safety * r_safety + w_efficiency * r_efficiency + w_comfort * r_comfort

        return r_t