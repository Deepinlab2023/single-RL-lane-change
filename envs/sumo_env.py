import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import traci.constants as tc
import time
import xml.etree.ElementTree as ET
import os
import random


class SumoEnv(gym.Env):
    """SUMO-based Highway Lane Change Environment"""  
    def __init__(self, gui=False, step_length=0.1, L1=20, L2=50, 
                 v_max=16.7, max_steps=100, w_efficiency=1.0, w_safety=1.0,
                 road_type='highway', decision_steps=10, render_mode=None, delta_t=1.0, tau=3.0, 
                 safe_distance=5.0, route_distance=500, **kwargs):
        super().__init__()
        
        # Environment parameters
        self.gui = gui
        self.render_mode = render_mode
        self.step_length = step_length
        self.L1 = L1
        self.L2 = L2
        self.delta_t = delta_t
        self.v_max = v_max
        self.max_decision_steps = max_steps
        self.w_efficiency = w_efficiency
        self.w_safety = w_safety
        self.tau = tau
        self.decision_steps = decision_steps
        self.max_simulation_steps = self.max_decision_steps * self.decision_steps
        self.safe_distance = safe_distance
        
        # File paths
        base_dir = kwargs.get('base_dir', os.path.dirname(__file__))
        self.config_file = os.path.join(base_dir, road_type, 'env.sumocfg')
        self.network_file = os.path.join(base_dir, road_type, 'env.net.xml')
        
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        if not os.path.exists(self.network_file):
            raise FileNotFoundError(f"Network file not found: {self.network_file}")
        
        # Vehicle and simulation state
        self.ego_id = "ego"
        self.sumo_running = False
        self.base_label = f"sumo_{id(self)}"          
        self.label = self.base_label 
        self.steps = 0
        self.decision_step_count = 0
        self.episode_count = 0 
        self.vehicle_spawn_counter = 0  
        self.conn = None
        self.last_reward = 0.0
        self.collision_occurred = False 
        self.collision_threshold = 3.0
        self.previous_lane = None
        self.termination_reason = None
        
        # Network properties
        self.edge_id = None
        self.edge_length = None
        self.num_lanes = None
        self.lane_width = 3.2
        
        # Read network to get edge name
        self._read_network_config()
        
        # Traffic management
        self.min_despawn_distance = 300
        
        # Rendering state
        self.screen = None
        self.clock = None
        self.font = None
        self.scroll_offset = 0.0
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        
        max_coord = 10000.0
        max_vel = self.v_max * 2
        max_rel_dist = max(self.L1, self.L2) + 10
        max_lat_dist = 100.0
        
        obs_low = np.array([
            -max_coord, -max_coord, -max_vel, -max_vel,
            *[-max_rel_dist, -max_lat_dist, -max_vel, -max_vel] * 4
        ], dtype=np.float32)
        
        obs_high = np.array([
            max_coord, max_coord, max_vel, max_vel,
            *[max_rel_dist, max_lat_dist, max_vel, max_vel] * 4
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Stop previous simulation
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
            time.sleep(0.5)
        
        # Initialize episode state
        self.episode_count += 1 
        self.vehicle_spawn_counter = 0
        self.label = f"{self.base_label}_ep{self.episode_count}"
        self.scroll_offset = 0.0
        
        # Start SUMO and read network
        self._start_sumo()
        self.conn.simulationStep()
        
        # Reset tracking variables
        self.steps = 0
        self.decision_step_count = 0
        self.last_action = None
        self.last_reward = 0.0
        self.collision_occurred = False
        self.collision_vehicle_id = None
        self.route_end = False
        self.previous_lane = None
        self.termination_reason = None
        self.min_traffic = random.randint(1, 5)
        self.max_traffic = random.randint(10, 15)

        # Spawn initial vehicles
        target_vehicles = random.randint(10, 20)
        self._spawn_random_vehicles(target_vehicles)

        for _ in range(10):
            self.conn.simulationStep()
            self.steps += 1
        
        # Select ego vehicle
        all_vehicles = self.conn.vehicle.getIDList()
        
        if len(all_vehicles) == 0:
            return self.reset(seed, options)
            
        self.ego_id = np.random.choice(all_vehicles)
                
        # Configure ego vehicle control
        try:
            self.conn.vehicle.setSpeedMode(self.ego_id, 0)
            self.conn.vehicle.setLaneChangeMode(self.ego_id, 0)
            self.previous_lane = self.conn.vehicle.getLaneIndex(self.ego_id)
        except traci.TraCIException:
            return self.reset(seed, options)
            
        
        initial_state = self._get_observation()
        
        return initial_state, {}

    def step(self, action):
        """Execute environment step"""
        
        # Initialize step variables
        total_reward = 0.0
        truncated = False
                     
        # Get current state
        try:
            current_speed = self.conn.vehicle.getSpeed(self.ego_id)
            current_lane = self.conn.vehicle.getLaneIndex(self.ego_id)
        except traci.TraCIException:
            self.termination_reason = "TRACI ERROR"
            return self._get_observation(), 0.0, True, False, {'traci_error': True}

        # Apply speed actions
        if action == 0:
            new_speed = max(0, current_speed - self.delta_t)
            self.conn.vehicle.setSpeed(self.ego_id, new_speed)
        elif action == 1:
            pass  # Maintain current speed
        elif action == 2:
            new_speed = min(self.v_max, current_speed + self.delta_t)
            self.conn.vehicle.setSpeed(self.ego_id, new_speed)

        # Apply lane change actions
        if action == 3:
            if current_lane > 0:
                self.conn.vehicle.changeLane(self.ego_id, current_lane - 1, self.tau)
        elif action == 4:
            if current_lane < self.num_lanes - 1:
                self.conn.vehicle.changeLane(self.ego_id, current_lane + 1, self.tau)
        
        # Advance simulation
        for _ in range(self.decision_steps):               
            self._manage_traffic()
            self.conn.simulationStep()
            self.steps += 1

            # Terminal Conditions
            if self.steps >= self.max_simulation_steps:
                self.route_end = True

            self._check_collision()

            step_reward = self._compute_reward()
            total_reward += step_reward
            

            # Set termination reasons
            if self.collision_occurred:
                self.termination_reason = "COLLISION"
            elif self.route_end:
                self.termination_reason = "ROUTE COMPLETED"
            
            if self.render_mode == 'human':
                self.render()
                    
            if self.collision_occurred or self.route_end:
                break

        # Prepare return values
        next_state = self._get_observation()
        
        self.decision_step_count += 1
                        
        done = self.collision_occurred or self.route_end
        
        self.last_reward = total_reward
        
        return next_state, total_reward, done, truncated, {'collision': self.collision_occurred}

    def _get_observation(self):
        """Construct observation vector"""
        
        current_vehicles = self.conn.vehicle.getIDList()
        
        if self.ego_id not in current_vehicles:
            return np.zeros(20, dtype=np.float32)
        
        # Ego vehicle state
        try:
            ego_x, ego_y = self.conn.vehicle.getPosition(self.ego_id)
            ego_speed = self.conn.vehicle.getSpeed(self.ego_id)
            ego_angle = np.radians(self.conn.vehicle.getAngle(self.ego_id))

        except traci.TraCIException:
            return np.zeros(20, dtype=np.float32)
            
        ego_vx = ego_speed * np.cos(ego_angle)
        ego_vy = ego_speed * np.sin(ego_angle)
        
        # Nearby vehicles state
        nearby_vehicles = []
        
        for veh_id in current_vehicles:
            if veh_id == self.ego_id:
                continue
            
            try:
                veh_x, veh_y = self.conn.vehicle.getPosition(veh_id)
                veh_speed = self.conn.vehicle.getSpeed(veh_id)
                veh_angle = np.radians(self.conn.vehicle.getAngle(veh_id))
            except traci.TraCIException:
                continue
            
            dx = veh_x - ego_x
            dy = veh_y - ego_y
            
            if -self.L1 <= dx <= self.L2:
                veh_vx = veh_speed * np.cos(veh_angle)
                veh_vy = veh_speed * np.sin(veh_angle)
                
                dvx = veh_vx - ego_vx
                dvy = veh_vy - ego_vy
                
                nearby_vehicles.append([dx, dy, dvx, dvy])
        
        # Select and pad vehicles
        nearby_vehicles.sort(key=lambda v: abs(v[0]))
        selected_vehicles = nearby_vehicles[:4]
        
        while len(selected_vehicles) < 4:
            selected_vehicles.append([0, 0, 0, 0])
        
        # Build observation
        observation = [ego_x, ego_y, ego_vx, ego_vy]
        for vehicle in selected_vehicles:
            observation.extend(vehicle)
        
        return np.array(observation, dtype=np.float32)

    def _manage_traffic(self):
        """Dynamic traffic spawning and despawning"""
        
        current_vehicles = self.conn.vehicle.getIDList()
        
        if self.ego_id not in current_vehicles:
            return
            
        try:
            ego_pos = self.conn.vehicle.getLanePosition(self.ego_id)
        except traci.TraCIException:
            return
        
        # Despawn distant vehicles
        for veh_id in current_vehicles:
            if veh_id == self.ego_id:
                continue
            
            try:
                veh_pos = self.conn.vehicle.getLanePosition(veh_id)
                distance = abs(veh_pos - ego_pos)
                
                if distance > self.min_despawn_distance:
                    self.conn.vehicle.remove(veh_id)
            except traci.TraCIException:
                continue
        
        # Calculate spawn demand
        current_vehicles = self.conn.vehicle.getIDList()
        vehicle_count = len(current_vehicles) - 1

        traffic_demand = random.randint(self.min_traffic, self.max_traffic)
        
        if vehicle_count < traffic_demand:
            demand = traffic_demand - vehicle_count
        else:
            demand = 0
  
        # Spawn parameters
        min_spawn_distance = self.min_despawn_distance
        max_spawn_distance = 400
        safe_spawn_clearance = 15
        
        # Spawn new vehicles
        for _ in range(demand):
            max_spawn_attempts = 10
            
            for _ in range(max_spawn_attempts):
                # Random spawn location
                spawn_distance = np.random.uniform(min_spawn_distance, max_spawn_distance)
                direction = np.random.uniform(0, 1)
                
                if direction < 0.5:
                    spawn_position = ego_pos - spawn_distance
                else:
                    spawn_position = ego_pos + spawn_distance
                
                spawn_lane = random.randint(0, self.num_lanes - 1)
                
                # Validate spawn position 
                if not (100 < spawn_position < self.edge_length - 100):
                    continue 
                
                # Check safety clearance
                position_is_safe = True
                current_vehicles = self.conn.vehicle.getIDList()
                for existing_veh_id in current_vehicles:
                    try:
                        existing_pos = self.conn.vehicle.getLanePosition(existing_veh_id)
                        existing_lane = self.conn.vehicle.getLaneIndex(existing_veh_id)
                        
                        longitudinal_distance = abs(existing_pos - spawn_position)
                        lane_difference = abs(existing_lane - spawn_lane)
                        
                        if longitudinal_distance < safe_spawn_clearance and lane_difference <= 1:
                            position_is_safe = False
                            break
                    except traci.TraCIException:
                        continue
                
                if not position_is_safe:
                    continue
                
                # Spawn vehicle
                spawn_speed = np.random.uniform(12, 16.7)
                veh_id = f"dyn_{self.episode_count}_{self.vehicle_spawn_counter}"
                self.vehicle_spawn_counter += 1
                
                try:
                    self.conn.vehicle.add(
                        veh_id, 
                        "route0", 
                        typeID="car",
                        departLane=str(spawn_lane),
                        departPos=str(spawn_position),
                        departSpeed=str(spawn_speed)
                    )
                    break
                except traci.TraCIException:
                    continue
                    
    def _compute_reward(self):
        """Compute reward signal"""
        
        try:
            current_lane = self.conn.vehicle.getLaneIndex(self.ego_id)
            current_speed = self.conn.vehicle.getSpeed(self.ego_id)
            
            # Check collision
            colliding_vehicles = self.conn.simulation.getCollidingVehiclesIDList()
            collision_occurred = self.ego_id in colliding_vehicles
            
            # Check lane change
            lane_changed = False
            if self.previous_lane is not None and current_lane != self.previous_lane:
                lane_changed = True
            
            self.previous_lane = current_lane
                        
            # Calculate reward components
            reward = 0.0
            
            # Collision penalty
            if collision_occurred:
                reward = -5.0
            
            # Route completion bonus
            if self.route_end and not collision_occurred:
                reward += 5.0
            
            # Lane change penalty
            if lane_changed:
                reward -= 1.0
            
            # Efficiency reward based on speed
            v_min = 0.0
            efficiency_reward = -1.0 + 2.0 * (current_speed - v_min) / (self.v_max - v_min)
            reward += efficiency_reward
            
        except traci.TraCIException:
            return 0.0

        return reward

    def _check_collision(self):
        """Check collision with same-lane vehicles"""
        try:
            # Check vehicles in the same lane
            ego_lane = self.conn.vehicle.getLaneIndex(self.ego_id)
            ego_pos = self.conn.vehicle.getLanePosition(self.ego_id)
            
            current_vehicles = self.conn.vehicle.getIDList()
            for veh_id in current_vehicles:
                if veh_id == self.ego_id:
                    continue
                
                try:
                    veh_lane = self.conn.vehicle.getLaneIndex(veh_id)
                    
                    if veh_lane != ego_lane:
                        continue
                    
                    veh_pos = self.conn.vehicle.getLanePosition(veh_id)
                    distance = abs(veh_pos - ego_pos)
                    
                    if distance < self.safe_distance:
                        self.collision_occurred = True
                        self.collision_vehicle_id = veh_id
                        return
                        
                except traci.TraCIException:
                    continue
            
        except traci.TraCIException:
            pass

    def _read_network_config(self):
        """Read network configuration - FIXED to read actual network values"""
        # Default values
        self.edge_id = "highway"
        self.num_lanes = 4
        self.edge_length = 2000.0
        self.lane_width = 3.2
        
        try:
            tree = ET.parse(self.network_file)
            root = tree.getroot()
            
            for edge in root.findall('.//edge'):
                eid = edge.get('id')
                if eid and not eid.startswith(':'):
                    self.edge_id = eid
                    lanes = edge.findall('.//lane')
                    self.num_lanes = len(lanes)
                    if lanes:
                        # Parse length correctly - your network has 20100m length
                        self.edge_length = float(lanes[0].get('length', 2000.0))
                        self.lane_width = float(lanes[0].get('width', 3.2))
                    break
            
        except Exception as e:
            print(f"Warning: Could not read network file: {e}")

    def _spawn_random_vehicles(self, num_vehicles):
        """Spawn initial vehicle distribution - FIXED spawn range"""
        vehicle_count = 0
        spawned_positions = []
        safe_spawn_clearance = 15

        # Use safer spawn range based on actual edge length
        spawn_start = 100
        spawn_end = min(1000, self.edge_length - 100)  # Limit initial spawn to first 1km

        for i in range(num_vehicles * 3): 
            if vehicle_count >= num_vehicles:
                break

            spawn_lane = random.randint(0, self.num_lanes - 1)
            spawn_position = random.uniform(spawn_start, spawn_end)
            spawn_speed = random.uniform(11, 16)
            
            position_is_safe = True
            for (existing_pos, existing_lane) in spawned_positions:
                longitudinal_distance = abs(existing_pos - spawn_position)
                lane_difference = abs(existing_lane - spawn_lane)
                
                if longitudinal_distance < safe_spawn_clearance and lane_difference <= 1:
                    position_is_safe = False
                    break
            
            if not position_is_safe:
                continue
            
            veh_id = f"random_{self.episode_count}_{vehicle_count}_{i}"
            
            try:
                self.conn.vehicle.add(
                    veh_id,
                    "route0",
                    typeID="car",
                    departLane=str(spawn_lane),
                    departPos=str(spawn_position),
                    departSpeed=str(spawn_speed)
                )
                spawned_positions.append((spawn_position, spawn_lane)) 
                vehicle_count += 1
            except traci.TraCIException as e:
                pass

    def _start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--collision.action", "none",
            "--lateral-resolution", "1.0"
        ]
        
        traci.start(sumo_cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
        self.sumo_running = True

    def render(self, mode='human'):
        """Pygame rendering with scrolling effect - FIXED coordinate system"""
        if not self.sumo_running:
            return
        
        if self.ego_id not in self.conn.vehicle.getIDList():
            return
            
        try:
            import pygame
            
            pixels_per_meter = 10.0
            
            # Initialize Pygame
            if self.screen is None:
                pygame.init()
                self.screen_width = 1200
                self.screen_height = 700
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("SUMO Highway - Lane Change Environment")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 24)
                self.font_small = pygame.font.Font(None, 18)
                self.font_large = pygame.font.Font(None, 32)
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            # Get ego state
            ego_pos = self.conn.vehicle.getLanePosition(self.ego_id)
            ego_lane = self.conn.vehicle.getLaneIndex(self.ego_id)
            ego_speed = self.conn.vehicle.getSpeed(self.ego_id)
            
            # Update scroll
            self.scroll_offset += ego_speed * self.step_length * pixels_per_meter
            
            # Color palette
            ROAD_GRAY = (60, 60, 60)
            GRASS_GREEN = (34, 139, 34)
            LINE_WHITE = (255, 255, 255)
            LINE_YELLOW = (255, 215, 0)
            EGO_GREEN = (0, 255, 0)
            SKY_BLUE = (135, 206, 235)
            
            # Road dimensions
            lane_height_px = self.lane_width * pixels_per_meter
            road_height = lane_height_px * self.num_lanes
            road_top = (self.screen_height - road_height) / 2.0
            
            # Draw background
            self.screen.fill(SKY_BLUE)
            pygame.draw.rect(self.screen, GRASS_GREEN, (0, 0, self.screen_width, road_top))
            pygame.draw.rect(self.screen, GRASS_GREEN, (0, road_top + road_height, self.screen_width, self.screen_height))
            
            # Draw road
            pygame.draw.rect(self.screen, ROAD_GRAY, (0, road_top, self.screen_width, road_height))
            
            # Draw lane markings
            dash_length = 25
            dash_gap = 35
            dash_width = 4
            marking_offset = int(self.scroll_offset % (dash_length + dash_gap))
            
            for lane_idx in range(1, self.num_lanes):
                y = road_top + lane_idx * lane_height_px
                x = -marking_offset
                while x < self.screen_width:
                    pygame.draw.rect(self.screen, LINE_WHITE, 
                                   (x, y - dash_width // 2, dash_length, dash_width))
                    x += dash_length + dash_gap
            
            # Draw road edges
            pygame.draw.line(self.screen, LINE_YELLOW, (0, road_top), 
                           (self.screen_width, road_top), 6)
            pygame.draw.line(self.screen, LINE_YELLOW, (0, road_top + road_height), 
                           (self.screen_width, road_top + road_height), 6)
            
            # Get vehicles
            all_vehicles = self.conn.vehicle.getIDList()
            ego_screen_x = self.screen_width // 2
            
            try:
                ego_length_m = self.conn.vehicle.getLength(self.ego_id)
                ego_width_m = self.conn.vehicle.getWidth(self.ego_id)
            except:
                ego_length_m = 5.0
                ego_width_m = 1.8
            
            vehicle_length = int(ego_length_m * pixels_per_meter)
            vehicle_width = int(ego_width_m * pixels_per_meter)
            
            render_range = (self.screen_width / pixels_per_meter) / 2
            
            # Draw vehicles
            vehicles_drawn = 0
            for veh_id in all_vehicles:
                try:
                    veh_pos = self.conn.vehicle.getLanePosition(veh_id)
                    veh_lane = self.conn.vehicle.getLaneIndex(veh_id)
                    veh_speed = self.conn.vehicle.getSpeed(veh_id)
                except traci.TraCIException:
                    continue
                
                try:
                    veh_length_m = self.conn.vehicle.getLength(veh_id)
                    veh_width_m = self.conn.vehicle.getWidth(veh_id)
                    veh_length_px = int(veh_length_m * pixels_per_meter)
                    veh_width_px = int(veh_width_m * pixels_per_meter)
                except:
                    veh_length_px = vehicle_length
                    veh_width_px = vehicle_width
                
                rel_pos = veh_pos - ego_pos
                
                if abs(rel_pos) > render_range + 50:
                    continue
                
                veh_screen_x = ego_screen_x + int(rel_pos * pixels_per_meter)
                
                if -veh_length_px * 2 <= veh_screen_x <= self.screen_width + veh_length_px * 2:
                    vehicles_drawn += 1
                    
                    base_lane_y = road_top + veh_lane * lane_height_px
                    lane_center_y = base_lane_y + lane_height_px / 2.0
                    
                    lateral_offset_m = 0.0
                    try:
                        lateral_offset_m = self.conn.vehicle.getLateralLanePosition(veh_id)
                        if abs(lateral_offset_m) < 0.05:
                            lateral_offset_m = 0.0
                    except:
                        pass
                    
                    lateral_scale = pixels_per_meter
                    lateral_offset_px = lateral_offset_m * lateral_scale
                    
                    veh_y = int(lane_center_y + lateral_offset_px - veh_width_px / 2.0)
                    
                    vehicle_rect = pygame.Rect(
                        veh_screen_x - veh_length_px // 2, 
                        veh_y, 
                        veh_length_px, 
                        veh_width_px
                    )
                    
                    if veh_id == self.ego_id:
                        # Ego vehicle rendering
                        if self.collision_occurred:
                            ego_color = (255, 0, 0)
                            glow_color = (139, 0, 0)
                            label_text = "COLLISION!"
                            label_color = (255, 255, 0)
                        else:
                            ego_color = EGO_GREEN
                            glow_color = (0, 100, 0)
                            label_text = "EGO"
                            label_color = (255, 255, 0)
                        
                        glow_rect = vehicle_rect.inflate(4, 4)
                        pygame.draw.rect(self.screen, glow_color, glow_rect, border_radius=5)
                        pygame.draw.rect(self.screen, ego_color, vehicle_rect, border_radius=4)
                        
                        ego_label = self.font_small.render(label_text, True, label_color)
                        label_rect = ego_label.get_rect(center=(veh_screen_x, veh_y + veh_width_px // 2))
                        self.screen.blit(ego_label, label_rect)
                        
                        if veh_width_px > 10:
                            if self.collision_occurred:
                                windshield_color = (200, 0, 0)
                            else:
                                windshield_color = (0, 200, 0)
                            
                            windshield = pygame.Rect(
                                veh_screen_x - veh_length_px // 4,
                                veh_y + 2,
                                veh_length_px // 3,
                                max(2, veh_width_px - 4)
                            )
                            pygame.draw.rect(self.screen, windshield_color, windshield, border_radius=2)
                    else:
                        # Other vehicles rendering
                        if self.collision_occurred and veh_id == self.collision_vehicle_id:
                            color = (255, 0, 0)
                        else:
                            color = (100, 100, 255)
                        
                        shadow_rect = vehicle_rect.inflate(2, 2)
                        pygame.draw.rect(self.screen, (40, 40, 40), shadow_rect, border_radius=4)
                        pygame.draw.rect(self.screen, color, vehicle_rect, border_radius=3)
                        
                        if veh_width_px > 5:
                            if self.collision_occurred and veh_id == self.collision_vehicle_id:
                                windshield_color = (200, 0, 0)
                            else:
                                windshield_color = (60, 60, 180)
                            windshield = pygame.Rect(
                                veh_screen_x - veh_length_px // 4,
                                veh_y + 2,
                                veh_length_px // 3,
                                max(2, veh_width_px - 4)
                            )
                            pygame.draw.rect(self.screen, windshield_color, windshield, border_radius=1)
            
            # Draw HUD
            hud_bg = pygame.Surface((self.screen_width, 80))
            hud_bg.set_alpha(200)
            hud_bg.fill((0, 0, 0))
            self.screen.blit(hud_bg, (0, 0))
            
            # Display termination reason with different colors
            if self.termination_reason == "COLLISION":
                title = self.font_large.render("COLLISION", True, (255, 0, 0))
            elif self.termination_reason == "ROUTE COMPLETED":
                title = self.font_large.render("ROUTE COMPLETED", True, (0, 255, 0))
            elif self.termination_reason == "TRACI ERROR":
                title = self.font_large.render("TRACI ERROR", True, (255, 0, 255))
            else:
                title = self.font_large.render("SUMO Highway Environment", True, (255, 255, 255))
            self.screen.blit(title, (20, 10))
            
            stats = [
                f"Decision: {self.decision_step_count}/{self.max_decision_steps}",
                f"Sim Step: {self.steps}/{self.max_simulation_steps}",
                f"Speed: {ego_speed:.1f} m/s ({ego_speed * 3.6:.1f} km/h)",
                f"Lane: {ego_lane}",
                f"Position: {ego_pos:.0f}m",
                f"Vehicles: {len(all_vehicles)} (Rendered: {vehicles_drawn})",
                f"Reward: {self.last_reward:.3f}"
            ]
            
            x_offset = 20
            for i, stat in enumerate(stats):
                stat_text = self.font_small.render(stat, True, (255, 255, 255))
                self.screen.blit(stat_text, (x_offset, 45))
                x_offset += stat_text.get_width() + 30
            
            # Draw minimap
            minimap_width = 200
            minimap_height = 100
            minimap_x = self.screen_width - minimap_width - 20
            minimap_y = self.screen_height - minimap_height - 20
            
            pygame.draw.rect(self.screen, (0, 0, 0), 
                           (minimap_x - 5, minimap_y - 5, minimap_width + 10, minimap_height + 10))
            pygame.draw.rect(self.screen, (40, 40, 40), 
                           (minimap_x, minimap_y, minimap_width, minimap_height))
            
            mini_lane_height = minimap_height / self.num_lanes
            for i in range(self.num_lanes + 1):
                pygame.draw.line(self.screen, (100, 100, 100), 
                               (minimap_x, minimap_y + i * mini_lane_height),
                               (minimap_x + minimap_width, minimap_y + i * mini_lane_height), 1)
            
            # Draw minimap vehicles
            view_range = 200
            for veh_id in all_vehicles:
                try:
                    veh_pos = self.conn.vehicle.getLanePosition(veh_id)
                    veh_lane = self.conn.vehicle.getLaneIndex(veh_id)
                except traci.TraCIException:
                    continue
                
                rel_pos = veh_pos - ego_pos
                if -view_range <= rel_pos <= view_range:
                    mini_x = minimap_x + minimap_width // 2 + int((rel_pos / view_range) * minimap_width // 2)
                    mini_y = minimap_y + int((veh_lane + 0.5) * mini_lane_height)
                    
                    if veh_id == self.ego_id:
                        pygame.draw.circle(self.screen, EGO_GREEN, (mini_x, mini_y), 4)
                    else:
                        pygame.draw.circle(self.screen, (100, 150, 255), (mini_x, mini_y), 2)
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            if "video system not initialized" in str(e):
                self.screen = None
                return

    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
        
        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.sumo_running = False


# Register environment
try:
    from gymnasium.envs.registration import register
    register(
        id='sumo-v0',
        entry_point='__main__:SumoEnv',
        max_episode_steps=10000,
    )
except ImportError:
    pass
except Exception:
    pass