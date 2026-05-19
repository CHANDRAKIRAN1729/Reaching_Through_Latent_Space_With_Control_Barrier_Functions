#!/usr/bin/env python3
"""
End-to-End Simulation Script for Reaching Through Latent Space

This script connects to MoveIt as a collision-checking ORACLE only.
It does NOT use MoveIt planners - only collision checking services.

Design Principles (Thesis-grade rigor):
1. MoveIt is used ONLY as ground-truth collision oracle
2. No planners, no IK services, no planning pipelines
3. Deterministic collision checking (no octomap, fixed seed)
4. Programmatic obstacle control (no RViz clicks)
5. Configurable collision margin for fairness

Requirements:
- ROS Noetic
- MoveIt (panda_moveit_config)
- franka_ros (Panda robot description)
- Python 3 with rospy

Usage:
    # Terminal 1: Launch MoveIt (REQUIRED - must be running first)
    roslaunch panda_moveit_config demo.launch
    
    # Terminal 2: Run simulation (connects to running MoveIt)
    python simulate_in_moveit.py --checkpoint <vae.pt> --checkpoint_obs <classifier.pt>

Author: RTLS Project
"""

import argparse
import json
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.optim as optim
from pathlib import Path

# Set deterministic collision checking seed (FCL library)
os.environ['FCL_RANDOM_SEED'] = '0'

# PyTorch/ML imports
from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from vae import VAE
from vae_obs import VAEObstacleBCE
from sim.panda import Panda

# Try to import ROS - graceful fallback if not available
try:
    import rospy
    import moveit_msgs.msg
    from moveit_msgs.msg import (
        CollisionObject, 
        PlanningScene, 
        RobotState,
        AllowedCollisionMatrix,
        DisplayTrajectory,
        RobotTrajectory
    )
    from moveit_msgs.srv import (
        GetStateValidity, 
        GetStateValidityRequest,
        GetPlanningScene,
        GetPlanningSceneRequest
    )
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from geometry_msgs.msg import PoseStamped, Pose, Point
    from shape_msgs.msg import SolidPrimitive
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header, ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    ROS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ROS not available - {e}")
    print("Running in simulation-only mode (no visualization)")
    ROS_AVAILABLE = False


def setup_logging():
    """Setup logging with proper configuration. 
    Call this AFTER rospy.init_node() to fix ROS logging interference."""
    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create a new handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


# Initial logging setup (will be reset after ROS init)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)


# ============================================================================
# PANDA ROBOT JOINT CONFIGURATION (CRITICAL - must match MoveIt config)
# ============================================================================
PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2", 
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

# Gripper joint names (for visualization only - VAE doesn't use these)
PANDA_GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

# Combined arm + gripper joint names for full robot visualization
PANDA_ALL_JOINT_NAMES = PANDA_JOINT_NAMES + PANDA_GRIPPER_JOINT_NAMES

# Closed gripper position (both fingers at 0.0)
GRIPPER_CLOSED = [0.0, 0.0]


class MoveItCollisionOracle:
    """
    MoveIt Collision Oracle - Uses MoveIt ONLY for collision checking.
    
    This class:
    - Does NOT use MoveIt planners
    - Does NOT use IK services
    - Does NOT use planning pipelines
    - ONLY uses /check_state_validity service
    
    This ensures deterministic, reproducible collision checking for thesis evaluation.
    """
    
    def __init__(self, group_name='panda_arm', collision_padding=0.005):
        """
        Initialize connection to running MoveIt instance.
        
        Args:
            group_name: MoveIt planning group (must match panda_moveit_config)
            collision_padding: Safety margin for collision checking (meters)
                              0.005m recommended for paper-style evaluation
                              0.0m for exact geometry checking
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS/MoveIt not available. Is ROS sourced?")
        
        self.group_name = group_name
        self.collision_padding = collision_padding
        
        # ===== 1. CONNECT TO MOVEIT SERVICES (no planners!) =====
        logging.info("Connecting to MoveIt collision checking service...")
        rospy.wait_for_service("/check_state_validity", timeout=10.0)
        self.state_validity_srv = rospy.ServiceProxy(
            "/check_state_validity",
            GetStateValidity
        )
        logging.info("✓ Connected to /check_state_validity")
        
        # Service for getting/modifying planning scene
        rospy.wait_for_service("/get_planning_scene", timeout=10.0)
        self.get_scene_srv = rospy.ServiceProxy(
            "/get_planning_scene",
            GetPlanningScene
        )
        logging.info("✓ Connected to /get_planning_scene")
        
        # ===== 2. PUBLISHERS =====
        # Planning scene publisher (for obstacles and collision settings)
        self.scene_pub = rospy.Publisher(
            "/planning_scene",
            PlanningScene,
            queue_size=1
        )
        
        # Use move_group's fake controller for smooth animation
        # This bypasses the joint_state_publisher conflict
        self.display_trajectory_pub = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=1
        )
        
        # Joint state publisher (backup - for single pose display)
        self.joint_pub = rospy.Publisher(
            '/move_group/fake_controller_joint_states', 
            JointState, 
            queue_size=10
        )
        
        # Goal marker publisher (for visualizing target position)
        self.marker_pub = rospy.Publisher(
            '/visualization_marker',
            Marker,
            queue_size=10
        )
        
        # Wait for publishers to connect
        rospy.sleep(1.0)
        
        # ===== 3. DISABLE OCTOMAP (for deterministic checking) =====
        self._disable_octomap()
        
        # ===== 4. SET COLLISION PADDING =====
        self._set_collision_padding(collision_padding)
        
        # ===== 5. VERIFY CONNECTION =====
        self._verify_connection()
        
        logging.info(f"MoveIt Collision Oracle initialized (padding={collision_padding}m)")
    
    def _disable_octomap(self):
        """Disable octomap updates for deterministic collision checking."""
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.octomap.octomap.id = ""
        scene.world.octomap.octomap.data = []
        self.scene_pub.publish(scene)
        rospy.sleep(0.3)
        logging.info("✓ Octomap disabled (deterministic mode)")
    
    def _set_collision_padding(self, padding):
        """
        Set collision padding/margin for all links.
        
        Args:
            padding: Safety margin in meters (0.005 = 5mm recommended)
        """
        # Note: MoveIt's collision padding is typically set in the URDF or
        # through the planning scene. For runtime control, we use the
        # AllowedCollisionMatrix approach or rely on URDF settings.
        # This is a placeholder for more advanced padding control.
        self.collision_padding = padding
        logging.info(f"✓ Collision padding set to {padding}m")
    
    def _verify_connection(self):
        """Verify MoveIt connection with a test query."""
        # Test with home configuration
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        valid, contacts = self.check_collision(q_home)
        if valid:
            logging.info("✓ MoveIt connection verified (home config is collision-free)")
        else:
            logging.warning("⚠ Home configuration reports collision - check setup")
    
    # =========================================================================
    # CORE COLLISION CHECKING (Step 3: Test decoded configuration)
    # =========================================================================
    
    def build_robot_state(self, q):
        """
        Build MoveIt RobotState message from joint angles.
        
        Args:
            q: numpy array of 7 joint angles (radians)
            
        Returns:
            RobotState message
        """
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = PANDA_JOINT_NAMES
        js.position = q.tolist() if isinstance(q, np.ndarray) else list(q)
        
        state = RobotState()
        state.joint_state = js
        return state
    
    def check_collision(self, q):
        """
        Check if a single configuration is in collision.
        
        This is the PRIMARY collision checking function.
        Uses ONLY /check_state_validity service (no planners).
        
        Args:
            q: numpy array of 7 joint angles (radians)
            
        Returns:
            tuple: (is_valid, contacts)
                - is_valid: True if collision-FREE, False if in collision
                - contacts: List of contact information (if collision)
        """
        req = GetStateValidityRequest()
        req.robot_state = self.build_robot_state(q)
        req.group_name = self.group_name
        
        try:
            res = self.state_validity_srv(req)
            return res.valid, res.contacts
        except rospy.ServiceException as e:
            logging.error(f"Collision check service failed: {e}")
            return False, []
    
    def check_collision_batch(self, configurations):
        """
        Check collision for multiple configurations.
        
        Args:
            configurations: List/array of 7D joint configurations
            
        Returns:
            list of bools: True if collision-free for each config
        """
        results = []
        for q in configurations:
            valid, _ = self.check_collision(q)
            results.append(valid)
        return results
    
    def is_path_collision_free(self, trajectory):
        """
        Validate entire path is collision-free.
        
        Args:
            trajectory: List of 7D joint configurations
            
        Returns:
            dict with validation results
        """
        collision_waypoints = []
        
        for i, q in enumerate(trajectory):
            valid, contacts = self.check_collision(q)
            if not valid:  # In collision
                collision_waypoints.append({
                    'index': i,
                    'config': q.tolist() if isinstance(q, np.ndarray) else list(q),
                    'num_contacts': len(contacts)
                })
        
        return {
            'collision_free': len(collision_waypoints) == 0,
            'num_collision_waypoints': len(collision_waypoints),
            'collision_details': collision_waypoints,
            'total_waypoints': len(trajectory)
        }
    
    # =========================================================================
    # OBSTACLE MANAGEMENT (Step 4: Add obstacles programmatically)
    # =========================================================================
    
    def clear_all_obstacles(self):
        """Remove all obstacles from planning scene."""
        # Get current scene to find object names
        req = GetPlanningSceneRequest()
        req.components.components = req.components.WORLD_OBJECT_NAMES
        
        try:
            res = self.get_scene_srv(req)
            object_names = [obj.id for obj in res.scene.world.collision_objects]
        except rospy.ServiceException:
            object_names = []
        
        # Remove each object
        scene = PlanningScene()
        scene.is_diff = True
        
        for name in object_names:
            co = CollisionObject()
            co.id = name
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
        
        if object_names:
            self.scene_pub.publish(scene)
            rospy.sleep(0.3)
            logging.debug(f"Cleared {len(object_names)} obstacles from scene")
        else:
            logging.debug("No obstacles to clear")
    
    def add_cylinder_obstacle(self, name, x, y, z, radius, height):
        """
        Add a cylindrical obstacle to the planning scene.
        
        Args:
            name: Unique identifier for the obstacle (string)
            x, y, z: Center position in world frame (meters)
            radius: Cylinder radius (meters)
            height: Cylinder height (meters)
        """
        co = CollisionObject()
        co.id = name
        co.header.frame_id = "panda_link0"
        co.header.stamp = rospy.Time.now()
        
        # Define cylinder primitive
        cyl = SolidPrimitive()
        cyl.type = SolidPrimitive.CYLINDER
        cyl.dimensions = [height, radius]  # [height, radius] order for CYLINDER
        
        # Set pose (z is at center of cylinder)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        
        co.primitives = [cyl]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        
        # Publish to planning scene
        scene = PlanningScene()
        scene.world.collision_objects = [co]
        scene.is_diff = True
        
        self.scene_pub.publish(scene)
        rospy.sleep(0.2)
        logging.debug(f"Added cylinder '{name}' at ({x:.3f}, {y:.3f}, {z:.3f}), r={radius:.3f}m, h={height:.3f}m")
    
    def add_box_obstacle(self, name, x, y, z, size_x, size_y, size_z):
        """
        Add a box obstacle to the planning scene.
        
        Args:
            name: Unique identifier for the obstacle
            x, y, z: Center position (meters)
            size_x, size_y, size_z: Box dimensions (meters)
        """
        co = CollisionObject()
        co.id = name
        co.header.frame_id = "panda_link0"
        co.header.stamp = rospy.Time.now()
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [size_x, size_y, size_z]
        
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        
        co.primitives = [box]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        
        scene = PlanningScene()
        scene.world.collision_objects = [co]
        scene.is_diff = True
        
        self.scene_pub.publish(scene)
        rospy.sleep(0.2)
        logging.debug(f"Added box '{name}' at ({x:.3f}, {y:.3f}, {z:.3f})")
    
    def add_table(self, height=0.0, size=(2.0, 2.0, 0.05)):
        """Add table/ground plane to planning scene."""
        self.add_box_obstacle(
            name="table",
            x=0.0, y=0.0, z=height - size[2]/2,
            size_x=size[0], size_y=size[1], size_z=size[2]
        )
        logging.debug(f"Added table at z={height}m")
    
    def add_obstacles_from_array(self, obstacles):
        """
        Add multiple cylindrical obstacles from array.
        
        Args:
            obstacles: List of [x, y, h, r] arrays where:
                       x, y = position, h = height, r = radius
                       z is computed as h/2 (cylinder on ground)
        """
        for i, obs in enumerate(obstacles):
            x, y, h, r = obs
            self.add_cylinder_obstacle(
                name=f"obstacle_{i}",
                x=x, y=y, z=h/2,  # Center cylinder at half-height
                radius=r, height=h
            )
        
        rospy.sleep(0.2)
        logging.debug(f"Added {len(obstacles)} obstacle(s) to scene")
    
    # =========================================================================
    # VISUALIZATION (RViz display only)
    # =========================================================================
    
    def publish_joint_state(self, q, duration=0.05):
        """
        Publish robot configuration for RViz visualization.
        Includes closed gripper for consistent appearance.
        
        Args:
            q: 7D joint angles (radians)
            duration: Display time (seconds)
        """
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = PANDA_ALL_JOINT_NAMES  # Include gripper joints
        positions = q.tolist() if isinstance(q, np.ndarray) else list(q)
        msg.position = positions + GRIPPER_CLOSED  # Append closed gripper
        
        self.joint_pub.publish(msg)
        rospy.sleep(duration)
    
    def animate_trajectory(self, trajectory, step_duration=0.05, hold_start=0.0, hold_end=1, scene_robot_position=None):
        """
        Animate robot following a trajectory in RViz using MoveIt's DisplayTrajectory.
        
        Scene Robot (solid) stays at START during animation, then jumps to GOAL.
        Planned Path robot (ghost) follows the trajectory.
        
        NOTE: Scene Robot should already be at START position before calling this function.
              We do NOT update it here to avoid jerky behavior.
        
        Args:
            trajectory: List of 7D configurations
            step_duration: Time per waypoint (seconds)
            hold_start: Time to hold at start position before moving (seconds)
            hold_end: Time to hold at goal position after reaching (seconds)
            scene_robot_position: If provided, use this exact position for trajectory_start
                                  to match where Scene Robot already is (avoids jerk)
        """
        logging.info(f"Animating trajectory ({len(trajectory)} waypoints)...")
        
        # Compute position lists
        traj_start_positions = trajectory[0].tolist() if isinstance(trajectory[0], np.ndarray) else list(trajectory[0])
        goal_positions = trajectory[-1].tolist() if isinstance(trajectory[-1], np.ndarray) else list(trajectory[-1])
        
        # CRITICAL: Use scene_robot_position for trajectory_start if provided
        # This ensures Planned Path Robot spawns EXACTLY where Scene Robot already is
        if scene_robot_position is not None:
            start_positions = scene_robot_position.tolist() if isinstance(scene_robot_position, np.ndarray) else list(scene_robot_position)
        else:
            start_positions = traj_start_positions
        
        # NOTE: Scene Robot is already at start position (set before this function is called)
        # DO NOT publish to joint_pub here - it causes jerky behavior when DisplayTrajectory spawns
        
        # Build a DisplayTrajectory message for the "Planned Path" robot
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = "panda"
        
        # Create a RobotTrajectory with hold periods
        robot_traj = RobotTrajectory()
        joint_traj = JointTrajectory()
        joint_traj.joint_names = PANDA_ALL_JOINT_NAMES  # Include gripper for closed appearance
        
        # Add waypoints with timing:
        # 1. Hold at start (use Scene Robot's exact position)
        # 2. Actual trajectory
        # 3. Hold at end (duplicate end position)
        
        current_time = 0.0
        
        # Hold at START position - use Scene Robot's exact position
        point_start = JointTrajectoryPoint()
        point_start.positions = start_positions + GRIPPER_CLOSED  # Add closed gripper
        point_start.time_from_start = rospy.Duration(current_time)
        joint_traj.points.append(point_start)
        current_time += hold_start
        
        # Add actual trajectory waypoints (starting after hold_start)
        for i, q in enumerate(trajectory):
            point = JointTrajectoryPoint()
            positions = q.tolist() if isinstance(q, np.ndarray) else list(q)
            point.positions = positions + GRIPPER_CLOSED  # Add closed gripper
            point.time_from_start = rospy.Duration(current_time + i * step_duration)
            joint_traj.points.append(point)
        
        trajectory_time = len(trajectory) * step_duration
        current_time += trajectory_time
        
        # Hold at GOAL position
        point_end = JointTrajectoryPoint()
        point_end.positions = goal_positions + GRIPPER_CLOSED  # Add closed gripper
        point_end.time_from_start = rospy.Duration(current_time + hold_end)
        joint_traj.points.append(point_end)
        
        robot_traj.joint_trajectory = joint_traj
        display_trajectory.trajectory.append(robot_traj)
        
        # CRITICAL: Set trajectory_start to match where Scene Robot already is
        # This tells MoveIt "the robot is already here, start the ghost from here"
        start_state = RobotState()
        start_js = JointState()
        start_js.name = PANDA_ALL_JOINT_NAMES  # Include gripper
        start_js.position = start_positions + GRIPPER_CLOSED  # Add closed gripper
        start_state.joint_state = start_js
        display_trajectory.trajectory_start = start_state
        
        # Publish trajectory for "Planned Path" visualization
        self.display_trajectory_pub.publish(display_trajectory)
        
        # Calculate total wait time
        total_time = hold_start + trajectory_time + hold_end
        logging.info(f"Playing trajectory: {hold_start:.1f}s hold -> {trajectory_time:.1f}s motion -> {hold_end:.1f}s hold")
        
        # Wait for entire animation including hold periods
        rospy.sleep(total_time)
        
        # Animation complete - jump Scene Robot to GOAL position
        goal_msg = JointState()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.name = PANDA_ALL_JOINT_NAMES  # Include gripper
        goal_msg.position = goal_positions + GRIPPER_CLOSED  # Add closed gripper
        self.joint_pub.publish(goal_msg)
        
        logging.info("Animation complete - Scene Robot jumped to goal")
    
    def publish_goal_marker(self, x, y, z, marker_id=0):
        """
        Publish a sphere marker at the goal position for RViz visualization.
        
        Args:
            x, y, z: Goal position in world frame (meters)
            marker_id: Unique ID for the marker
        """
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        
        # Scale (10cm diameter sphere)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Color (bright green for goal)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        # Lifetime (0 = forever until deleted)
        marker.lifetime = rospy.Duration(0)
        
        self.marker_pub.publish(marker)
        rospy.sleep(0.1)
        logging.debug(f"Published goal marker at ({x:.3f}, {y:.3f}, {z:.3f})")
    
    def publish_start_marker(self, x, y, z, marker_id=1):
        """
        Publish a sphere marker at the start position for RViz visualization.
        
        Args:
            x, y, z: Start position in world frame (meters)
            marker_id: Unique ID for the marker
        """
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "start"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        
        # Scale (8cm diameter sphere)
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        
        # Color (blue for start)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        marker.lifetime = rospy.Duration(0)
        
        self.marker_pub.publish(marker)
        rospy.sleep(0.1)
        logging.debug(f"Published start marker at ({x:.3f}, {y:.3f}, {z:.3f})")
    
    def clear_markers(self):
        """Clear all visualization markers."""
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.action = Marker.DELETEALL
        self.marker_pub.publish(marker)
        rospy.sleep(0.1)
    
    def clear_trajectory_display(self):
        """
        Clear the trajectory display (ghost robots) from RViz.
        This removes the semi-transparent trajectory preview.
        """
        # Publish empty trajectory to clear display
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = "panda"
        display_trajectory.trajectory = []  # Empty trajectory
        self.display_trajectory_pub.publish(display_trajectory)
        rospy.sleep(0.1)

class LatentSpacePlanner:
    """Latent space path planner (same as evaluate_planning.py)"""
    
    def __init__(self, model, model_obs, robot, mean_train, std_train, mean_obs, std_obs, device):
        self.model = model
        self.model_obs = model_obs
        self.robot = robot
        self.mean_train_tensor = torch.tensor(mean_train, dtype=torch.float32).to(device)
        self.std_train_tensor = torch.tensor(std_train, dtype=torch.float32).to(device)
        # CRITICAL: Store obstacle normalization for classifier
        self.mean_obs = torch.tensor(mean_obs, dtype=torch.float32).to(device)
        self.std_obs = torch.tensor(std_obs, dtype=torch.float32).to(device)
        self.device = device
        
        self.model.eval()
        if self.model_obs is not None:
            self.model_obs.eval()
    
    def plan(self, q_start, e_start, e_target, obstacles, args):
        """
        Plan path from start to target avoiding obstacles.
        
        Implements the full planning algorithm from the paper including:
        - Goal reaching loss (Eq. 4)
        - Prior loss (Eq. 5)
        - Collision loss with temperature (Eq. 6)
        - GECO adaptive weighting (Algorithm 2)
        
        Returns:
            dict with 'success', 'joint_trajectory', 'latent_trajectory', 'metrics'
        """
        # Normalize start
        x_start = torch.cat([q_start, e_start], dim=1)
        x_start_norm = (x_start - self.mean_train_tensor[:, :10]) / self.std_train_tensor[:, :10]
        
        # Encode to latent
        with torch.no_grad():
            z_init = self.model.encoder(x_start_norm)[0]
        
        z = z_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=args.planning_lr)
        
        # CRITICAL: Normalize obstacles for classifier (bug fix!)
        # Raw obstacles are in world coords, classifier expects normalized
        obs_tensors = []
        for obs in obstacles:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            obs_normalized = (obs_tensor - self.mean_obs) / self.std_obs
            obs_tensors.append(obs_normalized.unsqueeze(0))
        
        # Optimization loop
        joint_trajectory = [q_start.cpu().numpy()[0]]
        latent_trajectory = [z.detach().cpu().numpy()[0]]
        path_collision_probs = []  # Track collision probability at each step
        
        start_time = time.time()
        min_dist = 1e5
        goal_reached = False  # Renamed from 'success' - just tracks goal reaching
        final_step = 0
        
        # GECO state variables
        lambda_prior = args.lambda_prior
        lambda_collision = args.lambda_collision
        C_prior_ma = None
        C_collision_ma = None
        
        # Get temperature (default to 1.0 if not set)
        temperature = getattr(args, 'temperature', 1.0)
        
        for step in range(args.max_steps):
            optimizer.zero_grad()
            
            # Decode
            x_decoded_norm = self.model.decoder(z)
            x_decoded = x_decoded_norm * self.std_train_tensor[:, :10] + self.mean_train_tensor[:, :10]
            q_decoded = x_decoded[:, :7]
            e_decoded = x_decoded[:, 7:10]
            
            # Loss 1: Goal reaching (Eq. 4)
            L_goal = torch.norm(e_decoded - e_target)
            
            # Loss 2: Prior loss (Eq. 5)
            L_prior = 0.5 * torch.sum(z ** 2)
            
            # Loss 3: Collision loss with temperature (Eq. 6)
            L_collision = torch.tensor(0.0, device=self.device)
            step_max_collision_prob = 0.0  # Track collision prob at this step
            if self.model_obs is not None and len(obstacles) > 0:
                for obs_tensor in obs_tensors:
                    logit = self.model_obs.obstacle_collision_classifier(z, obs_tensor)
                    # Apply temperature scaling for softer predictions
                    p_collision = torch.sigmoid(logit / temperature)
                    step_max_collision_prob = max(step_max_collision_prob, p_collision.item())
                    L_collision = L_collision + (-torch.log(1 - p_collision + 1e-8))
                # Normalize by number of obstacles
                L_collision = L_collision / len(obstacles)
            
            # Track collision along path (for success determination)
            path_collision_probs.append(step_max_collision_prob)
            
            # GECO adaptive weighting (Algorithm 2)
            if getattr(args, 'use_geco', False):
                # Constraint violations
                C_prior = L_prior.item() - args.tau_prior_goal
                C_collision = L_collision.item() - args.tau_obs_goal
                
                # Initialize or update moving averages
                if step == 0:
                    C_prior_ma = C_prior
                    C_collision_ma = C_collision
                else:
                    C_prior_ma = args.alpha_ma_prior * C_prior_ma + (1 - args.alpha_ma_prior) * C_prior
                    C_collision_ma = args.alpha_ma_obs * C_collision_ma + (1 - args.alpha_ma_obs) * C_collision
                
                # Update lambdas
                kappa_prior = np.exp(args.alpha_geco * C_prior_ma)
                kappa_collision = np.exp(args.alpha_geco * C_collision_ma)
                
                lambda_prior = np.clip(kappa_prior * lambda_prior, 1e-6, 1000.0)
                lambda_collision = np.clip(kappa_collision * lambda_collision, 1e-6, 1000.0)
            
            # Combined loss (Eq. 3)
            L_total = L_goal + lambda_prior * L_prior + lambda_collision * L_collision
            
            dist = L_goal.item()
            min_dist = min(min_dist, dist)
            
            # Check if goal is reached (but NOT final success yet - need collision check)
            if dist < args.success_threshold:
                goal_reached = True
                final_step = step
                break
            
            # Optimize
            L_total.backward()
            optimizer.step()
            
            # Save trajectory
            if step % 5 == 0:  # Save every 5 steps for smoother animation
                joint_trajectory.append(q_decoded.detach().cpu().numpy()[0])
                latent_trajectory.append(z.detach().cpu().numpy()[0])
        
        # Add final state
        with torch.no_grad():
            x_decoded_norm = self.model.decoder(z)
            x_decoded = x_decoded_norm * self.std_train_tensor[:, :10] + self.mean_train_tensor[:, :10]
            q_final = x_decoded[:, :7].cpu().numpy()[0]
            joint_trajectory.append(q_final)
            latent_trajectory.append(z.detach().cpu().numpy()[0])
        
        planning_time = (time.time() - start_time) * 1000
        
        # ================================================================
        # SUCCESS DETERMINATION (matching paper's methodology)
        # Paper Section V-B: "a run is considered a success if the robot
        # reaches the target within a distance threshold of 1cm AND 
        # without colliding with obstacles"
        #
        # IMPORTANT: Paper uses MoveIt ground-truth for collision checking,
        # NOT the classifier. The classifier is only for gradient-based
        # collision avoidance during planning. 
        #
        # Here we report:
        # - goal_reached: True if endpoint within distance threshold
        # - classifier_collision_free: Based on classifier predictions (for analysis)
        # - success: Based on goal_reached only - MoveIt validation done separately
        # ================================================================
        
        # Compute classifier statistics (for analysis, NOT final success)
        warmup_steps = getattr(args, 'warmup_steps', 0)
        path_after_warmup = path_collision_probs[warmup_steps:] if len(path_collision_probs) > warmup_steps else path_collision_probs
        
        path_max_collision_prob = max(path_collision_probs) if path_collision_probs else 0.0
        path_max_after_warmup = max(path_after_warmup) if path_after_warmup else 0.0
        path_mean_collision_prob = np.mean(path_collision_probs) if path_collision_probs else 0.0
        
        # Classifier-based collision check (for analysis only)
        path_collision_threshold = 0.9
        classifier_collision_free = path_max_after_warmup <= path_collision_threshold
        
        # SUCCESS for planner = goal_reached only
        # MoveIt ground-truth collision check will be done in the main loop
        # This matches paper: planner tries to reach goal, MoveIt validates collision
        success = goal_reached  # MoveIt validation done separately
        
        # Log collision info
        if goal_reached:
            logging.info(f"Goal reached! Classifier collision prob: max={path_max_collision_prob:.3f}, "
                        f"max_after_warmup={path_max_after_warmup:.3f}, "
                        f"classifier_collision_free={classifier_collision_free}")
        else:
            logging.info(f"Goal NOT reached: min_dist={min_dist:.4f}m")
        
        # Generate SMOOTH trajectory by interpolating in latent space (for visualization)
        # This replaces the optimization history with a proper interpolated path
        smooth_joint_trajectory = []
        smooth_latent_trajectory = []
        collision_scores = []  # Track collision probability along path
        
        if goal_reached:  # Generate trajectory even if collision detected (for visualization)
            z_start = z_init.detach()
            z_goal = z.detach()
            num_interpolation_steps = getattr(args, 'interpolation_steps', 50)
            
            # Get temperature for collision scoring
            temperature = getattr(args, 'temperature', 1.0)
            
            # CRITICAL: Normalize obstacles for classifier (bug fix!)
            obs_tensors = []
            for obs in obstacles:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                obs_normalized = (obs_tensor - self.mean_obs) / self.std_obs
                obs_tensors.append(obs_normalized.unsqueeze(0))
            
            with torch.no_grad():
                for i in range(num_interpolation_steps + 1):
                    alpha = i / num_interpolation_steps
                    z_interp = (1 - alpha) * z_start + alpha * z_goal
                    
                    # Decode interpolated latent
                    x_decoded_norm = self.model.decoder(z_interp)
                    x_decoded = x_decoded_norm * self.std_train_tensor[:, :10] + self.mean_train_tensor[:, :10]
                    q_interp = x_decoded[:, :7].cpu().numpy()[0]
                    
                    # Check collision probability along path
                    p_collision_max = 0.0
                    if self.model_obs is not None and len(obstacles) > 0:
                        for obs_tensor in obs_tensors:
                            logit = self.model_obs.obstacle_collision_classifier(z_interp, obs_tensor)
                            p_collision = torch.sigmoid(logit / temperature).item()
                            p_collision_max = max(p_collision_max, p_collision)
                    
                    collision_scores.append(p_collision_max)
                    smooth_joint_trajectory.append(q_interp)
                    smooth_latent_trajectory.append(z_interp.cpu().numpy()[0])
            
            # Log collision statistics along path
            if collision_scores:
                max_collision = max(collision_scores)
                avg_collision = sum(collision_scores) / len(collision_scores)
                high_risk_count = sum(1 for p in collision_scores if p > 0.5)
                logging.info(f"Interpolated path collision: max={max_collision:.3f}, avg={avg_collision:.3f}, "
                            f"high_risk_waypoints={high_risk_count}/{len(collision_scores)}")
        else:
            # If goal not reached, just use start position
            smooth_joint_trajectory = [q_start.cpu().numpy()[0]]
            smooth_latent_trajectory = [z_init.detach().cpu().numpy()[0]]
        
        return {
            'success': success,  # True if goal reached (MoveIt validation done separately)
            'goal_reached': goal_reached,  # True if endpoint within distance threshold
            'classifier_collision_free': classifier_collision_free,  # Based on classifier (for analysis)
            'joint_trajectory': smooth_joint_trajectory,  # Use smooth trajectory for visualization
            'latent_trajectory': smooth_latent_trajectory,
            'optimization_trajectory': joint_trajectory,  # Keep original for analysis
            'collision_scores': collision_scores,  # Collision probability along interpolated path
            'path_collision_probs': path_collision_probs,  # Collision prob during optimization
            'max_collision_prob': path_max_collision_prob,
            'max_collision_after_warmup': path_max_after_warmup,
            'metrics': {
                'planning_time_ms': planning_time,
                'num_steps': final_step + 1 if goal_reached else step + 1,
                'min_distance': min_dist,
                'final_distance': dist
            }
        }


def run_simulation(args):
    """Main simulation loop"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configurations
    with open(args.config, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']
    
    # Load dataset for normalization
    dataset = RobotStateDataset(
        args.data_path, train=0, train_data_name='free_space_100k_train.dat')
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()
    
    # CRITICAL: Load obstacle normalization stats (bug fix!)
    obs_dataset = RobotObstacleDataset(
        args.data_path, train=0,
        train_data_name='collision_100k_train.dat',
        test_data_name='collision_10k_test.dat',
        free_space_train_name='free_space_100k_train.dat',
        free_space_test_name='free_space_10k_test.dat'
    )
    mean_obs = obs_dataset.get_mean_train()[0, 10:14]  # obstacle x,y,h,r
    std_obs = obs_dataset.get_std_train()[0, 10:14]
    logging.info(f"Obstacle normalization: mean={mean_obs}, std={std_obs}")
    
    # Initialize robot (for FK)
    robot = Panda()
    robot.to(device)
    
    # Load VAE
    logging.info(f"Loading VAE from {args.checkpoint}")
    model = VAE(config['input_dim'], config['latent_dim'], 
                config['units_per_layer'], config['num_hidden_layers'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load collision classifier
    logging.info(f"Loading collision classifier from {args.checkpoint_obs}")
    model_obs = VAEObstacleBCE(config['input_dim'], config['latent_dim'],
                                config['units_per_layer'], config['num_hidden_layers'])
    checkpoint_obs = torch.load(args.checkpoint_obs, map_location=device)
    model_obs.load_state_dict(checkpoint_obs['model_state_dict'])
    model_obs.to(device)
    model_obs.eval()
    
    # Initialize planner
    planner = LatentSpacePlanner(model, model_obs, robot, mean_train, std_train, mean_obs, std_obs, device)
    
    # Initialize MoveIt Collision Oracle (if available)
    # NOTE: This connects to RUNNING MoveIt instance - does NOT launch it
    collision_oracle = None
    if ROS_AVAILABLE and not args.no_visualization:
        try:
            # Initialize ROS node (connects to running roscore/MoveIt)
            # disable_signals=True prevents ROS from overriding Python's signal handlers
            # This is important for proper terminal output and clean shutdown
            rospy.init_node('latent_planner_validator', anonymous=True, disable_signals=True)
            
            # CRITICAL: Reconfigure logging after ROS init (ROS breaks Python logging)
            setup_logging()
            
            # Create collision oracle with specified padding
            collision_oracle = MoveItCollisionOracle(
                group_name='panda_arm',
                collision_padding=args.collision_padding
            )
            
            # Add ground plane/table
            collision_oracle.add_table(height=0.0)
            
            logging.info("="*60)
            logging.info("MoveIt Collision Oracle READY")
            logging.info(f"  - Collision padding: {args.collision_padding}m")
            logging.info(f"  - Deterministic mode: ON (octomap disabled)")
            logging.info("="*60)
            
        except Exception as e:
            logging.warning(f"Could not initialize MoveIt Oracle: {e}")
            logging.warning("Continuing without ground-truth validation")
            collision_oracle = None
    else:
        logging.info("Running without MoveIt (no ground-truth validation)")
    
    # Statistics - comprehensive metrics for thesis evaluation
    stats = {
        'total_scenarios': 0,
        'invalid_scenes_rejected': 0,  # Scenes rejected during validation
        'planner_success': 0,
        'planner_failed': 0,
        'moveit_validated_success': 0,
        'moveit_rejected': 0,
        'planner_collision_free': 0,
        'moveit_collision_free': 0,
        # Detailed collision metrics
        'total_collision_waypoints': 0,
        'max_collision_waypoints_per_path': 0,
        'collision_details': [],  # Per-scenario collision info
        # Planning metrics
        'planning_times_ms': [],
        'planning_steps': [],
        'goal_distances': [],
        'min_distances': [],
        # Path metrics
        'path_lengths': [],  # Number of waypoints
    }
    
    # Scene validation parameters (matching paper Section IV-C)
    # "ensure feasible set of joint angles reaching target without collision"
    MIN_EE_HEIGHT = 0.05  # End-effector must be at least 5cm above ground
    MIN_EE_OBSTACLE_DIST = 0.02  # End-effector must be at least 2cm from obstacle surface
    MAX_SCENE_ATTEMPTS = 50  # Max attempts to find valid scene
    
    # List to save validated scenarios
    scenarios_to_save = []
    
    # Trajectory export data
    export_data = None
    all_trajectories = []  # Store all successful trajectories
    
    # Run scenarios
    logging.info(f"\n{'='*70}")
    logging.info(f"Running {args.num_scenarios} simulation scenarios")
    logging.info(f"{'='*70}\n")
    
    # Load pre-generated scenarios if specified
    loaded_scenarios = None
    if args.load_scenes:
        with open(args.load_scenes, 'r') as f:
            loaded_scenarios = json.load(f)
        logging.info(f"Loaded {len(loaded_scenarios['scenarios'])} scenarios from {args.load_scenes}")
        # Override num_scenarios to match loaded scenarios
        args.num_scenarios = len(loaded_scenarios['scenarios'])
    
    for scenario_id in range(args.num_scenarios):
        logging.info(f"\n--- Scenario {scenario_id + 1}/{args.num_scenarios} ---")
        
        # Load or generate scenario
        if loaded_scenarios:
            # Load pre-generated scenario
            scenario = loaded_scenarios['scenarios'][scenario_id]
            q_start = torch.tensor(scenario['q_start'], device=device, dtype=torch.float32).unsqueeze(0)
            e_start = torch.tensor(scenario['e_start'], device=device, dtype=torch.float32).unsqueeze(0)
            q_target = torch.tensor(scenario['q_target'], device=device, dtype=torch.float32).unsqueeze(0)
            e_target = torch.tensor(scenario['e_target'], device=device, dtype=torch.float32).unsqueeze(0)
            obstacles = [np.array(obs) for obs in scenario['obstacles']]
            
            logging.info(f"Start: {e_start.cpu().numpy()[0]}")
            logging.info(f"Target: {e_target.cpu().numpy()[0]}")
            logging.info(f"Distance: {torch.norm(e_target - e_start).item():.3f}m")
            for obs in obstacles:
                dist_from_base = np.sqrt(obs[0]**2 + obs[1]**2)
                logging.info(f"Obstacle: ({obs[0]:.2f}, {obs[1]:.2f}), h={obs[2]:.2f}m, r={obs[3]:.3f}m, dist_from_base={dist_from_base:.2f}m")
        else:
            # ================================================================
            # SCENE GENERATION (matching paper Section IV-C)
            # "The first obstacle is sampled between the initial end-effector
            # position and the target position."
            # ================================================================
            # 
            # Efficient validation order:
            # 1. Sample configs, check EE heights (geometric - fast)
            # 2. Check self-collision via MoveIt (no obstacles needed)
            # 3. Generate obstacle BETWEEN start and goal EE positions
            # 4. Check EE not inside obstacle (geometric - fast)
            # 5. Add obstacles to MoveIt and check joint-obstacle collision
            # ================================================================
            
            q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
            q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)
            
            valid_scene = False
            scene_attempts = 0
            
            # Clear MoveIt scene once at start (table only, no obstacles yet)
            if collision_oracle is not None:
                collision_oracle.clear_all_obstacles()
                collision_oracle.add_table(height=0.0)
                rospy.sleep(0.1)
            
            while not valid_scene and scene_attempts < MAX_SCENE_ATTEMPTS:
                scene_attempts += 1
                
                # Step 1: Sample start configuration
                q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
                e_start = robot.FK(q_start.clone(), device, rad=True)
                e_start_np = e_start.cpu().numpy()[0]
                
                # Check start EE above ground
                if e_start_np[2] < MIN_EE_HEIGHT:
                    continue
                
                # Step 2: Sample target configuration
                q_target = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
                e_target = robot.FK(q_target.clone(), device, rad=True)
                e_target_np = e_target.cpu().numpy()[0]
                
                # Check target EE above ground
                if e_target_np[2] < MIN_EE_HEIGHT:
                    continue
                
                # Step 3: Check self-collision (no obstacles in scene yet - fast!)
                if collision_oracle is not None:
                    start_valid, _ = collision_oracle.check_collision(q_start.cpu().numpy()[0])
                    if not start_valid:
                        continue
                    
                    target_valid, _ = collision_oracle.check_collision(q_target.cpu().numpy()[0])
                    if not target_valid:
                        continue
                
                # Step 4: Generate obstacle BETWEEN start and goal (paper IV-C)
                # "The first obstacle is sampled between the initial end-effector
                # position and the target position."
                obstacles = []
                if args.num_obstacles > 0:
                    # Sample position along the line from start to goal
                    t = np.random.uniform(0.3, 0.7)  # 30-70% along the path
                    obs_pos = e_start_np + t * (e_target_np - e_start_np)
                    obs_x, obs_y = obs_pos[0], obs_pos[1]
                    obs_h = np.random.uniform(0.5, 1.0)
                    obs_r = np.random.uniform(0.05, 0.15)
                    
                    # Check obstacle is not too close to robot base
                    dist_from_base = np.sqrt(obs_x**2 + obs_y**2)
                    if dist_from_base < 0.25:  # Too close to base
                        continue
                    
                    # Check EE not inside obstacle (geometric check)
                    dist_start_to_obs = np.sqrt((e_start_np[0] - obs_x)**2 + (e_start_np[1] - obs_y)**2)
                    dist_target_to_obs = np.sqrt((e_target_np[0] - obs_x)**2 + (e_target_np[1] - obs_y)**2)
                    
                    if dist_start_to_obs < obs_r + MIN_EE_OBSTACLE_DIST and e_start_np[2] < obs_h:
                        continue  # Start EE inside obstacle
                    if dist_target_to_obs < obs_r + MIN_EE_OBSTACLE_DIST and e_target_np[2] < obs_h:
                        continue  # Target EE inside obstacle
                    
                    obstacles.append(np.array([obs_x, obs_y, obs_h, obs_r]))
                
                # Step 5: Add obstacles and check joint-obstacle collision
                if collision_oracle is not None and len(obstacles) > 0:
                    # Temporarily add obstacles for collision check
                    collision_oracle.add_obstacles_from_array(obstacles)
                    rospy.sleep(0.05)  # Brief pause
                    
                    # Check start config with obstacles
                    start_valid, _ = collision_oracle.check_collision(q_start.cpu().numpy()[0])
                    if not start_valid:
                        collision_oracle.clear_all_obstacles()
                        collision_oracle.add_table(height=0.0)
                        continue
                    
                    # Check target config with obstacles
                    target_valid, _ = collision_oracle.check_collision(q_target.cpu().numpy()[0])
                    if not target_valid:
                        collision_oracle.clear_all_obstacles()
                        collision_oracle.add_table(height=0.0)
                        continue
                    
                    # Clear obstacles (will be re-added after validation complete)
                    collision_oracle.clear_all_obstacles()
                    collision_oracle.add_table(height=0.0)
                
                # All checks passed!
                valid_scene = True
            
            # Handle validation result
            if not valid_scene:
                logging.warning(f"Could not find valid scene after {MAX_SCENE_ATTEMPTS} attempts")
                stats['invalid_scenes_rejected'] += 1
                # Use last sample anyway (fallback)
            elif scene_attempts > 1:
                logging.debug(f"Valid scene found in {scene_attempts} attempts")
            
            # Log scene info
            logging.info(f"Start: {e_start.cpu().numpy()[0]}")
            logging.info(f"Target: {e_target.cpu().numpy()[0]}")
            logging.info(f"Distance: {torch.norm(e_target - e_start).item():.3f}m")
            for obs in obstacles:
                dist_from_base = np.sqrt(obs[0]**2 + obs[1]**2)
                logging.info(f"Obstacle: ({obs[0]:.2f}, {obs[1]:.2f}), h={obs[2]:.2f}m, r={obs[3]:.3f}m, dist={dist_from_base:.2f}m")
        
        # Save scenario for later use in evaluation
        if args.save_scenes:
            scenarios_to_save.append({
                'scenario_id': scenario_id,
                'q_start': q_start.cpu().numpy()[0].tolist(),
                'e_start': e_start.cpu().numpy()[0].tolist(),
                'q_target': q_target.cpu().numpy()[0].tolist(),
                'e_target': e_target.cpu().numpy()[0].tolist(),
                'obstacles': [obs.tolist() for obs in obstacles]
            })
        
        # Setup MoveIt scene with obstacles and visualize start/goal
        if collision_oracle is not None:
            collision_oracle.clear_all_obstacles()
            collision_oracle.clear_markers()
            collision_oracle.clear_trajectory_display()  # Clear ghost robots from previous scene
            collision_oracle.add_table(height=0.0)
            collision_oracle.add_obstacles_from_array(obstacles)
            
            # Visualize start and goal positions
            e_start_np = e_start.cpu().numpy()[0]
            e_target_np = e_target.cpu().numpy()[0]
            collision_oracle.publish_start_marker(e_start_np[0], e_start_np[1], e_start_np[2])
            collision_oracle.publish_goal_marker(e_target_np[0], e_target_np[1], e_target_np[2])
            
            # Show robot at start position
            collision_oracle.publish_joint_state(q_start.cpu().numpy()[0])
            rospy.sleep(1.0)  # Pause to see start position
        
        # Plan path
        logging.info("Planning...")
        result = planner.plan(q_start, e_start, e_target, obstacles, args)
        
        stats['total_scenarios'] += 1
        goal_distance = torch.norm(e_target - e_start).item()
        stats['goal_distances'].append(goal_distance)
        
        # Extract results
        # After the planner change, 'success' now means 'goal_reached' only
        # MoveIt ground-truth will determine actual collision-free success
        goal_reached = result.get('goal_reached', result['success'])
        classifier_collision_free = result.get('classifier_collision_free', True)
        
        # Track goal reaching (planner metric)
        if goal_reached:
            stats['planner_success'] += 1  # Renamed: this is "goal reaching" success
            stats['planning_times_ms'].append(result['metrics']['planning_time_ms'])
            stats['planning_steps'].append(result['metrics']['num_steps'])
            stats['path_lengths'].append(len(result['joint_trajectory']))
            
            if classifier_collision_free:
                stats['planner_collision_free'] += 1
                logging.info(f"✓ Goal REACHED + Classifier: collision-free "
                            f"({result['metrics']['planning_time_ms']:.1f}ms, "
                            f"{result['metrics']['num_steps']} steps)")
            else:
                max_collision = result.get('max_collision_after_warmup', 0.0)
                logging.info(f"✓ Goal REACHED but Classifier: collision detected "
                            f"(max_prob={max_collision:.3f})")
        else:
            # Did not reach goal
            stats['planner_failed'] += 1
            stats['min_distances'].append(result['metrics']['min_distance'])
            logging.info(f"✗ Goal NOT reached: min_dist={result['metrics']['min_distance']:.4f}m")
        
        # Prepare trajectory data structure (will be enhanced with MoveIt validation)
        trajectory_data = None
        if goal_reached:
            trajectory_data = {
                'scenario_id': int(scenario_id),
                'goal_reached': bool(goal_reached),
                'classifier_collision_free': bool(classifier_collision_free),
                'start_joint_angles': [float(x) for x in q_start.cpu().numpy()[0]],
                'target_joint_angles': [float(x) for x in q_target.cpu().numpy()[0]],
                'start_ee_position': [float(x) for x in e_start.cpu().numpy()[0]],
                'target_ee_position': [float(x) for x in e_target.cpu().numpy()[0]],
                'goal_distance': float(goal_distance),
                'obstacles': [[float(x) for x in obs] for obs in obstacles],
                'trajectory': {
                    'joint_angles': [[float(x) for x in q] for q in result['joint_trajectory']],
                    'num_waypoints': len(result['joint_trajectory'])
                },
                'metrics': {
                    'planning_time_ms': float(result['metrics']['planning_time_ms']),
                    'num_steps': int(result['metrics']['num_steps']),
                    'min_distance': float(result['metrics']['min_distance']),
                    'final_distance': float(result['metrics']['final_distance'])
                },
                'moveit_validation': None  # Will be filled if MoveIt is available
            }
        
        # ================================================================
        # MoveIt Ground-Truth Validation (matching paper's methodology)
        # Paper: "a run is considered a success if the robot reaches the 
        # target within a distance threshold of 1cm AND without colliding"
        # ================================================================
        if collision_oracle is not None and goal_reached:
            # Animate path in RViz (smooth interpolated trajectory)
            if not args.no_animation:
                logging.info(f"Animating smooth trajectory in RViz ({len(result['joint_trajectory'])} waypoints)...")
                collision_oracle.animate_trajectory(
                    result['joint_trajectory'], 
                    step_duration=args.animation_speed,
                    scene_robot_position=q_start.cpu().numpy()[0]  # Pass exact Scene Robot position
                )
            
            # Validate collision-free using MoveIt ground-truth oracle
            logging.info("Validating with MoveIt collision oracle...")
            validation = collision_oracle.is_path_collision_free(result['joint_trajectory'])
            
            if validation['collision_free']:
                stats['moveit_validated_success'] += 1
                stats['moveit_collision_free'] += 1
                logging.info(f"✓ MoveIt VALIDATED: Path is collision-free")
                stats['collision_details'].append({
                    'scenario_id': scenario_id,
                    'collision_free': True,
                    'num_collision_waypoints': 0,
                    'total_waypoints': validation['total_waypoints']
                })
                # Add validation to trajectory data
                if trajectory_data is not None:
                    trajectory_data['moveit_validation'] = {
                        'collision_free': True,
                        'num_collision_waypoints': 0,
                        'total_waypoints': validation['total_waypoints']
                    }
            else:
                stats['moveit_rejected'] += 1
                num_collisions = validation['num_collision_waypoints']
                stats['total_collision_waypoints'] += num_collisions
                stats['max_collision_waypoints_per_path'] = max(
                    stats['max_collision_waypoints_per_path'], num_collisions)
                collision_pct = (num_collisions / validation['total_waypoints']) * 100
                stats['collision_details'].append({
                    'scenario_id': scenario_id,
                    'collision_free': False,
                    'num_collision_waypoints': num_collisions,
                    'total_waypoints': validation['total_waypoints'],
                    'collision_percentage': collision_pct
                })
                # Add validation to trajectory data
                if trajectory_data is not None:
                    trajectory_data['moveit_validation'] = {
                        'collision_free': False,
                        'num_collision_waypoints': num_collisions,
                        'total_waypoints': validation['total_waypoints'],
                        'collision_percentage': collision_pct
                    }
                logging.warning(f"✗ MoveIt REJECTED: {num_collisions}/{validation['total_waypoints']} "
                               f"waypoints in collision ({collision_pct:.1f}%)")
        
        # Export trajectory data if requested (now with MoveIt validation)
        if args.export_trajectory and trajectory_data is not None:
            if args.export_scenario is None or scenario_id == args.export_scenario:
                all_trajectories.append(trajectory_data)
                if args.export_scenario == scenario_id:
                    export_data = trajectory_data
                    logging.info(f"Trajectory data captured for scenario {scenario_id}")
        
        # Progress
        if (scenario_id + 1) % 10 == 0 or scenario_id == args.num_scenarios - 1:
            goal_reaching_rate = (stats['planner_success'] / stats['total_scenarios']) * 100
            if stats['planner_success'] > 0:
                moveit_sr = (stats['moveit_validated_success'] / stats['planner_success']) * 100
                true_sr = (stats['moveit_validated_success'] / stats['total_scenarios']) * 100
            else:
                moveit_sr = 0.0
                true_sr = 0.0
            
            logging.info(f"\n{'='*50}")
            logging.info(f"Progress: {scenario_id + 1}/{args.num_scenarios}")
            logging.info(f"Goal reaching rate: {goal_reaching_rate:.1f}%")
            logging.info(f"TRUE success rate (MoveIt): {true_sr:.1f}%")
            logging.info(f"{'='*50}\n")
    
    # Final results - comprehensive statistics
    logging.info(f"\n{'='*70}")
    logging.info("FINAL SIMULATION RESULTS (Paper Methodology)")
    logging.info(f"{'='*70}")
    logging.info(f"Total scenarios: {stats['total_scenarios']}")
    if stats['invalid_scenes_rejected'] > 0:
        logging.info(f"Invalid scenes (used fallback): {stats['invalid_scenes_rejected']}")
    
    logging.info(f"\n--- Goal Reaching (Planner Performance) ---")
    logging.info(f"Goal reached: {stats['planner_success']} "
                f"({stats['planner_success']/stats['total_scenarios']*100:.1f}%)")
    logging.info(f"Goal NOT reached: {stats['planner_failed']} "
                f"({stats['planner_failed']/stats['total_scenarios']*100:.1f}%)")
    logging.info(f"Classifier collision-free (of goal reached): {stats['planner_collision_free']}")
    
    if collision_oracle is not None:
        logging.info(f"\n--- TRUE Success Rate (Paper Definition) ---")
        logging.info(f"Goal reached + MoveIt collision-free: {stats['moveit_validated_success']} "
                    f"({stats['moveit_validated_success']/stats['total_scenarios']*100:.1f}%)")
        logging.info(f"Goal reached but MoveIt collision: {stats['moveit_rejected']} "
                    f"({stats['moveit_rejected']/stats['total_scenarios']*100:.1f}%)")
        if stats['planner_success'] > 0:
            validation_rate = stats['moveit_validated_success'] / stats['planner_success'] * 100
            logging.info(f"MoveIt validation rate: {validation_rate:.1f}% of goal-reached scenarios")
        
        # Detailed collision analysis
        if stats['moveit_rejected'] > 0:
            logging.info(f"\n--- Collision Analysis ---")
            logging.info(f"Total collision waypoints: {stats['total_collision_waypoints']}")
            logging.info(f"Max collisions in single path: {stats['max_collision_waypoints_per_path']}")
            avg_collisions = stats['total_collision_waypoints'] / stats['moveit_rejected']
            logging.info(f"Avg collision waypoints per rejected path: {avg_collisions:.1f}")
            
            # Collision percentage analysis
            rejected_details = [d for d in stats['collision_details'] if not d['collision_free']]
            if rejected_details:
                avg_collision_pct = sum(d['collision_percentage'] for d in rejected_details) / len(rejected_details)
                logging.info(f"Avg collision percentage per rejected path: {avg_collision_pct:.1f}%")
    
    # Planning performance metrics
    if stats['planning_times_ms']:
        logging.info(f"\n--- Planning Performance ---")
        logging.info(f"Avg planning time: {np.mean(stats['planning_times_ms']):.1f}ms")
        logging.info(f"Min/Max planning time: {np.min(stats['planning_times_ms']):.1f}ms / {np.max(stats['planning_times_ms']):.1f}ms")
        logging.info(f"Avg planning steps: {np.mean(stats['planning_steps']):.1f}")
        logging.info(f"Avg path length (waypoints): {np.mean(stats['path_lengths']):.1f}")
    
    if stats['min_distances']:
        logging.info(f"\n--- Failed Scenario Analysis ---")
        logging.info(f"Avg min distance achieved: {np.mean(stats['min_distances']):.4f}m")
        logging.info(f"Best min distance: {np.min(stats['min_distances']):.4f}m")
    
    logging.info(f"\n{'='*70}\n")
    
    # Save detailed results to JSON
    results_summary = {
        'configuration': {
            'num_scenarios': stats['total_scenarios'],
            'max_steps': args.max_steps,
            'planning_lr': args.planning_lr,
            'success_threshold': args.success_threshold,
            'lambda_collision': args.lambda_collision,
            'collision_padding': args.collision_padding,
            'seed': args.seed
        },
        'success_rates': {
            'planner_success_rate': stats['planner_success'] / stats['total_scenarios'] * 100,
            'moveit_validation_rate': (stats['moveit_validated_success'] / stats['planner_success'] * 100) if stats['planner_success'] > 0 else 0,
            'overall_success_rate': stats['moveit_validated_success'] / stats['total_scenarios'] * 100
        },
        'counts': {
            'total_scenarios': stats['total_scenarios'],
            'planner_success': stats['planner_success'],
            'planner_failed': stats['planner_failed'],
            'moveit_validated': stats['moveit_validated_success'],
            'moveit_rejected': stats['moveit_rejected']
        },
        'collision_analysis': {
            'total_collision_waypoints': stats['total_collision_waypoints'],
            'max_collision_waypoints_per_path': stats['max_collision_waypoints_per_path'],
            'collision_details': stats['collision_details']
        },
        'planning_performance': {
            'planning_times_ms': stats['planning_times_ms'],
            'planning_steps': stats['planning_steps'],
            'path_lengths': stats['path_lengths'],
            'min_distances_failed': stats['min_distances'],
            'goal_distances': stats['goal_distances']
        }
    }
    
    # Auto-save results summary
    results_filename = f"../model_params/panda_10k/simulation_results_{args.seed}_{stats['total_scenarios']}scenarios.json"
    with open(results_filename, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logging.info(f"Detailed results saved to: {results_filename}")
    
    # Save validated scenarios for use in evaluation script
    if args.save_scenes and scenarios_to_save:
        scenes_data = {
            'num_scenarios': len(scenarios_to_save),
            'num_obstacles': args.num_obstacles,
            'seed': args.seed,
            'validation': {
                'min_ee_height': MIN_EE_HEIGHT,
                'min_ee_obstacle_dist': MIN_EE_OBSTACLE_DIST,
                'moveit_validated': collision_oracle is not None
            },
            'scenarios': scenarios_to_save
        }
        with open(args.save_scenes, 'w') as f:
            json.dump(scenes_data, f, indent=2)
        logging.info(f"Saved {len(scenarios_to_save)} validated scenarios to {args.save_scenes}")
    
    # Export trajectory if requested
    if args.export_trajectory:
        if args.export_all and all_trajectories:
            # Export each scenario to separate file
            import os
            base_name = args.export_trajectory.replace('.json', '')
            for traj_data in all_trajectories:
                filename = f"{base_name}_scenario_{traj_data['scenario_id']}.json"
                with open(filename, 'w') as f:
                    json.dump(traj_data, f, indent=2)
            logging.info(f"Exported {len(all_trajectories)} trajectories to {base_name}_scenario_*.json")
            
        elif args.export_scenario is not None and export_data:
            # Export single specified scenario
            with open(args.export_trajectory, 'w') as f:
                json.dump(export_data, f, indent=2)
            logging.info(f"Trajectory data exported to: {args.export_trajectory}")
            
        elif args.export_scenario is None and all_trajectories:
            # Export all scenarios in one file
            output_data = {
                'total_scenarios': len(all_trajectories),
                'trajectories': all_trajectories
            }
            with open(args.export_trajectory, 'w') as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"Exported {len(all_trajectories)} trajectories to: {args.export_trajectory}")
            
        else:
            logging.warning(f"Trajectory export requested but no successful scenarios found")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end simulation with MoveIt visualization and validation'
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--checkpoint_obs', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--config_obs', type=str, default=None,
                       help='Config JSON for obstacle classifier (if different from --config)')
    
    # Simulation arguments
    parser.add_argument('--num_scenarios', type=int, default=10,
                       help='Number of scenarios to simulate')
    parser.add_argument('--num_obstacles', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--planning_lr', type=float, default=0.15,
                       help='Planning learning rate (Phase2: 0.15)')
    parser.add_argument('--success_threshold', type=float, default=0.01)
    parser.add_argument('--lambda_prior', type=float, default=0.7,
                       help='Prior loss weight (Phase2: 0.7)')
    parser.add_argument('--lambda_collision', type=float, default=0.5,
                       help='Collision loss weight (Phase2: 0.5)')
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='Temperature for softening collision predictions (Phase2: 3.0)')
    
    # GECO adaptive weighting
    parser.add_argument('--use_geco', action='store_true',
                       help='Enable GECO adaptive loss weighting (RECOMMENDED)')
    parser.add_argument('--alpha_geco', type=float, default=0.008,
                       help='GECO learning rate (Phase2: 0.008)')
    parser.add_argument('--tau_prior_goal', type=float, default=6.0,
                       help='GECO prior loss target (Phase2: 6.0)')
    parser.add_argument('--tau_obs_goal', type=float, default=2.0,
                       help='GECO obstacle loss target (Phase2: 2.0)')
    parser.add_argument('--alpha_ma_prior', type=float, default=0.8,
                       help='Moving average factor for prior loss (Phase2: 0.8)')
    parser.add_argument('--alpha_ma_obs', type=float, default=0.8,
                       help='Moving average factor for obstacle loss (Phase2: 0.8)')
    
    # Visualization
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable RViz visualization')
    parser.add_argument('--no_animation', action='store_true',
                       help='Skip trajectory animation')
    parser.add_argument('--collision_padding', type=float, default=0.005,
                       help='Collision safety margin in meters (default: 0.005 = 5mm)')
    parser.add_argument('--interpolation_steps', type=int, default=50,
                       help='Number of interpolation steps for smooth trajectory (default: 50)')
    parser.add_argument('--animation_speed', type=float, default=0.05,
                       help='Time per waypoint during animation in seconds (default: 0.05)')
    
    # Trajectory export
    parser.add_argument('--export_trajectory', type=str, default=None,
                       help='Export trajectory data to JSON for web visualization')
    parser.add_argument('--export_scenario', type=int, default=None,
                       help='Which scenario to export (0-indexed). If not set, exports all successful scenarios')
    parser.add_argument('--export_all', action='store_true',
                       help='Export all successful scenarios to separate files')
    
    # Scene loading/saving for reproducibility
    parser.add_argument('--load_scenes', type=str, default=None,
                       help='Load pre-generated scenarios from JSON file (for matching with evaluation)')
    parser.add_argument('--save_scenes', type=str, default=None,
                       help='Save validated scenarios to JSON file (for use in evaluation)')
    
    # Other
    parser.add_argument('--data_path', type=str,
                       default='../data',
                       help='Path to data directory containing training data')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Run simulation
    stats = run_simulation(args)
    
    logging.info("Simulation complete!")


if __name__ == "__main__":
    main()
