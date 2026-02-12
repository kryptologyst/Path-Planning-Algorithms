"""
Robot models and simulation for path planning.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import math


class BaseRobot(ABC):
    """Base class for robot models."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize robot."""
        self.config = kwargs
    
    @abstractmethod
    def get_position(self) -> Tuple[float, float]:
        """Get current robot position."""
        pass
    
    @abstractmethod
    def set_position(self, position: Tuple[float, float]) -> None:
        """Set robot position."""
        pass
    
    @abstractmethod
    def can_move_to(self, position: Tuple[float, float]) -> bool:
        """Check if robot can move to position."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> Tuple[float, float]:
        """Get robot dimensions."""
        pass


class DifferentialDriveRobot(BaseRobot):
    """Differential drive robot model."""
    
    def __init__(
        self,
        width: float = 0.5,
        length: float = 0.8,
        max_velocity: float = 1.0,
        max_angular_velocity: float = 1.0,
        **kwargs: Any
    ) -> None:
        """
        Initialize differential drive robot.
        
        Args:
            width: Robot width
            length: Robot length
            max_velocity: Maximum linear velocity
            max_angular_velocity: Maximum angular velocity
        """
        super().__init__(**kwargs)
        self.width = width
        self.length = length
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
    
    def get_position(self) -> Tuple[float, float]:
        """Get current robot position."""
        return (self.position[0], self.position[1])
    
    def set_position(self, position: Tuple[float, float]) -> None:
        """Set robot position."""
        self.position = np.array(position)
    
    def can_move_to(self, position: Tuple[float, float]) -> bool:
        """Check if robot can move to position."""
        # Simple collision check - can be extended
        return True
    
    def get_dimensions(self) -> Tuple[float, float]:
        """Get robot dimensions."""
        return (self.length, self.width)
    
    def get_orientation(self) -> float:
        """Get robot orientation."""
        return self.orientation
    
    def set_orientation(self, orientation: float) -> None:
        """Set robot orientation."""
        self.orientation = orientation
    
    def move(self, linear_velocity: float, angular_velocity: float, dt: float) -> None:
        """
        Move robot with given velocities.
        
        Args:
            linear_velocity: Linear velocity
            angular_velocity: Angular velocity
            dt: Time step
        """
        # Clamp velocities
        linear_velocity = np.clip(linear_velocity, -self.max_velocity, self.max_velocity)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Update orientation
        self.orientation += angular_velocity * dt
        
        # Update position
        dx = linear_velocity * math.cos(self.orientation) * dt
        dy = linear_velocity * math.sin(self.orientation) * dt
        
        self.position[0] += dx
        self.position[1] += dy


class RobotSimulator:
    """Simple robot simulator."""
    
    def __init__(self, robot: BaseRobot) -> None:
        """
        Initialize simulator.
        
        Args:
            robot: Robot to simulate
        """
        self.robot = robot
        self.time = 0.0
        self.trajectory: List[Tuple[float, float, float]] = []  # (x, y, theta)
    
    def reset(self, position: Tuple[float, float], orientation: float = 0.0) -> None:
        """
        Reset simulator.
        
        Args:
            position: Initial position
            orientation: Initial orientation
        """
        self.robot.set_position(position)
        self.robot.set_orientation(orientation)
        self.time = 0.0
        self.trajectory = [(position[0], position[1], orientation)]
    
    def step(self, linear_velocity: float, angular_velocity: float, dt: float = 0.1) -> None:
        """
        Simulate one time step.
        
        Args:
            linear_velocity: Linear velocity command
            angular_velocity: Angular velocity command
            dt: Time step
        """
        self.robot.move(linear_velocity, angular_velocity, dt)
        self.time += dt
        
        # Record trajectory
        pos = self.robot.get_position()
        orientation = self.robot.get_orientation()
        self.trajectory.append((pos[0], pos[1], orientation))
    
    def follow_path(
        self,
        path: List[Tuple[int, int]],
        linear_velocity: float = 1.0,
        dt: float = 0.1
    ) -> List[Tuple[float, float, float]]:
        """
        Follow a discrete path.
        
        Args:
            path: List of discrete positions
            linear_velocity: Linear velocity
            dt: Time step
            
        Returns:
            Continuous trajectory
        """
        if not path:
            return []
        
        trajectory = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Calculate direction
            dx = next_pos[1] - current[1]  # Convert to continuous coordinates
            dy = next_pos[0] - current[0]
            
            # Calculate distance and time
            distance = math.sqrt(dx**2 + dy**2)
            time_to_reach = distance / linear_velocity
            
            # Calculate orientation
            target_orientation = math.atan2(dy, dx)
            
            # Move towards target
            steps = int(time_to_reach / dt)
            for _ in range(steps):
                # Simple control - move towards target
                pos = self.robot.get_position()
                orientation = self.robot.get_orientation()
                
                # Calculate error
                error_x = next_pos[1] - pos[0]
                error_y = next_pos[0] - pos[1]
                
                # Simple proportional control
                linear_cmd = linear_velocity
                angular_cmd = 2.0 * (target_orientation - orientation)
                
                self.step(linear_cmd, angular_cmd, dt)
                trajectory.append((pos[0], pos[1], orientation))
        
        return trajectory
    
    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get recorded trajectory."""
        return self.trajectory.copy()
    
    def get_current_state(self) -> Tuple[float, float, float]:
        """Get current robot state."""
        pos = self.robot.get_position()
        orientation = self.robot.get_orientation()
        return (pos[0], pos[1], orientation)
