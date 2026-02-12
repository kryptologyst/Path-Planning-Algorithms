"""
Utility functions and helpers for path planning.
"""

from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import random
import time
from pathlib import Path
import yaml


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def calculate_path_metrics(
    path: List[Tuple[int, int]],
    environment: "BaseEnvironment"
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a path.
    
    Args:
        path: List of positions
        environment: The environment
        
    Returns:
        Dictionary of metrics
    """
    if not path:
        return {
            'length': 0.0,
            'cost': 0.0,
            'clearance': 0.0,
            'smoothness': 0.0,
            'curvature': 0.0
        }
    
    # Path length
    length = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        length += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Path cost (same as length for now)
    cost = length
    
    # Average clearance
    clearances = []
    for position in path:
        clearance = environment.calculate_clearance(position)
        clearances.append(clearance)
    avg_clearance = np.mean(clearances) if clearances else 0.0
    
    # Smoothness (inverse of curvature)
    smoothness = 1.0
    curvature = 0.0
    
    if len(path) >= 3:
        total_curvature = 0.0
        
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            
            # Calculate angle between vectors
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                total_curvature += angle
        
        curvature = total_curvature / (len(path) - 2)
        smoothness = 1.0 / (1.0 + curvature)
    
    return {
        'length': length,
        'cost': cost,
        'clearance': avg_clearance,
        'smoothness': smoothness,
        'curvature': curvature
    }


def interpolate_path(
    path: List[Tuple[int, int]],
    resolution: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Interpolate a discrete path to higher resolution.
    
    Args:
        path: Discrete path
        resolution: Interpolation resolution
        
    Returns:
        Interpolated path
    """
    if len(path) <= 1:
        return [(float(p[0]), float(p[1])) for p in path]
    
    interpolated = []
    
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        
        # Calculate distance
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Number of interpolation points
        num_points = max(1, int(distance / resolution))
        
        # Interpolate
        for j in range(num_points):
            t = j / num_points
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            interpolated.append((x, y))
    
    # Add final point
    interpolated.append((float(path[-1][0]), float(path[-1][1])))
    
    return interpolated


def smooth_path_spline(
    path: List[Tuple[int, int]],
    smoothing_factor: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Smooth a path using spline interpolation.
    
    Args:
        path: Discrete path
        smoothing_factor: Smoothing factor (0-1)
        
    Returns:
        Smoothed path
    """
    if len(path) < 3:
        return [(float(p[0]), float(p[1])) for p in path]
    
    try:
        from scipy.interpolate import UnivariateSpline
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        
        # Create parameter t
        t = np.linspace(0, 1, len(path))
        
        # Create splines
        spline_x = UnivariateSpline(t, x_coords, s=smoothing_factor)
        spline_y = UnivariateSpline(t, y_coords, s=smoothing_factor)
        
        # Generate smooth path
        t_smooth = np.linspace(0, 1, len(path) * 3)
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)
        
        return list(zip(x_smooth, y_smooth))
    
    except ImportError:
        # Fallback to linear interpolation
        return interpolate_path(path, resolution=0.5)


def validate_path(
    path: List[Tuple[int, int]],
    environment: "BaseEnvironment"
) -> bool:
    """
    Validate that a path is collision-free.
    
    Args:
        path: Path to validate
        environment: The environment
        
    Returns:
        True if path is valid, False otherwise
    """
    if not path:
        return False
    
    # Check each position
    for position in path:
        if not environment.is_valid_position(position):
            return False
    
    # Check line of sight between consecutive points
    for i in range(len(path) - 1):
        if not environment.is_line_of_sight(path[i], path[i + 1]):
            return False
    
    return True


def optimize_path(
    path: List[Tuple[int, int]],
    environment: "BaseEnvironment"
) -> List[Tuple[int, int]]:
    """
    Optimize a path by removing unnecessary waypoints.
    
    Args:
        path: Original path
        environment: The environment
        
    Returns:
        Optimized path
    """
    if len(path) <= 2:
        return path
    
    optimized = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if environment.is_line_of_sight(path[i], path[j]):
                optimized.append(path[j])
                i = j
                break
            j -= 1
        else:
            optimized.append(path[i + 1])
            i += 1
    
    return optimized


def create_benchmark_problems(
    environment: "BaseEnvironment",
    num_problems: int = 100,
    seed: Optional[int] = None
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Create benchmark problem instances.
    
    Args:
        environment: The environment
        num_problems: Number of problems to create
        seed: Random seed
        
    Returns:
        List of (start, goal) tuples
    """
    if seed is not None:
        set_random_seed(seed)
    
    problems = []
    
    for _ in range(num_problems):
        start = environment.get_random_valid_position()
        goal = environment.get_random_valid_position()
        
        # Ensure start and goal are different
        while start == goal:
            goal = environment.get_random_valid_position()
        
        problems.append((start, goal))
    
    return problems


def format_time(seconds: float) -> str:
    """
    Format time in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    else:
        return f"{seconds:.3f} s"


def format_distance(distance: float) -> str:
    """
    Format distance in a human-readable way.
    
    Args:
        distance: Distance value
        
    Returns:
        Formatted distance string
    """
    if distance < 1.0:
        return f"{distance:.3f}"
    else:
        return f"{distance:.2f}"
