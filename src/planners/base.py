"""
Base classes for path planning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import time
import numpy as np


@dataclass
class PlanningResult:
    """Result of a path planning operation."""
    
    success: bool
    path: List[Tuple[int, int]]
    cost: float
    computation_time: float
    nodes_expanded: int
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate the planning result."""
        if self.success and not self.path:
            raise ValueError("Successful planning must have a path")
        if self.success and self.cost < 0:
            raise ValueError("Cost must be non-negative for successful planning")


class BasePlanner(ABC):
    """Base class for all path planning algorithms."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the planner with configuration parameters."""
        self.config = kwargs
        self._reset_stats()
    
    def _reset_stats(self) -> None:
        """Reset planning statistics."""
        self._nodes_expanded = 0
        self._start_time = 0.0
    
    def _start_timing(self) -> None:
        """Start timing the planning operation."""
        self._start_time = time.time()
    
    def _end_timing(self) -> float:
        """End timing and return elapsed time."""
        return time.time() - self._start_time
    
    def _increment_nodes_expanded(self) -> None:
        """Increment the counter of expanded nodes."""
        self._nodes_expanded += 1
    
    @abstractmethod
    def plan(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs: Any
    ) -> PlanningResult:
        """
        Plan a path from start to goal in the given environment.
        
        Args:
            environment: The environment to plan in
            start: Starting position (row, col)
            goal: Goal position (row, col)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            PlanningResult containing the path and metadata
        """
        pass
    
    def get_neighbors(
        self,
        node: Tuple[int, int],
        environment: "BaseEnvironment",
        allow_diagonal: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Get valid neighbors of a node in the environment.
        
        Args:
            node: Current node position
            environment: The environment
            allow_diagonal: Whether to include diagonal moves
            
        Returns:
            List of valid neighbor positions
        """
        neighbors = []
        row, col = node
        
        # 4-connected neighbors
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Add diagonal neighbors if allowed
        if allow_diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if environment.is_valid_position((new_row, new_col)):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def calculate_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        metric: str = "euclidean"
    ) -> float:
        """
        Calculate distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            metric: Distance metric ('euclidean', 'manhattan', 'diagonal')
            
        Returns:
            Distance between positions
        """
        r1, c1 = pos1
        r2, c2 = pos2
        
        if metric == "euclidean":
            return np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
        elif metric == "manhattan":
            return abs(r1 - r2) + abs(c1 - c2)
        elif metric == "diagonal":
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            return max(dr, dc) + (np.sqrt(2) - 1) * min(dr, dc)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        environment: "BaseEnvironment"
    ) -> List[Tuple[int, int]]:
        """
        Smooth a path by removing unnecessary waypoints.
        
        Args:
            path: Original path
            environment: The environment
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if environment.is_line_of_sight(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed
