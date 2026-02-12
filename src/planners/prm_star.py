"""
PRM* (Probabilistic Roadmap Method Star) path planning algorithm.

PRM* is a sampling-based path planning algorithm that builds a roadmap
by sampling random configurations and connecting nearby configurations
with collision-free paths.
"""

import random
import math
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np

from .base import BasePlanner, PlanningResult


class PRMStarPlanner(BasePlanner):
    """
    PRM* path planning algorithm.
    
    PRM* builds a roadmap by sampling random configurations and
    connecting nearby configurations, then uses graph search to
    find paths.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        connection_radius: float = 2.0,
        max_connections: int = 10,
        **kwargs: Any
    ) -> None:
        """
        Initialize PRM* planner.
        
        Args:
            num_samples: Number of samples to generate
            connection_radius: Radius for connecting samples
            max_connections: Maximum connections per sample
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.max_connections = max_connections
        self.roadmap: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.samples: List[Tuple[int, int]] = []
    
    def plan(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs: Any
    ) -> PlanningResult:
        """
        Plan a path using PRM* algorithm.
        
        Args:
            environment: The environment to plan in
            start: Starting position (row, col)
            goal: Goal position (row, col)
            **kwargs: Additional parameters
            
        Returns:
            PlanningResult containing the path and metadata
        """
        self._reset_stats()
        self._start_timing()
        
        # Validate inputs
        if not environment.is_valid_position(start):
            return PlanningResult(
                success=False,
                path=[],
                cost=float('inf'),
                computation_time=self._end_timing(),
                nodes_expanded=self._nodes_expanded,
                metadata={"error": "Invalid start position"}
            )
        
        if not environment.is_valid_position(goal):
            return PlanningResult(
                success=False,
                path=[],
                cost=float('inf'),
                computation_time=self._end_timing(),
                nodes_expanded=self._nodes_expanded,
                metadata={"error": "Invalid goal position"}
            )
        
        # Build roadmap
        self._build_roadmap(environment)
        
        # Add start and goal to roadmap
        self._connect_to_roadmap(start, environment)
        self._connect_to_roadmap(goal, environment)
        
        # Search for path
        path = self._search_path(start, goal)
        
        computation_time = self._end_timing()
        
        if path:
            cost = self._calculate_path_cost(path)
            
            return PlanningResult(
                success=True,
                path=path,
                cost=cost,
                computation_time=computation_time,
                nodes_expanded=self._nodes_expanded,
                metadata={
                    "algorithm": "PRM*",
                    "num_samples": self.num_samples,
                    "connection_radius": self.connection_radius,
                    "max_connections": self.max_connections,
                    "path_length": len(path)
                }
            )
        else:
            return PlanningResult(
                success=False,
                path=[],
                cost=float('inf'),
                computation_time=computation_time,
                nodes_expanded=self._nodes_expanded,
                metadata={"error": "No path found"}
            )
    
    def _build_roadmap(self, environment: "BaseEnvironment") -> None:
        """Build the PRM* roadmap."""
        self.roadmap = {}
        self.samples = []
        
        # Generate random samples
        for _ in range(self.num_samples):
            sample = self._sample_random_point(environment)
            self.samples.append(sample)
            self.roadmap[sample] = []
        
        # Connect nearby samples
        for sample in self.samples:
            nearby_samples = self._find_nearby_samples(sample)
            
            # Sort by distance
            nearby_samples.sort(key=lambda s: self._distance(sample, s))
            
            # Connect to nearest samples
            connections = 0
            for nearby in nearby_samples:
                if connections >= self.max_connections:
                    break
                
                if self._can_connect(sample, nearby, environment):
                    self.roadmap[sample].append(nearby)
                    self.roadmap[nearby].append(sample)
                    connections += 1
    
    def _sample_random_point(self, environment: "BaseEnvironment") -> Tuple[int, int]:
        """Sample a random valid point."""
        height, width = environment.get_dimensions()
        
        for _ in range(100):  # Try up to 100 times
            row = random.randint(0, height - 1)
            col = random.randint(0, width - 1)
            if environment.is_valid_position((row, col)):
                return (row, col)
        
        # Fallback
        return environment.get_random_valid_position()
    
    def _find_nearby_samples(self, sample: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find samples within connection radius."""
        nearby = []
        
        for other_sample in self.samples:
            if other_sample != sample:
                if self._distance(sample, other_sample) <= self.connection_radius:
                    nearby.append(other_sample)
        
        return nearby
    
    def _can_connect(
        self,
        sample1: Tuple[int, int],
        sample2: Tuple[int, int],
        environment: "BaseEnvironment"
    ) -> bool:
        """Check if two samples can be connected."""
        return environment.is_line_of_sight(sample1, sample2)
    
    def _connect_to_roadmap(
        self,
        point: Tuple[int, int],
        environment: "BaseEnvironment"
    ) -> None:
        """Connect a point to the roadmap."""
        if point not in self.roadmap:
            self.roadmap[point] = []
        
        # Find nearby samples
        nearby_samples = []
        for sample in self.samples:
            if self._distance(point, sample) <= self.connection_radius:
                nearby_samples.append(sample)
        
        # Sort by distance
        nearby_samples.sort(key=lambda s: self._distance(point, s))
        
        # Connect to nearest samples
        connections = 0
        for nearby in nearby_samples:
            if connections >= self.max_connections:
                break
            
            if self._can_connect(point, nearby, environment):
                self.roadmap[point].append(nearby)
                self.roadmap[nearby].append(point)
                connections += 1
    
    def _search_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Search for path using Dijkstra's algorithm."""
        if start not in self.roadmap or goal not in self.roadmap:
            return None
        
        # Dijkstra's algorithm
        open_list = [(0.0, start)]
        closed_set = set()
        g_scores = {start: 0.0}
        came_from = {}
        
        while open_list:
            current_cost, current = open_list.pop(0)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            self._increment_nodes_expanded()
            
            if current == goal:
                # Reconstruct path
                path = [goal]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            # Explore neighbors
            for neighbor in self.roadmap[current]:
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_scores[current] + self._distance(current, neighbor)
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    open_list.append((tentative_g, neighbor))
            
            # Sort by cost
            open_list.sort(key=lambda x: x[0])
        
        return None
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total path cost."""
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._distance(path[i], path[i + 1])
        
        return total_cost
