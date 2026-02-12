"""
Dijkstra's path planning algorithm implementation.

Dijkstra's algorithm finds the shortest path from a start node to all
other nodes in a graph, guaranteeing optimality but being less efficient
than A* for single-goal pathfinding.
"""

import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np

from .base import BasePlanner, PlanningResult


class DijkstraPlanner(BasePlanner):
    """
    Dijkstra's path planning algorithm.
    
    Dijkstra's algorithm guarantees finding the optimal path but
    explores more nodes than A* since it doesn't use a heuristic.
    """
    
    def __init__(
        self,
        allow_diagonal: bool = True,
        smoothing: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize Dijkstra planner.
        
        Args:
            allow_diagonal: Whether to allow diagonal moves
            smoothing: Whether to smooth the resulting path
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.allow_diagonal = allow_diagonal
        self.smoothing = smoothing
    
    def plan(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs: Any
    ) -> PlanningResult:
        """
        Plan a path using Dijkstra's algorithm.
        
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
        
        # Initialize data structures
        open_list = []  # Priority queue: (cost, node)
        closed_set: Set[Tuple[int, int]] = set()
        
        # Cost from start to each node
        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # Parent pointers for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Add start node to open list
        heapq.heappush(open_list, (0.0, start))
        
        while open_list:
            # Get node with lowest cost
            current_cost, current = heapq.heappop(open_list)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            closed_set.add(current)
            self._increment_nodes_expanded()
            
            # Check if goal reached
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                
                # Smooth path if requested
                if self.smoothing:
                    path = self.smooth_path(path, environment)
                
                cost = self._calculate_path_cost(path)
                computation_time = self._end_timing()
                
                return PlanningResult(
                    success=True,
                    path=path,
                    cost=cost,
                    computation_time=computation_time,
                    nodes_expanded=self._nodes_expanded,
                    metadata={
                        "algorithm": "Dijkstra",
                        "allow_diagonal": self.allow_diagonal,
                        "smoothing": self.smoothing,
                        "path_length": len(path)
                    }
                )
            
            # Explore neighbors
            neighbors = self.get_neighbors(current, environment, self.allow_diagonal)
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_scores[current] + self._get_move_cost(current, neighbor)
                
                # If this path to neighbor is better than any previous path
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    
                    heapq.heappush(open_list, (tentative_g, neighbor))
        
        # No path found
        return PlanningResult(
            success=False,
            path=[],
            cost=float('inf'),
            computation_time=self._end_timing(),
            nodes_expanded=self._nodes_expanded,
            metadata={"error": "No path found"}
        )
    
    def _get_move_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int]
    ) -> float:
        """
        Calculate the cost of moving from one position to another.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            Move cost
        """
        r1, c1 = from_pos
        r2, c2 = to_pos
        
        # Diagonal moves cost sqrt(2), straight moves cost 1
        if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
            return np.sqrt(2)
        else:
            return 1.0
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal.
        
        Args:
            came_from: Parent pointers
            current: Current node (goal)
            
        Returns:
            Path from start to goal
        """
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate the total cost of a path.
        
        Args:
            path: List of positions
            
        Returns:
            Total path cost
        """
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._get_move_cost(path[i], path[i + 1])
        
        return total_cost
