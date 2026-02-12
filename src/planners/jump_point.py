"""
Jump Point Search path planning algorithm implementation.

Jump Point Search is an optimization of A* that reduces the number of
nodes explored by identifying "jump points" that are guaranteed to be
on optimal paths.
"""

import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np

from .base import BasePlanner, PlanningResult


class JumpPointPlanner(BasePlanner):
    """
    Jump Point Search path planning algorithm.
    
    Jump Point Search identifies jump points that are guaranteed to be
    on optimal paths, reducing the search space compared to A*.
    """
    
    def __init__(
        self,
        allow_diagonal: bool = True,
        smoothing: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize Jump Point Search planner.
        
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
        Plan a path using Jump Point Search algorithm.
        
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
                computation_time=self._end_timing(),
                nodes_expanded=self._nodes_expanded,
                metadata={"error": "Invalid goal position"}
            )
        
        # Initialize data structures
        open_list = []
        closed_set: Set[Tuple[int, int]] = set()
        
        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
        f_scores: Dict[Tuple[int, int], float] = {
            start: self._heuristic(start, goal)
        }
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        heapq.heappush(open_list, (f_scores[start], start))
        
        while open_list:
            current_f, current = heapq.heappop(open_list)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            self._increment_nodes_expanded()
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                
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
                        "algorithm": "Jump Point Search",
                        "allow_diagonal": self.allow_diagonal,
                        "smoothing": self.smoothing,
                        "path_length": len(path)
                    }
                )
            
            # Get jump points
            jump_points = self._get_jump_points(current, goal, environment)
            
            for jump_point in jump_points:
                if jump_point in closed_set:
                    continue
                
                tentative_g = g_scores[current] + self._distance(current, jump_point)
                
                if jump_point not in g_scores or tentative_g < g_scores[jump_point]:
                    came_from[jump_point] = current
                    g_scores[jump_point] = tentative_g
                    f_scores[jump_point] = tentative_g + self._heuristic(jump_point, goal)
                    heapq.heappush(open_list, (f_scores[jump_point], jump_point))
        
        return PlanningResult(
            success=False,
            path=[],
            cost=float('inf'),
            computation_time=self._end_timing(),
            nodes_expanded=self._nodes_expanded,
            metadata={"error": "No path found"}
        )
    
    def _get_jump_points(
        self,
        current: Tuple[int, int],
        goal: Tuple[int, int],
        environment: "BaseEnvironment"
    ) -> List[Tuple[int, int]]:
        """Get jump points from current position."""
        jump_points = []
        
        # Get all possible directions
        directions = self._get_directions(current, goal)
        
        for direction in directions:
            jump_point = self._jump(current, direction, goal, environment)
            if jump_point:
                jump_points.append(jump_point)
        
        return jump_points
    
    def _get_directions(
        self,
        current: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Get directions to explore from current position."""
        directions = []
        
        # Calculate direction to goal
        dr = 1 if goal[0] > current[0] else -1 if goal[0] < current[0] else 0
        dc = 1 if goal[1] > current[1] else -1 if goal[1] < current[1] else 0
        
        # Add primary direction
        if dr != 0 or dc != 0:
            directions.append((dr, dc))
        
        # Add orthogonal directions
        if dr != 0:
            directions.append((dr, 0))
        if dc != 0:
            directions.append((0, dc))
        
        # Add diagonal directions if allowed
        if self.allow_diagonal and dr != 0 and dc != 0:
            directions.append((dr, dc))
        
        return directions
    
    def _jump(
        self,
        current: Tuple[int, int],
        direction: Tuple[int, int],
        goal: Tuple[int, int],
        environment: "BaseEnvironment"
    ) -> Optional[Tuple[int, int]]:
        """Jump in a direction until hitting obstacle or jump point."""
        dr, dc = direction
        next_pos = (current[0] + dr, current[1] + dc)
        
        # Check if next position is valid
        if not environment.is_valid_position(next_pos):
            return None
        
        # Check if we reached the goal
        if next_pos == goal:
            return next_pos
        
        # Check for forced neighbors (jump point condition)
        if self._has_forced_neighbors(next_pos, direction, environment):
            return next_pos
        
        # Continue jumping
        return self._jump(next_pos, direction, goal, environment)
    
    def _has_forced_neighbors(
        self,
        pos: Tuple[int, int],
        direction: Tuple[int, int],
        environment: "BaseEnvironment"
    ) -> bool:
        """Check if position has forced neighbors (jump point condition)."""
        dr, dc = direction
        r, c = pos
        
        # Check for forced neighbors based on direction
        if dr != 0 and dc != 0:  # Diagonal direction
            # Check for forced neighbors in diagonal direction
            if not environment.is_valid_position((r - dr, c)) and \
               environment.is_valid_position((r - dr, c + dc)):
                return True
            if not environment.is_valid_position((r, c - dc)) and \
               environment.is_valid_position((r + dr, c - dc)):
                return True
        elif dr != 0:  # Horizontal direction
            # Check for forced neighbors in horizontal direction
            if not environment.is_valid_position((r, c - 1)) and \
               environment.is_valid_position((r + dr, c - 1)):
                return True
            if not environment.is_valid_position((r, c + 1)) and \
               environment.is_valid_position((r + dr, c + 1)):
                return True
        elif dc != 0:  # Vertical direction
            # Check for forced neighbors in vertical direction
            if not environment.is_valid_position((r - 1, c)) and \
               environment.is_valid_position((r - 1, c + dc)):
                return True
            if not environment.is_valid_position((r + 1, c)) and \
               environment.is_valid_position((r + 1, c + dc)):
                return True
        
        return False
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance."""
        return self._distance(pos1, pos2)
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from start to goal."""
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total path cost."""
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._distance(path[i], path[i + 1])
        
        return total_cost
