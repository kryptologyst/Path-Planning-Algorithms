"""
D* path planning algorithm implementation.

D* is a dynamic path planning algorithm that can efficiently replan
when the environment changes, making it suitable for real-time
applications with dynamic obstacles.
"""

from typing import List, Tuple, Dict, Set, Optional, Any
import heapq
import numpy as np

from .base import BasePlanner, PlanningResult


class DStarPlanner(BasePlanner):
    """
    D* path planning algorithm.
    
    D* is designed for dynamic environments where obstacles can change
    during path execution. It maintains a graph and efficiently updates
    the path when changes are detected.
    """
    
    def __init__(
        self,
        allow_diagonal: bool = True,
        smoothing: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize D* planner.
        
        Args:
            allow_diagonal: Whether to allow diagonal moves
            smoothing: Whether to smooth the resulting path
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.allow_diagonal = allow_diagonal
        self.smoothing = smoothing
        self.graph: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.open_list: List[Tuple[float, Tuple[int, int]]] = []
    
    def plan(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs: Any
    ) -> PlanningResult:
        """
        Plan a path using D* algorithm.
        
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
        
        # Initialize graph
        self._initialize_graph(environment, start, goal)
        
        # Run D* algorithm
        success = self._d_star_search(environment, start, goal)
        
        computation_time = self._end_timing()
        
        if success:
            path = self._reconstruct_path(start, goal)
            
            # Smooth path if requested
            if self.smoothing:
                path = self.smooth_path(path, environment)
            
            cost = self._calculate_path_cost(path)
            
            return PlanningResult(
                success=True,
                path=path,
                cost=cost,
                computation_time=computation_time,
                nodes_expanded=self._nodes_expanded,
                metadata={
                    "algorithm": "D*",
                    "allow_diagonal": self.allow_diagonal,
                    "smoothing": self.smoothing,
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
    
    def _initialize_graph(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> None:
        """Initialize the graph for D* search."""
        self.graph = {}
        self.open_list = []
        
        # Add all valid positions to graph
        height, width = environment.get_dimensions()
        
        for r in range(height):
            for c in range(width):
                if environment.is_valid_position((r, c)):
                    self.graph[(r, c)] = {
                        'g': float('inf'),
                        'rhs': float('inf'),
                        'parent': None,
                        'tag': 'NEW'  # NEW, OPEN, CLOSED
                    }
        
        # Initialize goal
        self.graph[goal]['rhs'] = 0.0
        heapq.heappush(self.open_list, (self._calculate_key(goal), goal))
    
    def _d_star_search(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> bool:
        """Run D* search algorithm."""
        while self.open_list:
            key, current = heapq.heappop(self.open_list)
            
            if self.graph[current]['tag'] == 'CLOSED':
                continue
            
            self.graph[current]['tag'] = 'CLOSED'
            self._increment_nodes_expanded()
            
            # Check if we reached the start
            if current == start:
                return True
            
            # Update neighbors
            neighbors = self.get_neighbors(current, environment, self.allow_diagonal)
            
            for neighbor in neighbors:
                if neighbor not in self.graph:
                    continue
                
                # Calculate new rhs value
                new_rhs = self.graph[current]['g'] + self._get_move_cost(current, neighbor)
                
                if new_rhs < self.graph[neighbor]['rhs']:
                    self.graph[neighbor]['rhs'] = new_rhs
                    self.graph[neighbor]['parent'] = current
                    
                    # Add to open list if not already there
                    if self.graph[neighbor]['tag'] != 'OPEN':
                        heapq.heappush(self.open_list, (self._calculate_key(neighbor), neighbor))
                        self.graph[neighbor]['tag'] = 'OPEN'
        
        return False
    
    def _calculate_key(self, node: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate D* key for a node."""
        g = self.graph[node]['g']
        rhs = self.graph[node]['rhs']
        
        return (min(g, rhs), min(g, rhs))
    
    def _reconstruct_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from start to goal."""
        path = [start]
        current = start
        
        while current != goal and self.graph[current]['parent'] is not None:
            current = self.graph[current]['parent']
            path.append(current)
        
        return path
    
    def _get_move_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int]
    ) -> float:
        """Calculate move cost between positions."""
        r1, c1 = from_pos
        r2, c2 = to_pos
        
        # Diagonal moves cost sqrt(2), straight moves cost 1
        if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
            return np.sqrt(2)
        else:
            return 1.0
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total path cost."""
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._get_move_cost(path[i], path[i + 1])
        
        return total_cost
