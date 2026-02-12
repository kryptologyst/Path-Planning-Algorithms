"""
RRT* (Rapidly-exploring Random Trees Star) path planning algorithm.

RRT* is an asymptotically optimal sampling-based path planning algorithm
that incrementally builds a tree of feasible paths while rewiring to
improve path quality.
"""

import random
import math
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np

from .base import BasePlanner, PlanningResult


class RRTStarPlanner(BasePlanner):
    """
    RRT* path planning algorithm.
    
    RRT* builds a tree incrementally by sampling random points and
    connecting them to the nearest node in the tree, then rewiring
    nearby nodes to improve path quality.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        step_size: float = 1.0,
        goal_threshold: float = 1.0,
        rewire_radius: float = 2.0,
        goal_sample_rate: float = 0.1,
        **kwargs: Any
    ) -> None:
        """
        Initialize RRT* planner.
        
        Args:
            max_iterations: Maximum number of iterations
            step_size: Step size for tree expansion
            goal_threshold: Distance threshold to consider goal reached
            rewire_radius: Radius for rewiring connections
            goal_sample_rate: Probability of sampling goal instead of random point
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.rewire_radius = rewire_radius
        self.goal_sample_rate = goal_sample_rate
    
    def plan(
        self,
        environment: "BaseEnvironment",
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs: Any
    ) -> PlanningResult:
        """
        Plan a path using RRT* algorithm.
        
        Args:
            environment: The environment to plan in
            start: Starting position
            goal: Goal position
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
        
        # Initialize tree
        tree: Dict[Tuple[int, int], Dict[str, Any]] = {
            start: {
                'parent': None,
                'cost': 0.0,
                'children': set()
            }
        }
        
        best_path = []
        best_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                sample_point = goal
            else:
                sample_point = self._sample_random_point(environment)
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(sample_point, tree)
            
            # Extend tree towards sample point
            new_node = self._extend_tree(nearest_node, sample_point, environment)
            
            if new_node is None:
                continue
            
            self._increment_nodes_expanded()
            
            # Add new node to tree
            tree[new_node] = {
                'parent': nearest_node,
                'cost': tree[nearest_node]['cost'] + self._distance(nearest_node, new_node),
                'children': set()
            }
            
            # Add to parent's children
            tree[nearest_node]['children'].add(new_node)
            
            # Rewire nearby nodes
            self._rewire_tree(new_node, tree, environment)
            
            # Check if goal is reached
            if self._distance(new_node, goal) <= self.goal_threshold:
                # Try to connect to goal
                if environment.is_line_of_sight(new_node, goal):
                    goal_cost = tree[new_node]['cost'] + self._distance(new_node, goal)
                    
                    if goal_cost < best_cost:
                        best_cost = goal_cost
                        best_path = self._reconstruct_path(tree, new_node, goal)
        
        computation_time = self._end_timing()
        
        if best_path:
            return PlanningResult(
                success=True,
                path=best_path,
                cost=best_cost,
                computation_time=computation_time,
                nodes_expanded=self._nodes_expanded,
                metadata={
                    "algorithm": "RRT*",
                    "iterations": iteration + 1,
                    "step_size": self.step_size,
                    "rewire_radius": self.rewire_radius,
                    "goal_sample_rate": self.goal_sample_rate,
                    "path_length": len(best_path)
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
    
    def _sample_random_point(self, environment: "BaseEnvironment") -> Tuple[int, int]:
        """
        Sample a random valid point in the environment.
        
        Args:
            environment: The environment
            
        Returns:
            Random valid point
        """
        height, width = environment.get_dimensions()
        
        for _ in range(100):  # Try up to 100 times
            row = random.randint(0, height - 1)
            col = random.randint(0, width - 1)
            if environment.is_valid_position((row, col)):
                return (row, col)
        
        # Fallback to environment method
        return environment.get_random_valid_position()
    
    def _find_nearest_node(
        self,
        point: Tuple[int, int],
        tree: Dict[Tuple[int, int], Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Find the nearest node in the tree to the given point.
        
        Args:
            point: Point to find nearest node to
            tree: Current tree
            
        Returns:
            Nearest node
        """
        min_distance = float('inf')
        nearest_node = None
        
        for node in tree.keys():
            distance = self._distance(node, point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _extend_tree(
        self,
        from_node: Tuple[int, int],
        to_point: Tuple[int, int],
        environment: "BaseEnvironment"
    ) -> Optional[Tuple[int, int]]:
        """
        Extend the tree from a node towards a point.
        
        Args:
            from_node: Node to extend from
            to_point: Point to extend towards
            environment: The environment
            
        Returns:
            New node if extension successful, None otherwise
        """
        distance = self._distance(from_node, to_point)
        
        if distance <= self.step_size:
            # Can reach the point directly
            if environment.is_line_of_sight(from_node, to_point):
                return to_point
        else:
            # Step towards the point
            direction = (
                (to_point[0] - from_node[0]) / distance,
                (to_point[1] - from_node[1]) / distance
            )
            
            new_point = (
                int(from_node[0] + direction[0] * self.step_size),
                int(from_node[1] + direction[1] * self.step_size)
            )
            
            if environment.is_valid_position(new_point) and \
               environment.is_line_of_sight(from_node, new_point):
                return new_point
        
        return None
    
    def _rewire_tree(
        self,
        new_node: Tuple[int, int],
        tree: Dict[Tuple[int, int], Dict[str, Any]],
        environment: "BaseEnvironment"
    ) -> None:
        """
        Rewire nearby nodes to improve path quality.
        
        Args:
            new_node: Newly added node
            tree: Current tree
            environment: The environment
        """
        nearby_nodes = self._find_nearby_nodes(new_node, tree)
        
        for nearby_node in nearby_nodes:
            if nearby_node == new_node:
                continue
            
            # Check if we can improve the path to nearby_node
            new_cost = tree[new_node]['cost'] + self._distance(new_node, nearby_node)
            
            if new_cost < tree[nearby_node]['cost']:
                # Check if path is collision-free
                if environment.is_line_of_sight(new_node, nearby_node):
                    # Remove from old parent's children
                    old_parent = tree[nearby_node]['parent']
                    if old_parent is not None:
                        tree[old_parent]['children'].discard(nearby_node)
                    
                    # Update parent and cost
                    tree[nearby_node]['parent'] = new_node
                    tree[nearby_node]['cost'] = new_cost
                    
                    # Add to new parent's children
                    tree[new_node]['children'].add(nearby_node)
    
    def _find_nearby_nodes(
        self,
        node: Tuple[int, int],
        tree: Dict[Tuple[int, int], Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """
        Find nodes within rewire radius of the given node.
        
        Args:
            node: Node to find nearby nodes for
            tree: Current tree
            
        Returns:
            List of nearby nodes
        """
        nearby_nodes = []
        
        for other_node in tree.keys():
            if self._distance(node, other_node) <= self.rewire_radius:
                nearby_nodes.append(other_node)
        
        return nearby_nodes
    
    def _reconstruct_path(
        self,
        tree: Dict[Tuple[int, int], Dict[str, Any]],
        node: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct path from start to goal through the tree.
        
        Args:
            tree: Current tree
            node: Node to reconstruct path from
            goal: Goal position
            
        Returns:
            Path from start to goal
        """
        path = [goal]
        current = node
        
        while current is not None:
            path.append(current)
            current = tree[current]['parent']
        
        path.reverse()
        return path
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
