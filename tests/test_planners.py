"""
Test suite for path planning algorithms.
"""

import pytest
import numpy as np
from typing import Tuple

from src.planners import AStarPlanner, RRTStarPlanner, DijkstraPlanner
from src.environments import GridEnvironment


@pytest.fixture
def simple_grid():
    """Create a simple test grid."""
    grid = np.zeros((10, 10))
    grid[4, 4:7] = 1  # Add obstacles
    return GridEnvironment(grid)


@pytest.fixture
def start_goal():
    """Standard start and goal positions."""
    return (0, 0), (9, 9)


class TestAStarPlanner:
    """Test cases for A* planner."""
    
    def test_a_star_success(self, simple_grid, start_goal):
        """Test successful path planning."""
        start, goal = start_goal
        planner = AStarPlanner()
        
        result = planner.plan(simple_grid, start, goal)
        
        assert result.success
        assert len(result.path) > 0
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.cost > 0
        assert result.computation_time > 0
        assert result.nodes_expanded > 0
    
    def test_a_star_no_path(self, simple_grid):
        """Test case where no path exists."""
        # Create impossible scenario
        grid = np.ones((5, 5))
        grid[0, 0] = 0  # Only start is free
        grid[4, 4] = 0  # Only goal is free
        environment = GridEnvironment(grid)
        
        planner = AStarPlanner()
        result = planner.plan(environment, (0, 0), (4, 4))
        
        assert not result.success
        assert len(result.path) == 0
        assert result.cost == float('inf')
    
    def test_a_star_invalid_start(self, simple_grid):
        """Test with invalid start position."""
        planner = AStarPlanner()
        result = planner.plan(simple_grid, (5, 5), (9, 9))  # Start in obstacle
        
        assert not result.success
        assert "Invalid start position" in result.metadata["error"]
    
    def test_a_star_invalid_goal(self, simple_grid):
        """Test with invalid goal position."""
        planner = AStarPlanner()
        result = planner.plan(simple_grid, (0, 0), (5, 5))  # Goal in obstacle
        
        assert not result.success
        assert "Invalid goal position" in result.metadata["error"]
    
    def test_a_star_heuristics(self, simple_grid, start_goal):
        """Test different heuristic functions."""
        start, goal = start_goal
        
        heuristics = ["euclidean", "manhattan", "diagonal"]
        results = []
        
        for heuristic in heuristics:
            planner = AStarPlanner(heuristic=heuristic)
            result = planner.plan(simple_grid, start, goal)
            results.append(result)
        
        # All should succeed
        for result in results:
            assert result.success
        
        # All should find paths
        for result in results:
            assert len(result.path) > 0


class TestRRTStarPlanner:
    """Test cases for RRT* planner."""
    
    def test_rrt_star_success(self, simple_grid, start_goal):
        """Test successful path planning."""
        start, goal = start_goal
        planner = RRTStarPlanner(max_iterations=100)
        
        result = planner.plan(simple_grid, start, goal)
        
        # RRT* might not always find a path in limited iterations
        if result.success:
            assert len(result.path) > 0
            assert result.path[0] == start
            assert result.cost > 0
            assert result.computation_time > 0
            assert result.nodes_expanded > 0
    
    def test_rrt_star_parameters(self, simple_grid, start_goal):
        """Test different RRT* parameters."""
        start, goal = start_goal
        
        # Test with different parameters
        planner = RRTStarPlanner(
            max_iterations=50,
            step_size=0.5,
            goal_threshold=2.0,
            rewire_radius=1.5,
            goal_sample_rate=0.2
        )
        
        result = planner.plan(simple_grid, start, goal)
        
        # Should not crash
        assert result.computation_time >= 0
        assert result.nodes_expanded >= 0


class TestDijkstraPlanner:
    """Test cases for Dijkstra planner."""
    
    def test_dijkstra_success(self, simple_grid, start_goal):
        """Test successful path planning."""
        start, goal = start_goal
        planner = DijkstraPlanner()
        
        result = planner.plan(simple_grid, start, goal)
        
        assert result.success
        assert len(result.path) > 0
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.cost > 0
        assert result.computation_time > 0
        assert result.nodes_expanded > 0
    
    def test_dijkstra_optimality(self, simple_grid, start_goal):
        """Test that Dijkstra finds optimal path."""
        start, goal = start_goal
        
        # Compare with A*
        dijkstra_planner = DijkstraPlanner()
        astar_planner = AStarPlanner()
        
        dijkstra_result = dijkstra_planner.plan(simple_grid, start, goal)
        astar_result = astar_planner.plan(simple_grid, start, goal)
        
        assert dijkstra_result.success
        assert astar_result.success
        
        # Dijkstra should find optimal path (same or better than A*)
        assert dijkstra_result.cost <= astar_result.cost


class TestGridEnvironment:
    """Test cases for grid environment."""
    
    def test_grid_creation(self):
        """Test grid environment creation."""
        grid = np.zeros((5, 5))
        environment = GridEnvironment(grid)
        
        assert environment.get_dimensions() == (5, 5)
        assert environment.is_valid_position((0, 0))
        assert not environment.is_valid_position((5, 5))  # Out of bounds
    
    def test_obstacle_placement(self):
        """Test obstacle placement and removal."""
        grid = np.zeros((5, 5))
        environment = GridEnvironment(grid)
        
        # Add obstacle
        environment.add_obstacle((2, 2))
        assert not environment.is_valid_position((2, 2))
        
        # Remove obstacle
        environment.remove_obstacle((2, 2))
        assert environment.is_valid_position((2, 2))
    
    def test_line_of_sight(self):
        """Test line of sight calculation."""
        grid = np.zeros((5, 5))
        grid[2, 1:4] = 1  # Horizontal wall
        environment = GridEnvironment(grid)
        
        # Should have line of sight
        assert environment.is_line_of_sight((0, 0), (4, 0))
        
        # Should not have line of sight
        assert not environment.is_line_of_sight((0, 0), (4, 4))
    
    def test_random_environment(self):
        """Test random environment creation."""
        environment = GridEnvironment.create_random(10, 10, seed=42)
        
        assert environment.get_dimensions() == (10, 10)
        
        # Should have some obstacles
        obstacles = environment.get_obstacles()
        assert len(obstacles) > 0
    
    def test_maze_environment(self):
        """Test maze environment creation."""
        environment = GridEnvironment.create_maze(11, 11, seed=42)
        
        assert environment.get_dimensions() == (11, 11)
        
        # Should be able to find path from corner to corner
        start, goal = (0, 0), (10, 10)
        if environment.is_valid_position(start) and environment.is_valid_position(goal):
            planner = AStarPlanner()
            result = planner.plan(environment, start, goal)
            # Might succeed or fail depending on maze structure
            assert result.computation_time >= 0


class TestIntegration:
    """Integration tests."""
    
    def test_planner_comparison(self, simple_grid, start_goal):
        """Test comparing multiple planners."""
        start, goal = start_goal
        
        planners = [
            AStarPlanner(),
            DijkstraPlanner(),
            RRTStarPlanner(max_iterations=100)
        ]
        
        results = []
        for planner in planners:
            result = planner.plan(simple_grid, start, goal)
            results.append(result)
        
        # At least A* and Dijkstra should succeed
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 2
        
        # All successful results should have valid paths
        for result in successful_results:
            assert len(result.path) > 0
            assert result.path[0] == start
            assert result.path[-1] == goal
    
    def test_different_environments(self, start_goal):
        """Test planners on different environment types."""
        start, goal = start_goal
        
        # Create different environments
        environments = {
            "simple": GridEnvironment.create_random(10, 10, obstacle_ratio=0.1, seed=42),
            "complex": GridEnvironment.create_random(10, 10, obstacle_ratio=0.4, seed=42),
            "maze": GridEnvironment.create_maze(11, 11, seed=42)
        }
        
        planner = AStarPlanner()
        
        for env_name, environment in environments.items():
            # Adjust start/goal if needed
            if not environment.is_valid_position(start):
                start = environment.get_random_valid_position()
            if not environment.is_valid_position(goal):
                goal = environment.get_random_valid_position()
            
            result = planner.plan(environment, start, goal)
            
            # Should not crash
            assert result.computation_time >= 0
            assert result.nodes_expanded >= 0
            
            if result.success:
                assert len(result.path) > 0
