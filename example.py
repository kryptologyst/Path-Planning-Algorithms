#!/usr/bin/env python3
"""
Example script demonstrating path planning algorithms.

This script shows how to use the path planning algorithms library
to solve pathfinding problems in different environments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from src.planners import AStarPlanner, RRTStarPlanner, DijkstraPlanner
from src.environments import GridEnvironment
from src.evaluation import ComprehensiveEvaluator
from src.robots import DifferentialDriveRobot, RobotSimulator


def create_sample_environments():
    """Create sample environments for demonstration."""
    environments = {}
    
    # Simple grid with obstacles
    grid1 = np.zeros((20, 20))
    grid1[5:15, 10] = 1  # Vertical wall
    environments["Simple Grid"] = GridEnvironment(grid1)
    
    # Maze environment
    environments["Maze"] = GridEnvironment.create_maze(21, 21, seed=42)
    
    # Random obstacles
    environments["Random Obstacles"] = GridEnvironment.create_random(
        20, 20, obstacle_ratio=0.3, seed=42
    )
    
    return environments


def demonstrate_planners():
    """Demonstrate different path planning algorithms."""
    print("🤖 Path Planning Algorithms Demonstration")
    print("=" * 50)
    
    # Create environments
    environments = create_sample_environments()
    
    # Create planners
    planners = {
        "A*": AStarPlanner(heuristic="euclidean", allow_diagonal=True),
        "RRT*": RRTStarPlanner(max_iterations=500, step_size=1.0),
        "Dijkstra": DijkstraPlanner(allow_diagonal=True)
    }
    
    # Test on each environment
    for env_name, environment in environments.items():
        print(f"\n📍 Testing on {env_name}")
        print("-" * 30)
        
        # Define start and goal
        start = (0, 0)
        goal = (19, 19)
        
        # Ensure positions are valid
        if not environment.is_valid_position(start):
            start = environment.get_random_valid_position()
        if not environment.is_valid_position(goal):
            goal = environment.get_random_valid_position()
        
        print(f"Start: {start}, Goal: {goal}")
        
        # Test each planner
        results = {}
        for planner_name, planner in planners.items():
            print(f"\n🔍 Testing {planner_name}...")
            
            result = planner.plan(environment, start, goal)
            
            if result.success:
                print(f"  ✅ Success!")
                print(f"  📏 Path length: {len(result.path)}")
                print(f"  ⏱️  Computation time: {result.computation_time:.3f}s")
                print(f"  🔢 Nodes expanded: {result.nodes_expanded}")
                print(f"  💰 Path cost: {result.cost:.2f}")
                
                results[planner_name] = {
                    'success': True,
                    'path': result.path,
                    'computation_time': result.computation_time,
                    'nodes_expanded': result.nodes_expanded,
                    'cost': result.cost
                }
            else:
                print(f"  ❌ Failed: {result.metadata.get('error', 'Unknown error')}")
                results[planner_name] = {
                    'success': False,
                    'path': [],
                    'computation_time': result.computation_time,
                    'nodes_expanded': result.nodes_expanded,
                    'cost': float('inf')
                }
        
        # Visualize results
        visualize_results(environment, results, start, goal, env_name)


def visualize_results(environment, results, start, goal, env_name):
    """Visualize planning results."""
    num_planners = len([r for r in results.values() if r['success']])
    
    if num_planners == 0:
        print("  No successful planners to visualize")
        return
    
    fig, axes = plt.subplots(1, num_planners, figsize=(5*num_planners, 5))
    if num_planners == 1:
        axes = [axes]
    
    ax_idx = 0
    for planner_name, result in results.items():
        if not result['success']:
            continue
        
        ax = axes[ax_idx]
        
        # Create visualization grid
        vis_grid = environment.grid.copy()
        
        # Mark start and goal
        vis_grid[start] = 2
        vis_grid[goal] = 3
        
        # Mark path
        if result['path']:
            for pos in result['path'][1:-1]:  # Skip start and goal
                vis_grid[pos] = 4
        
        # Plot
        im = ax.imshow(vis_grid, cmap='tab10', interpolation='nearest')
        ax.set_title(f"{planner_name}\n"
                    f"Time: {result['computation_time']:.3f}s\n"
                    f"Nodes: {result['nodes_expanded']}\n"
                    f"Cost: {result['cost']:.2f}")
        ax.invert_yaxis()
        
        ax_idx += 1
    
    plt.suptitle(f"Path Planning Results - {env_name}")
    plt.tight_layout()
    
    # Save plot
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    plt.savefig(assets_dir / f"{env_name.lower().replace(' ', '_')}_results.png", 
                dpi=300, bbox_inches='tight')
    
    plt.show()


def demonstrate_robot_simulation():
    """Demonstrate robot simulation."""
    print("\n🤖 Robot Simulation Demonstration")
    print("=" * 40)
    
    # Create robot and simulator
    robot = DifferentialDriveRobot(width=0.5, length=0.8)
    simulator = RobotSimulator(robot)
    
    # Create simple environment
    grid = np.zeros((10, 10))
    grid[4, 4:7] = 1  # Add obstacles
    environment = GridEnvironment(grid)
    
    # Plan path
    planner = AStarPlanner()
    start = (0, 0)
    goal = (9, 9)
    
    result = planner.plan(environment, start, goal)
    
    if result.success:
        print(f"✅ Path found with {len(result.path)} waypoints")
        
        # Simulate robot following path
        simulator.reset((0.0, 0.0), 0.0)
        trajectory = simulator.follow_path(result.path, linear_velocity=1.0)
        
        print(f"🤖 Robot followed path with {len(trajectory)} trajectory points")
        
        # Visualize trajectory
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot environment
        ax.imshow(environment.grid, cmap='gray', interpolation='nearest')
        
        # Plot path
        path_x = [pos[1] for pos in result.path]
        path_y = [pos[0] for pos in result.path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        # Plot trajectory
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        ax.plot(traj_x, traj_y, 'r--', linewidth=1, label='Robot Trajectory')
        
        ax.set_title("Robot Path Following Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig("assets/robot_simulation.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("❌ No path found for robot simulation")


def run_benchmark():
    """Run comprehensive benchmark."""
    print("\n📊 Running Comprehensive Benchmark")
    print("=" * 40)
    
    # Create environments
    environments = create_sample_environments()
    
    # Create planners
    planners = [
        AStarPlanner(heuristic="euclidean", allow_diagonal=True),
        AStarPlanner(heuristic="manhattan", allow_diagonal=False),
        RRTStarPlanner(max_iterations=300, step_size=1.0),
        DijkstraPlanner(allow_diagonal=True)
    ]
    
    # Run benchmark
    from src.evaluation.comprehensive import run_comprehensive_benchmark
    
    results_df = run_comprehensive_benchmark(
        planners=planners,
        environments=list(environments.values()),
        num_trials_per_env=20,
        save_results=True
    )
    
    # Create leaderboard
    from src.evaluation.comprehensive import create_leaderboard, generate_benchmark_report
    
    print("\n🏆 Success Rate Leaderboard:")
    success_leaderboard = create_leaderboard(results_df, 'success_rate')
    print(success_leaderboard.to_string(index=False))
    
    print("\n⚡ Computation Time Leaderboard:")
    time_leaderboard = create_leaderboard(results_df, 'avg_computation_time')
    print(time_leaderboard.to_string(index=False))
    
    # Generate report
    report = generate_benchmark_report(results_df, "assets/benchmark_report.md")
    print(f"\n📄 Benchmark report generated")


def main():
    """Main demonstration function."""
    print("🚀 Path Planning Algorithms - Comprehensive Demo")
    print("=" * 60)
    print("⚠️  This is for educational purposes only!")
    print("⚠️  Do not use on real robots without proper safety review!")
    print("")
    
    try:
        # Demonstrate planners
        demonstrate_planners()
        
        # Demonstrate robot simulation
        demonstrate_robot_simulation()
        
        # Run benchmark
        run_benchmark()
        
        print("\n✅ Demo completed successfully!")
        print("📁 Check the 'assets/' directory for saved results and visualizations")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
