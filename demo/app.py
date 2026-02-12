"""
Interactive Streamlit demo for path planning algorithms.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict, Any
import time

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.planners import AStarPlanner, RRTStarPlanner, DijkstraPlanner
from src.environments import GridEnvironment


def create_sample_environments() -> Dict[str, GridEnvironment]:
    """Create sample environments for demonstration."""
    environments = {}
    
    # Simple grid
    grid = np.zeros((20, 20))
    grid[5:15, 10] = 1  # Vertical wall
    environments["Simple Grid"] = GridEnvironment(grid)
    
    # Maze
    environments["Maze"] = GridEnvironment.create_maze(21, 21, seed=42)
    
    # Random obstacles
    environments["Random Obstacles"] = GridEnvironment.create_random(
        20, 20, obstacle_ratio=0.3, seed=42
    )
    
    return environments


def run_planner_comparison(
    environment: GridEnvironment,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    planners: List[str]
) -> Dict[str, Any]:
    """Run multiple planners and compare results."""
    results = {}
    
    planner_classes = {
        "A*": AStarPlanner(),
        "RRT*": RRTStarPlanner(max_iterations=500),
        "Dijkstra": DijkstraPlanner()
    }
    
    for planner_name in planners:
        if planner_name in planner_classes:
            planner = planner_classes[planner_name]
            
            start_time = time.time()
            result = planner.plan(environment, start, goal)
            end_time = time.time()
            
            results[planner_name] = {
                'success': result.success,
                'path': result.path,
                'cost': result.cost,
                'computation_time': result.computation_time,
                'nodes_expanded': result.nodes_expanded,
                'path_length': len(result.path) if result.path else 0
            }
    
    return results


def visualize_results(
    environment: GridEnvironment,
    results: Dict[str, Any],
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> None:
    """Visualize planning results."""
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    
    if len(results) == 1:
        axes = [axes]
    
    for i, (planner_name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Create visualization grid
        vis_grid = environment.grid.copy()
        
        # Mark start and goal
        vis_grid[start] = 2
        vis_grid[goal] = 3
        
        # Mark path
        if result['success'] and result['path']:
            for pos in result['path'][1:-1]:  # Skip start and goal
                vis_grid[pos] = 4
        
        # Plot
        im = ax.imshow(vis_grid, cmap='tab10', interpolation='nearest')
        ax.set_title(f"{planner_name}\n"
                    f"Success: {result['success']}\n"
                    f"Time: {result['computation_time']:.3f}s\n"
                    f"Nodes: {result['nodes_expanded']}\n"
                    f"Length: {result['path_length']}")
        ax.invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Path Planning Algorithms Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Path Planning Algorithms Demo")
    st.markdown("""
    This interactive demo showcases different path planning algorithms
    for robotics applications. **This is for educational purposes only.**
    """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Environment selection
    environments = create_sample_environments()
    env_name = st.sidebar.selectbox(
        "Select Environment",
        list(environments.keys())
    )
    environment = environments[env_name]
    
    # Start and goal positions
    st.sidebar.subheader("Start and Goal Positions")
    
    height, width = environment.get_dimensions()
    
    start_col, start_row = st.sidebar.columns(2)
    with start_col:
        start_x = st.number_input("Start X", 0, width-1, 0)
    with start_row:
        start_y = st.number_input("Start Y", 0, height-1, 0)
    
    goal_col, goal_row = st.sidebar.columns(2)
    with goal_col:
        goal_x = st.number_input("Goal X", 0, width-1, width-1)
    with goal_row:
        goal_y = st.number_input("Goal Y", 0, height-1, height-1)
    
    start = (start_y, start_x)
    goal = (goal_y, goal_x)
    
    # Planner selection
    st.sidebar.subheader("Algorithms")
    selected_planners = st.sidebar.multiselect(
        "Select Planners",
        ["A*", "RRT*", "Dijkstra"],
        default=["A*"]
    )
    
    # Validate positions
    if not environment.is_valid_position(start):
        st.error(f"Start position {start} is invalid (obstacle or out of bounds)")
        return
    
    if not environment.is_valid_position(goal):
        st.error(f"Goal position {goal} is invalid (obstacle or out of bounds)")
        return
    
    if start == goal:
        st.error("Start and goal positions cannot be the same")
        return
    
    # Run planning
    if st.sidebar.button("Plan Path", type="primary"):
        with st.spinner("Planning paths..."):
            results = run_planner_comparison(environment, start, goal, selected_planners)
        
        # Display results
        st.subheader("Planning Results")
        
        # Create comparison table
        if results:
            comparison_data = []
            for planner_name, result in results.items():
                comparison_data.append({
                    'Algorithm': planner_name,
                    'Success': '✅' if result['success'] else '❌',
                    'Path Length': result['path_length'],
                    'Computation Time (s)': f"{result['computation_time']:.3f}",
                    'Nodes Expanded': result['nodes_expanded'],
                    'Path Cost': f"{result['cost']:.2f}" if result['success'] else "∞"
                })
            
            st.table(comparison_data)
        
        # Visualize results
        st.subheader("Visualization")
        visualize_results(environment, results, start, goal)
        
        # Performance metrics
        if len(results) > 1:
            st.subheader("Performance Comparison")
            
            # Create performance charts
            metrics = ['computation_time', 'nodes_expanded', 'path_length']
            metric_names = ['Computation Time (s)', 'Nodes Expanded', 'Path Length']
            
            for metric, metric_name in zip(metrics, metric_names):
                fig = go.Figure()
                
                planners = list(results.keys())
                values = [results[p][metric] for p in planners]
                
                fig.add_trace(go.Bar(
                    x=planners,
                    y=values,
                    text=values,
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f"{metric_name} Comparison",
                    xaxis_title="Algorithm",
                    yaxis_title=metric_name,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Environment visualization
    st.subheader("Environment")
    
    # Show environment with start/goal
    fig, ax = plt.subplots(figsize=(8, 8))
    
    vis_grid = environment.grid.copy()
    vis_grid[start] = 2
    vis_grid[goal] = 3
    
    im = ax.imshow(vis_grid, cmap='tab10', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(range(4))
    cbar.set_ticklabels(['Free', 'Obstacle', 'Start', 'Goal'])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"{env_name} Environment")
    ax.invert_yaxis()
    
    st.pyplot(fig)
    
    # Information section
    st.subheader("Algorithm Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **A* Algorithm**
        - Optimal pathfinding
        - Uses heuristic to guide search
        - Guarantees shortest path
        - Good for grid-based environments
        """)
    
    with col2:
        st.markdown("""
        **RRT* Algorithm**
        - Sampling-based planning
        - Asymptotically optimal
        - Good for complex environments
        - Can handle dynamic obstacles
        """)
    
    with col3:
        st.markdown("""
        **Dijkstra Algorithm**
        - Optimal shortest path
        - Explores all possible paths
        - No heuristic guidance
        - Guaranteed optimality
        """)
    
    # Safety disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ SAFETY DISCLAIMER**
    
    This software is for research and educational purposes only. 
    Do not use on real robots without proper safety review and testing.
    """)

if __name__ == "__main__":
    main()
