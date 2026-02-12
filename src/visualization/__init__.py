"""
Visualization utilities for path planning.
"""

from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


def plot_path_comparison(
    environment: "BaseEnvironment",
    results: Dict[str, Any],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of multiple planning results.
    
    Args:
        environment: The environment
        results: Dictionary of planner results
        start: Start position
        goal: Goal position
        save_path: Optional path to save plot
    """
    num_planners = len(results)
    fig, axes = plt.subplots(1, num_planners, figsize=(5*num_planners, 5))
    
    if num_planners == 1:
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_metrics(
    results: Dict[str, Any],
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot performance metrics comparison.
    
    Args:
        results: Dictionary of planner results
        metrics: List of metrics to plot
        save_path: Optional path to save plot
    """
    if metrics is None:
        metrics = ['computation_time', 'nodes_expanded', 'path_length']
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        planners = list(results.keys())
        values = [results[p][metric] for p in planners]
        
        bars = ax.bar(planners, values)
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_interactive_plot(
    environment: "BaseEnvironment",
    results: Dict[str, Any],
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> go.Figure:
    """
    Create interactive Plotly visualization.
    
    Args:
        environment: The environment
        results: Dictionary of planner results
        start: Start position
        goal: Goal position
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add obstacles
    obstacles = environment.get_obstacles()
    if obstacles:
        obs_x = [pos[1] for pos in obstacles]
        obs_y = [pos[0] for pos in obstacles]
        
        fig.add_trace(go.Scatter(
            x=obs_x, y=obs_y,
            mode='markers',
            marker=dict(color='black', size=10),
            name='Obstacles',
            showlegend=True
        ))
    
    # Add start and goal
    fig.add_trace(go.Scatter(
        x=[start[1]], y=[start[0]],
        mode='markers',
        marker=dict(color='green', size=15, symbol='star'),
        name='Start',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[goal[1]], y=[goal[0]],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name='Goal',
        showlegend=True
    ))
    
    # Add paths
    colors = ['blue', 'orange', 'purple', 'brown', 'pink']
    for i, (planner_name, result) in enumerate(results.items()):
        if result['success'] and result['path']:
            path_x = [pos[1] for pos in result['path']]
            path_y = [pos[0] for pos in result['path']]
            
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=4),
                name=f"{planner_name} Path",
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title="Path Planning Results",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    
    return fig


def plot_benchmark_results(
    benchmark_data: "pd.DataFrame",
    save_path: Optional[str] = None
) -> None:
    """
    Plot benchmark results.
    
    Args:
        benchmark_data: DataFrame with benchmark results
        save_path: Optional path to save plot
    """
    metrics = ['success_rate', 'avg_computation_time', 'avg_path_length']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group by algorithm
        grouped = benchmark_data.groupby('algorithm')[metric].mean().sort_values(ascending=False)
        
        bars = ax.bar(grouped.index, grouped.values)
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, grouped.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_leaderboard_plot(
    benchmark_data: "pd.DataFrame",
    metric: str = 'success_rate',
    save_path: Optional[str] = None
) -> None:
    """
    Create leaderboard plot.
    
    Args:
        benchmark_data: DataFrame with benchmark results
        metric: Metric to rank by
        save_path: Optional path to save plot
    """
    # Group by algorithm and calculate mean
    leaderboard = benchmark_data.groupby('algorithm')[metric].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(leaderboard.index, leaderboard.values)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f"Algorithm Leaderboard - {metric.replace('_', ' ').title()}")
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, leaderboard.values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
               f'{value:.3f}', ha='left', va='center')
        
        # Add rank number
        ax.text(-0.01, bar.get_y() + bar.get_height()/2.,
               f'#{i+1}', ha='right', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_visualization(
    fig: plt.Figure,
    filename: str,
    directory: str = "assets"
) -> str:
    """
    Save visualization to file.
    
    Args:
        fig: Matplotlib figure
        filename: Filename to save
        directory: Directory to save in
        
    Returns:
        Path to saved file
    """
    Path(directory).mkdir(exist_ok=True)
    filepath = Path(directory) / filename
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return str(filepath)
