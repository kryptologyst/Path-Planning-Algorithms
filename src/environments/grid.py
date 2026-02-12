"""
Grid-based environment for path planning.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .base import BaseEnvironment


class GridEnvironment(BaseEnvironment):
    """
    Grid-based environment for path planning.
    
    The environment is represented as a 2D grid where:
    - 0: Free space
    - 1: Obstacle
    - 2: Start position (optional)
    - 3: Goal position (optional)
    """
    
    def __init__(
        self,
        grid: np.ndarray,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Initialize grid environment.
        
        Args:
            grid: 2D numpy array representing the environment
            start: Optional start position
            goal: Optional goal position
        """
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start = start
        self.goal = goal
        
        # Validate grid
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError("Grid must be a 2D numpy array")
        
        if np.any(grid < 0) or np.any(grid > 3):
            raise ValueError("Grid values must be between 0 and 3")
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (not occupied by obstacles).
        
        Args:
            position: Position to check (row, col)
            
        Returns:
            True if position is valid, False otherwise
        """
        row, col = position
        
        # Check bounds
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        
        # Check if not obstacle
        return self.grid[row, col] != 1
    
    def is_line_of_sight(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> bool:
        """
        Check if there's a clear line of sight between two positions.
        
        Uses Bresenham's line algorithm to check all cells along the line.
        
        Args:
            start: Starting position
            end: Ending position
            
        Returns:
            True if line of sight is clear, False otherwise
        """
        x0, y0 = start[1], start[0]  # Convert to (x, y) coordinates
        x1, y1 = end[1], end[0]
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Check if current position is obstacle
            if not self.is_valid_position((y, x)):
                return False
            
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
            
            if error2 < dx:
                error += dx
                y += y_step
        
        return True
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of the environment.
        
        Returns:
            (height, width) tuple
        """
        return (self.height, self.width)
    
    def add_obstacle(self, position: Tuple[int, int]) -> None:
        """
        Add an obstacle at the specified position.
        
        Args:
            position: Position to add obstacle
        """
        row, col = position
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row, col] = 1
    
    def remove_obstacle(self, position: Tuple[int, int]) -> None:
        """
        Remove an obstacle from the specified position.
        
        Args:
            position: Position to remove obstacle
        """
        row, col = position
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row, col] = 0
    
    def add_rectangle_obstacle(
        self,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int]
    ) -> None:
        """
        Add a rectangular obstacle.
        
        Args:
            top_left: Top-left corner (row, col)
            bottom_right: Bottom-right corner (row, col)
        """
        r1, c1 = top_left
        r2, c2 = bottom_right
        
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = 1
    
    def get_obstacles(self) -> List[Tuple[int, int]]:
        """
        Get all obstacle positions.
        
        Returns:
            List of obstacle positions
        """
        obstacles = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 1:
                    obstacles.append((r, c))
        return obstacles
    
    def visualize_path(
        self,
        path: List[Tuple[int, int]],
        title: str = "Path Planning Result",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the environment and path.
        
        Args:
            path: Path to visualize
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create visualization grid
        vis_grid = self.grid.copy()
        
        # Mark path
        if path:
            for i, (r, c) in enumerate(path):
                if i == 0:
                    vis_grid[r, c] = 2  # Start
                elif i == len(path) - 1:
                    vis_grid[r, c] = 3  # Goal
                else:
                    vis_grid[r, c] = 4  # Path
        
        # Create colormap
        colors = ['white', 'black', 'green', 'red', 'blue']
        labels = ['Free', 'Obstacle', 'Start', 'Goal', 'Path']
        
        # Plot grid
        im = ax.imshow(vis_grid, cmap='tab10', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks(range(5))
        cbar.set_ticklabels(labels)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Set labels and title
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        
        # Invert y-axis to match matrix indexing
        ax.invert_yaxis()
        
        # Add path information
        if path:
            path_length = len(path)
            ax.text(0.02, 0.98, f'Path Length: {path_length}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @classmethod
    def create_random(
        cls,
        height: int,
        width: int,
        obstacle_ratio: float = 0.2,
        seed: Optional[int] = None
    ) -> "GridEnvironment":
        """
        Create a random grid environment.
        
        Args:
            height: Grid height
            width: Grid width
            obstacle_ratio: Ratio of obstacles to total cells
            seed: Random seed for reproducibility
            
        Returns:
            Random grid environment
        """
        if seed is not None:
            np.random.seed(seed)
        
        grid = np.zeros((height, width))
        total_cells = height * width
        num_obstacles = int(total_cells * obstacle_ratio)
        
        # Randomly place obstacles
        obstacle_positions = np.random.choice(
            total_cells, num_obstacles, replace=False
        )
        
        for pos in obstacle_positions:
            row = pos // width
            col = pos % width
            grid[row, col] = 1
        
        return cls(grid)
    
    @classmethod
    def create_maze(
        cls,
        height: int,
        width: int,
        complexity: float = 0.75,
        density: float = 0.75,
        seed: Optional[int] = None
    ) -> "GridEnvironment":
        """
        Create a maze-like environment.
        
        Args:
            height: Grid height (must be odd)
            width: Grid width (must be odd)
            complexity: Maze complexity (0-1)
            density: Maze density (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            Maze environment
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure dimensions are odd
        height = height if height % 2 == 1 else height + 1
        width = width if width % 2 == 1 else width + 1
        
        # Create maze using recursive backtracking
        grid = np.ones((height, width))
        
        # Start with all walls
        for r in range(0, height, 2):
            for c in range(0, width, 2):
                grid[r, c] = 0  # Create passages
        
        # Add complexity
        for _ in range(int(complexity * (height + width))):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            if grid[y, x] == 1:
                neighbors = []
                if x > 1:
                    neighbors.append((y, x - 2))
                if x < width - 2:
                    neighbors.append((y, x + 2))
                if y > 1:
                    neighbors.append((y - 2, x))
                if y < height - 2:
                    neighbors.append((y + 2, x))
                
                if neighbors:
                    ny, nx = np.random.choice(len(neighbors))
                    ny, nx = neighbors[ny]
                    
                    if grid[ny, nx] == 1:
                        grid[y, x] = 0
                        grid[ny, nx] = 0
                        grid[(y + ny) // 2, (x + nx) // 2] = 0
        
        # Add density
        for r in range(height):
            for c in range(width):
                if grid[r, c] == 1 and np.random.random() < density:
                    grid[r, c] = 0
        
        return cls(grid)
