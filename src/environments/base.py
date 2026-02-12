"""
Base environment class for path planning.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import numpy as np


class BaseEnvironment(ABC):
    """Base class for all path planning environments."""
    
    @abstractmethod
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (not occupied by obstacles).
        
        Args:
            position: Position to check (row, col)
            
        Returns:
            True if position is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def is_line_of_sight(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> bool:
        """
        Check if there's a clear line of sight between two positions.
        
        Args:
            start: Starting position
            end: Ending position
            
        Returns:
            True if line of sight is clear, False otherwise
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of the environment.
        
        Returns:
            (height, width) tuple
        """
        pass
    
    @abstractmethod
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
        pass
    
    def get_random_valid_position(self) -> Tuple[int, int]:
        """
        Get a random valid position in the environment.
        
        Returns:
            Random valid position
        """
        height, width = self.get_dimensions()
        
        for _ in range(1000):  # Try up to 1000 times
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            if self.is_valid_position((row, col)):
                return (row, col)
        
        raise RuntimeError("Could not find a valid position")
    
    def calculate_clearance(self, position: Tuple[int, int]) -> float:
        """
        Calculate the minimum distance to obstacles from a position.
        
        Args:
            position: Position to check
            
        Returns:
            Minimum distance to obstacles
        """
        if not self.is_valid_position(position):
            return 0.0
        
        height, width = self.get_dimensions()
        min_distance = float('inf')
        
        # Check in a small radius around the position
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                if dr == 0 and dc == 0:
                    continue
                
                r, c = position[0] + dr, position[1] + dc
                if 0 <= r < height and 0 <= c < width:
                    if not self.is_valid_position((r, c)):
                        distance = np.sqrt(dr**2 + dc**2)
                        min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 5.0
