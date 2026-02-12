"""
Base evaluation classes for path planning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import pandas as pd

from ..planners.base import BasePlanner, PlanningResult
from ..environments.base import BaseEnvironment


@dataclass
class EvaluationResult:
    """Result of evaluating a path planning algorithm."""
    
    algorithm_name: str
    success_rate: float
    average_path_length: float
    average_computation_time: float
    average_nodes_expanded: float
    average_path_cost: float
    average_clearance: float
    average_smoothness: float
    total_trials: int
    successful_trials: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'algorithm': self.algorithm_name,
            'success_rate': self.success_rate,
            'avg_path_length': self.average_path_length,
            'avg_computation_time': self.average_computation_time,
            'avg_nodes_expanded': self.average_nodes_expanded,
            'avg_path_cost': self.average_path_cost,
            'avg_clearance': self.average_clearance,
            'avg_smoothness': self.average_smoothness,
            'total_trials': self.total_trials,
            'successful_trials': self.successful_trials,
            **self.metadata
        }


class BaseEvaluator(ABC):
    """Base class for evaluating path planning algorithms."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the evaluator."""
        self.config = kwargs
    
    @abstractmethod
    def evaluate(
        self,
        planner: BasePlanner,
        environment: BaseEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Evaluate a planner on a single problem instance.
        
        Args:
            planner: The planner to evaluate
            environment: The environment
            start: Start position
            goal: Goal position
            **kwargs: Additional parameters
            
        Returns:
            Evaluation result
        """
        pass
    
    def benchmark(
        self,
        planners: List[BasePlanner],
        environments: List[BaseEnvironment],
        problem_instances: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Benchmark multiple planners on multiple problem instances.
        
        Args:
            planners: List of planners to benchmark
            environments: List of environments
            problem_instances: List of (start, goal) tuples
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        for planner in planners:
            for i, environment in enumerate(environments):
                start, goal = problem_instances[i]
                
                result = self.evaluate(planner, environment, start, goal, **kwargs)
                results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate the total length of a path.
        
        Args:
            path: List of positions
            
        Returns:
            Total path length
        """
        if len(path) <= 1:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            total_length += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        return total_length
    
    def calculate_path_clearance(
        self,
        path: List[Tuple[int, int]],
        environment: BaseEnvironment
    ) -> float:
        """
        Calculate the average clearance of a path.
        
        Args:
            path: List of positions
            environment: The environment
            
        Returns:
            Average clearance
        """
        if not path:
            return 0.0
        
        clearances = []
        for position in path:
            clearance = environment.calculate_clearance(position)
            clearances.append(clearance)
        
        return np.mean(clearances)
    
    def calculate_path_smoothness(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate the smoothness of a path (inverse of curvature).
        
        Args:
            path: List of positions
            
        Returns:
            Path smoothness metric
        """
        if len(path) < 3:
            return 1.0
        
        total_curvature = 0.0
        
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            
            # Calculate angle between vectors
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                total_curvature += angle
        
        # Return inverse of average curvature (higher is smoother)
        avg_curvature = total_curvature / (len(path) - 2)
        return 1.0 / (1.0 + avg_curvature)


class BenchmarkSuite:
    """Suite for running comprehensive benchmarks."""
    
    def __init__(self, evaluator: BaseEvaluator) -> None:
        """
        Initialize benchmark suite.
        
        Args:
            evaluator: The evaluator to use
        """
        self.evaluator = evaluator
        self.results: List[Dict[str, Any]] = []
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a result to the suite.
        
        Args:
            result: Result dictionary
        """
        self.results.append(result)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all results.
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        summary = df.groupby('algorithm').agg({
            'success_rate': ['mean', 'std'],
            'avg_path_length': ['mean', 'std'],
            'avg_computation_time': ['mean', 'std'],
            'avg_nodes_expanded': ['mean', 'std'],
            'avg_path_cost': ['mean', 'std'],
            'avg_clearance': ['mean', 'std'],
            'avg_smoothness': ['mean', 'std'],
            'total_trials': 'sum'
        }).round(4)
        
        return summary
    
    def get_leaderboard(self, metric: str = 'success_rate') -> pd.DataFrame:
        """
        Get leaderboard sorted by specified metric.
        
        Args:
            metric: Metric to sort by
            
        Returns:
            DataFrame with leaderboard
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # Group by algorithm and calculate mean
        leaderboard = df.groupby('algorithm')[metric].mean().sort_values(ascending=False)
        
        return pd.DataFrame({
            'algorithm': leaderboard.index,
            metric: leaderboard.values,
            'rank': range(1, len(leaderboard) + 1)
        })
    
    def save_results(self, filepath: str) -> None:
        """
        Save results to file.
        
        Args:
            filepath: Path to save results
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
    
    def load_results(self, filepath: str) -> None:
        """
        Load results from file.
        
        Args:
            filepath: Path to load results from
        """
        df = pd.read_csv(filepath)
        self.results = df.to_dict('records')
