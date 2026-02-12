"""
Comprehensive evaluation system for path planning algorithms.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import time
from pathlib import Path

from ..planners.base import BasePlanner, PlanningResult
from ..environments.base import BaseEnvironment
from .base import BaseEvaluator, EvaluationResult, BenchmarkSuite


class ComprehensiveEvaluator(BaseEvaluator):
    """Comprehensive evaluator for path planning algorithms."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize evaluator."""
        super().__init__(**kwargs)
    
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
        # Run planning
        result = planner.plan(environment, start, goal, **kwargs)
        
        if result.success:
            # Calculate additional metrics
            path_length = self.calculate_path_length(result.path)
            clearance = self.calculate_path_clearance(result.path, environment)
            smoothness = self.calculate_path_smoothness(result.path)
            
            return EvaluationResult(
                algorithm_name=planner.__class__.__name__,
                success_rate=1.0,
                average_path_length=path_length,
                average_computation_time=result.computation_time,
                average_nodes_expanded=result.nodes_expanded,
                average_path_cost=result.cost,
                average_clearance=clearance,
                average_smoothness=smoothness,
                total_trials=1,
                successful_trials=1,
                metadata=result.metadata
            )
        else:
            return EvaluationResult(
                algorithm_name=planner.__class__.__name__,
                success_rate=0.0,
                average_path_length=0.0,
                average_computation_time=result.computation_time,
                average_nodes_expanded=result.nodes_expanded,
                average_path_cost=float('inf'),
                average_clearance=0.0,
                average_smoothness=0.0,
                total_trials=1,
                successful_trials=0,
                metadata=result.metadata
            )
    
    def benchmark_multiple_trials(
        self,
        planner: BasePlanner,
        environment: BaseEnvironment,
        problem_instances: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Benchmark a planner on multiple problem instances.
        
        Args:
            planner: The planner to benchmark
            environment: The environment
            problem_instances: List of (start, goal) tuples
            **kwargs: Additional parameters
            
        Returns:
            Aggregated evaluation result
        """
        results = []
        
        for start, goal in problem_instances:
            result = self.evaluate(planner, environment, start, goal, **kwargs)
            results.append(result)
        
        # Aggregate results
        successful_results = [r for r in results if r.successful_trials > 0]
        
        if successful_results:
            success_rate = len(successful_results) / len(results)
            avg_path_length = np.mean([r.average_path_length for r in successful_results])
            avg_computation_time = np.mean([r.average_computation_time for r in results])
            avg_nodes_expanded = np.mean([r.average_nodes_expanded for r in results])
            avg_path_cost = np.mean([r.average_path_cost for r in successful_results])
            avg_clearance = np.mean([r.average_clearance for r in successful_results])
            avg_smoothness = np.mean([r.average_smoothness for r in successful_results])
        else:
            success_rate = 0.0
            avg_path_length = 0.0
            avg_computation_time = np.mean([r.average_computation_time for r in results])
            avg_nodes_expanded = np.mean([r.average_nodes_expanded for r in results])
            avg_path_cost = float('inf')
            avg_clearance = 0.0
            avg_smoothness = 0.0
        
        return EvaluationResult(
            algorithm_name=planner.__class__.__name__,
            success_rate=success_rate,
            average_path_length=avg_path_length,
            average_computation_time=avg_computation_time,
            average_nodes_expanded=avg_nodes_expanded,
            average_path_cost=avg_path_cost,
            average_clearance=avg_clearance,
            average_smoothness=avg_smoothness,
            total_trials=len(results),
            successful_trials=len(successful_results),
            metadata={
                "num_trials": len(results),
                "successful_trials": len(successful_results),
                "std_path_length": np.std([r.average_path_length for r in successful_results]) if successful_results else 0.0,
                "std_computation_time": np.std([r.average_computation_time for r in results]),
                "std_nodes_expanded": np.std([r.average_nodes_expanded for r in results])
            }
        )


def run_comprehensive_benchmark(
    planners: List[BasePlanner],
    environments: List[BaseEnvironment],
    num_trials_per_env: int = 50,
    save_results: bool = True,
    results_dir: str = "assets/benchmark_results"
) -> pd.DataFrame:
    """
    Run comprehensive benchmark across multiple planners and environments.
    
    Args:
        planners: List of planners to benchmark
        environments: List of environments
        num_trials_per_env: Number of trials per environment
        save_results: Whether to save results
        results_dir: Directory to save results
        
    Returns:
        DataFrame with benchmark results
    """
    evaluator = ComprehensiveEvaluator()
    benchmark_suite = BenchmarkSuite(evaluator)
    
    all_results = []
    
    for env_idx, environment in enumerate(environments):
        print(f"Benchmarking environment {env_idx + 1}/{len(environments)}")
        
        # Create problem instances for this environment
        problem_instances = []
        for _ in range(num_trials_per_env):
            start = environment.get_random_valid_position()
            goal = environment.get_random_valid_position()
            
            # Ensure start and goal are different
            while start == goal:
                goal = environment.get_random_valid_position()
            
            problem_instances.append((start, goal))
        
        # Benchmark each planner
        for planner in planners:
            print(f"  Benchmarking {planner.__class__.__name__}")
            
            result = evaluator.benchmark_multiple_trials(
                planner, environment, problem_instances
            )
            
            # Add environment info
            result_dict = result.to_dict()
            result_dict['environment'] = f"env_{env_idx}"
            result_dict['num_trials'] = num_trials_per_env
            
            all_results.append(result_dict)
            benchmark_suite.add_result(result_dict)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    if save_results:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = Path(results_dir) / f"benchmark_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
    
    return df


def create_leaderboard(
    benchmark_results: pd.DataFrame,
    metric: str = 'success_rate',
    top_k: int = 10
) -> pd.DataFrame:
    """
    Create leaderboard from benchmark results.
    
    Args:
        benchmark_results: DataFrame with benchmark results
        metric: Metric to rank by
        top_k: Number of top results to show
        
    Returns:
        Leaderboard DataFrame
    """
    # Group by algorithm and calculate mean
    grouped = benchmark_results.groupby('algorithm')[metric].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    
    # Create leaderboard
    leaderboard = pd.DataFrame({
        'rank': range(1, len(grouped) + 1),
        'algorithm': grouped.index,
        f'{metric}_mean': grouped['mean'],
        f'{metric}_std': grouped['std'],
        'num_experiments': grouped['count']
    })
    
    return leaderboard.head(top_k)


def generate_benchmark_report(
    benchmark_results: pd.DataFrame,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive benchmark report.
    
    Args:
        benchmark_results: DataFrame with benchmark results
        save_path: Optional path to save report
        
    Returns:
        Report text
    """
    report = []
    report.append("# Path Planning Algorithms Benchmark Report")
    report.append("=" * 50)
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    
    summary_stats = benchmark_results.groupby('algorithm').agg({
        'success_rate': ['mean', 'std'],
        'avg_computation_time': ['mean', 'std'],
        'avg_path_length': ['mean', 'std'],
        'avg_nodes_expanded': ['mean', 'std'],
        'avg_clearance': ['mean', 'std'],
        'avg_smoothness': ['mean', 'std']
    }).round(4)
    
    report.append(summary_stats.to_string())
    report.append("")
    
    # Success rate leaderboard
    report.append("## Success Rate Leaderboard")
    report.append("")
    
    success_leaderboard = create_leaderboard(benchmark_results, 'success_rate')
    report.append(success_leaderboard.to_string(index=False))
    report.append("")
    
    # Computation time leaderboard
    report.append("## Computation Time Leaderboard (lower is better)")
    report.append("")
    
    time_leaderboard = create_leaderboard(benchmark_results, 'avg_computation_time')
    report.append(time_leaderboard.to_string(index=False))
    report.append("")
    
    # Path length leaderboard
    report.append("## Path Length Leaderboard (lower is better)")
    report.append("")
    
    length_leaderboard = create_leaderboard(benchmark_results, 'avg_path_length')
    report.append(length_leaderboard.to_string(index=False))
    report.append("")
    
    # Best overall algorithm
    report.append("## Best Overall Algorithm")
    report.append("")
    
    # Calculate composite score (success_rate * 0.4 + (1/time) * 0.3 + (1/length) * 0.3)
    composite_scores = []
    for algorithm in benchmark_results['algorithm'].unique():
        alg_data = benchmark_results[benchmark_results['algorithm'] == algorithm]
        
        success_rate = alg_data['success_rate'].mean()
        avg_time = alg_data['avg_computation_time'].mean()
        avg_length = alg_data['avg_path_length'].mean()
        
        # Normalize scores (simple normalization)
        time_score = 1.0 / (1.0 + avg_time) if avg_time > 0 else 0
        length_score = 1.0 / (1.0 + avg_length) if avg_length > 0 else 0
        
        composite_score = success_rate * 0.4 + time_score * 0.3 + length_score * 0.3
        composite_scores.append((algorithm, composite_score))
    
    composite_scores.sort(key=lambda x: x[1], reverse=True)
    
    report.append(f"**Winner: {composite_scores[0][0]}** (Score: {composite_scores[0][1]:.4f})")
    report.append("")
    
    for i, (algorithm, score) in enumerate(composite_scores[:5]):
        report.append(f"{i+1}. {algorithm}: {score:.4f}")
    
    report.append("")
    
    # Report text
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")
    
    return report_text
