# Path Planning Algorithms

Path planning algorithms for robotics with comprehensive implementations of A*, RRT*, PRM*, D*, and other state-of-the-art methods.

## ⚠️ SAFETY DISCLAIMER

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. DO NOT USE ON REAL ROBOTS WITHOUT PROPER SAFETY REVIEW AND TESTING.**

This project includes simulation environments and path planning algorithms that have not been validated for real-world deployment. Before using on actual robots:

- Conduct thorough safety analysis
- Implement proper emergency stop mechanisms
- Add velocity and acceleration limits
- Test extensively in controlled environments
- Review all code with robotics safety experts

The authors assume no responsibility for any damage, injury, or loss resulting from the use of this software.

## Features

- **Multiple Algorithms**: A*, RRT*, PRM*, D*, Jump Point Search, and more
- **Modern Architecture**: Clean, typed code with comprehensive documentation
- **Simulation Support**: Integration with PyBullet and Gymnasium
- **Evaluation Framework**: Comprehensive metrics and benchmarking
- **Interactive Demos**: Streamlit-based visualization and testing
- **ROS 2 Ready**: Optional ROS 2 integration for real robot deployment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Path-Planning-Algorithms.git
cd Path-Planning-Algorithms

# Install in development mode
pip install -e ".[dev,simulation]"

# Or install minimal version
pip install -e .
```

### Basic Usage

```python
import numpy as np
from src.planners import AStarPlanner
from src.environments import GridEnvironment

# Create a simple grid environment
grid = np.zeros((20, 20))
grid[5:15, 10] = 1  # Add obstacles

# Create planner
planner = AStarPlanner()
environment = GridEnvironment(grid)

# Plan path
start = (0, 0)
goal = (19, 19)
path = planner.plan(environment, start, goal)

# Visualize
environment.visualize_path(path)
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
src/
├── planners/          # Path planning algorithms
├── environments/       # Simulation environments
├── robots/            # Robot models and kinematics
├── evaluation/        # Metrics and benchmarking
├── utils/             # Utilities and helpers
└── visualization/     # Plotting and visualization

config/                # Configuration files
tests/                 # Unit and integration tests
demo/                  # Interactive demos
assets/                # Images, videos, and data
docs/                  # Documentation
```

## Algorithms Implemented

### Deterministic Planners
- **A***: Optimal pathfinding with heuristic guidance
- **D***: Dynamic replanning for changing environments
- **Jump Point Search**: Optimized A* for uniform-cost grids
- **Dijkstra**: Optimal shortest path algorithm

### Sampling-Based Planners
- **RRT***: Rapidly-exploring Random Trees (optimal)
- **PRM***: Probabilistic Roadmap Method
- **RRT-Connect**: Bidirectional RRT variant
- **EST**: Expansive Space Trees

### Advanced Methods
- **Hybrid A***: Combines A* with sampling for complex environments
- **Anytime RRT***: Incrementally improving solutions
- **Multi-robot**: Coordinated path planning for robot teams

## Evaluation Metrics

- **Success Rate**: Percentage of successful pathfinding attempts
- **Path Length**: Distance of found paths vs optimal
- **Computation Time**: Planning time and memory usage
- **Smoothness**: Path curvature and jerk metrics
- **Clearance**: Minimum distance to obstacles
- **Robustness**: Performance under uncertainty

## Configuration

All algorithms can be configured via YAML files in the `config/` directory:

```yaml
# config/a_star.yaml
algorithm: "a_star"
heuristic: "euclidean"  # euclidean, manhattan, diagonal
allow_diagonal: true
smoothing: true
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_planners.py::test_a_star
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{path_planning_algorithms,
  title={Path Planning Algorithms for Robotics},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Path-Planning-Algorithms}
}
```

## Acknowledgments

- Open Motion Planning Library (OMPL) for inspiration
- ROS 2 Navigation Stack for real-world applications
- PyBullet community for simulation support
# Path-Planning-Algorithms
