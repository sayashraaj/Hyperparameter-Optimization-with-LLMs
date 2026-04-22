# Hyperparameter Optimisation of Convex Portfolio Trajectories Using Large Language Models

**Author:** Sayash Raaj
<br>
[LinkedIn](https://www.linkedin.com/in/sayashraaj/)

## Overview

This repository contains the code and experimental framework required to reproduce the findings presented in the paper **"Hyperparameter Optimisation of Convex Portfolio Trajectories Using Large Language Models"**. 

Multi-period portfolio allocation fundamentally relies on solving complex, constrained convex trajectory optimisation problems. The critical trade-offs between expected returns, risk aversion, and temporal reallocation smoothness are dictated by structural hyperparameters (lambda and gamma). Due to the computational expense of evaluating simulated economic regimes and the non-convex nature of the resulting hyperparameter landscape, identifying the optimal configuration is a non-trivial challenge.

This project introduces a **Hybrid Search** methodology that synthesizes the global contextual reasoning capabilities of Large Language Models (LLMs) with the precise local exploitation of quadratic surrogate models. Evaluated rigorously across four distinct simulated economic regimes, the Hybrid Search demonstrates statistically significant improvements in optimisation efficacy and sample efficiency compared to standard baselines.

## Methodology

The hyperparameter space is explored in log-space: log10(lambda) and log10(gamma) over the interval [-2, 2]. The codebase implements and evaluates four distinct search strategies:

1. **Grid Search:** A deterministic 10 x 10 uniform grid spanning the log-space.
2. **Random Search:** Uniform random sampling within the defined bounds.
3. **LLM Search:** An iterative approach leveraging the Gemini API (`gemini-flash-latest`). The LLM is prompted with historical evaluation traces (top performing pairs and their objective values) and proposes the next candidate pair.
4. **Hybrid Search:** A novel approach that alternates search logic. Following a random sampling burn-in period, the algorithm utilizes the LLM for global, context-aware proposals every third iteration. During the intervening iterations, it fits a local quadratic surrogate model to the top 20 historical points and analytically extracts the maximum of this quadratic surface for precise local exploitation.

## Repository Structure

* **`trajectory_optimisation.py`**: The primary execution script. Contains the regime generation, the CVXPY OSQP solver logic, objective evaluation formulation, LLM API integration, and the main multiprocessing execution loops.
* **`generate_plots.py`**: A dedicated visualization script utilizing `matplotlib` and `seaborn` to generate the 10 publication-ready figures mapping convergence, parameter sensitivity, and objective distributions.
* **`experiment_output/`**: (Generated during execution)
  * `results.csv`: Live-appended evaluation data.
  * `final_summary.txt`: A detailed statistical report of convergence, performance, and Welch's t-test comparisons.
  * `checkpoints/`: Pickled state files for trial resumption and fault tolerance.
  * `plots/`: Output directory for generated PDF and PNG figures.

## Installation and Prerequisites

The framework requires Python 3.8 or higher. The underlying convex programming is handled via CVXPY and the OSQP solver. 

Install the necessary dependencies using pip:

```bash
pip install cvxpy numpy pandas scipy requests matplotlib seaborn
```

## Execution Instructions

### 1. API Configuration
The LLM Search and Hybrid Search methods require access to the Google Gemini API. 
Before running the experiment, open `trajectory_optimisation.py`, locate the configuration section, and insert your API key:

```python
GEMINI_API_KEY = "YOUR_VALID_API_KEY"
GEMINI_MODEL = "gemini-flash-latest"
```

### 2. Running the Experiment
Execute the main script to initiate the trials across all four methodologies. The script features an automatic checkpointing system and will resume gracefully from the latest evaluation in the event of an interruption or API rate-limit exhaustion.

```bash
python trajectory_optimisation.py
```

### 3. Generating Visualizations
Once the trials are complete (or using partial data from `results.csv`), run the plotting script to generate the analytical figures. 

```bash
python generate_plots.py
```

## Key Results

The Hybrid Search method achieved the highest mean objective and demonstrated superior sample efficiency, reaching the 95% performance threshold in an average of 3.1 evaluations. This represents an approximate 93% reduction in necessary convex solves compared to the Grid Search baseline.

| Method | Best J (Mean) | Best J (Std) | Evals to 95% | Global Avg Stability |
| :--- | :--- | :--- | :--- | :--- |
| Grid Search | 0.021067 | 0.000340 | 58.4 | 2.76e-03 |
| Random Search | 0.021335 | 0.000761 | 3.6 | 2.70e-03 |
| LLM Search | 0.021306 | 0.001051 | 4.0 | 1.43e-03 |
| **Hybrid Search** | **0.022688** | **0.000899** | **3.1** | **1.44e-03** |

*Note: All comparisons against the Hybrid method yielded statistically significant performance differences (p < 0.01) based on independent Welch's t-tests.*

## Citation

If you utilize this codebase or methodology in your research, please consider citing the associated paper:

```bibtex
@article{raaj2026hyperparameter,
  title={Hyperparameter Optimisation of Convex Portfolio Trajectories Using Large Language Models},
  author={Raaj, Sayash},
  journal={NeurIPS 2026 (Preprint)},
  year={2026}
}
```

## License

This software is provided for academic and research purposes.