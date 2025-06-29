import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.problems.many import DTLZ1, DTLZ2, DTLZ4


from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

def run_experiment():
    """
    Main function to run the benchmark experiment.
    """
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 1. EXPERIMENTAL SETUP
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
    # Define the algorithms to be benchmarked
    # Using a population size of 100 as a standard choice
    algorithms = {
        'NSGA-II': NSGA2(pop_size=100),
        'SMS-EMOA': SMSEMOA(pop_size=100)
    }

    # Define the problems to be solved
    # All problems are configured with 10 decision variables and 3 objectives
    problems = {
        'DTLZ1': DTLZ1(n_var=10, n_obj=3),
        'DTLZ2': DTLZ2(n_var=10, n_obj=3),
        'DTLZ4': DTLZ4(n_var=10, n_obj=3)
    }

    # General experimental parameters
    n_runs = 30  # Number of independent runs for statistical significance
    termination = get_termination("n_gen", 200) # Terminate after 200 generations
    
    # List to store all results
    results_list = []

    print("Starting benchmark experiment...")
    print(f"Algorithms: {list(algorithms.keys())}")
    print(f"Problems: {list(problems.keys())}")
    print(f"Runs per setup: {n_runs}")
    print(f"Termination: {termination}")
    print("-" * 50)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 2. RUN THE EXPERIMENTS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
    for p_name, problem in problems.items():
        print(f"Running problem: {p_name}")
        
        # Get the true Pareto front for performance indicator calculation
        try:
            true_pf = problem.pareto_front()
        except Exception as e:
            print(f"Could not get true Pareto front for {p_name}: {e}")
            continue

        # Define the reference point for Hypervolume calculation
        # We use a point slightly worse than the nadir of the true Pareto front
        ref_point = np.max(true_pf, axis=0) * 1.1
        
        # Initialize performance indicators
        indicator_hv = HV(ref_point=ref_point)
        indicator_igd = IGD(true_pf)

        for a_name, algorithm in algorithms.items():
            print(f"  -> Algorithm: {a_name}")
            for i in range(n_runs):
                print(f"    - Run {i+1}/{n_runs}")
                
                # Perform the optimization
                res = minimize(problem,
                               algorithm,
                               termination,
                               seed=i, # Use run index as seed for reproducibility
                               verbose=False)

                # Calculate performance metrics
                # Check if solutions were found
                if res.F is not None and len(res.F) > 0:
                    hv = indicator_hv.do(res.F)
                    igd = indicator_igd.do(res.F)
                else:
                    # Assign worst-case values if no solution is found
                    hv = 0.0
                    igd = 1.0 # IGD is to be minimized, 1 is a poor value

                # Store the results
                results_list.append({
                    'problem': p_name,
                    'algorithm': a_name,
                    'run': i,
                    'hv': hv,
                    'igd': igd
                })
    
    # Convert results to a pandas DataFrame for easier analysis
    results_df = pd.DataFrame(results_list)
    
    # Save results to a CSV file for documentation and later use
    results_df.to_csv("benchmark_results.csv", index=False)
    print("\nExperiment finished. Results saved to benchmark_results.csv")
    print("-" * 50)

    return results_df


def analyze_and_visualize(df):
    """
    Analyzes the results and creates visualizations.
    """
    print("Analyzing and visualizing results...")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 3. VISUALIZATION: BOX PLOTS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
    metrics = ['hv', 'igd']
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='problem', y=metric, hue='algorithm', data=df)
        
        title = f'Hypervolume (HV)' if metric == 'hv' else 'Inverted Generational Distance (IGD)'
        ylabel = 'HV (higher is better)' if metric == 'hv' else 'IGD (lower is better)'
        
        plt.title(f'Distribution of {title} over 30 Runs', fontsize=16)
        plt.xlabel('Problem', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(title='Algorithm')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"boxplot_{metric}.png")
        plt.show()

    print("\nBox plots for HV and IGD have been generated and saved.")
    print("-" * 50)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # 4. STATISTICAL ANALYSIS
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
    print("Performing statistical tests (Wilcoxon rank-sum)...")
    
    problems = df['problem'].unique()
    algorithms = df['algorithm'].unique()
    
    # Ensure there are two algorithms to compare
    if len(algorithms) != 2:
        print("Statistical comparison requires exactly two algorithms.")
        return
        
    algo1, algo2 = algorithms[0], algorithms[1]
    
    # We are performing 2 metrics * 3 problems = 6 tests
    n_tests = len(metrics) * len(problems)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests
    
    print(f"Significance level (alpha): {alpha}")
    print(f"Number of tests: {n_tests}")
    print(f"Bonferroni corrected alpha: {bonferroni_alpha:.4f}")
    
    stat_results = []

    for p_name in problems:
        for metric in metrics:
            data1 = df[(df['problem'] == p_name) & (df['algorithm'] == algo1)][metric]
            data2 = df[(df['problem'] == p_name) & (df['algorithm'] == algo2)][metric]
            
            # Perform Wilcoxon rank-sum test
            stat, p_value = ranksums(data1, data2)
            
            # Determine which algorithm is better
            mean1 = data1.mean()
            mean2 = data2.mean()
            
            winner = 'None'
            if p_value < bonferroni_alpha:
                if metric == 'hv': # Higher is better
                    winner = algo1 if mean1 > mean2 else algo2
                else: # IGD, lower is better
                    winner = algo1 if mean1 < mean2 else algo2

            stat_results.append({
                'Problem': p_name,
                'Metric': metric.upper(),
                'p-value': p_value,
                'Significant': 'Yes' if p_value < bonferroni_alpha else 'No',
                'Winner': winner
            })

    stat_df = pd.DataFrame(stat_results)
    print("\nStatistical Test Results:")
    print(stat_df.to_string(index=False))


if __name__ == '__main__':
    # Run the experiment
    results_df = run_experiment()
    
    # Analyze and visualize the results
    analyze_and_visualize(results_df)

