# Multi-Objective Optimization Benchmark Study
# Comparing NSGA-II and SMS-EMOA on DTLZ Test Functions
# 
# Authors: [Your Names Here]
# Course: Multi-Objective Optimization, Summer Term 2025
# Institution: Paderborn University

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROBLEMS = ["dtlz1", "dtlz2", "dtlz4"]
N_VAR = 10  # decision variables
N_OBJ = 3   # objectives  
POP_SIZE = 100
N_GEN = 250  # generations
N_RUNS = 30  # independent runs for statistical significance
ALPHA = 0.05  # significance level

def setup_algorithms():
    """Create algorithm instances with standard parameters"""
    algorithms = {
        "NSGA-II": lambda: NSGA2(
            pop_size=POP_SIZE,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0/N_VAR, eta=20),
            eliminate_duplicates=True
        ),
        "SMS-EMOA": lambda: SMSEMOA(
            pop_size=POP_SIZE,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0/N_VAR, eta=20),
            eliminate_duplicates=True
        )
    }
    return algorithms

def run_benchmark():
    """Execute comprehensive benchmark study"""
    
    results = []
    algorithms = setup_algorithms()
    
    print("Starting Multi-Objective Optimization Benchmark Study")
    print("=" * 60)
    print(f"Problems: {PROBLEMS}")
    print(f"Algorithms: {list(algorithms.keys())}")
    print(f"Configuration: {N_VAR} vars, {N_OBJ} objs, {POP_SIZE} pop, {N_GEN} gen, {N_RUNS} runs")
    print("=" * 60)
    
    for problem_name in PROBLEMS:
        print(f"\nRunning benchmark on {problem_name.upper()}...")
        
        # Get the problem instance
        problem = get_problem(problem_name, n_var=N_VAR, n_obj=N_OBJ)
        
        # Get reference Pareto front for indicators
        pf = problem.pareto_front()
        if pf is None:
            from pymoo.util.ref_dirs import get_reference_directions
            ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=12)
            pf = problem.pareto_front(ref_dirs)
        
        # Setup performance indicators
        # Reference point should be dominated by all Pareto-optimal solutions
        ref_point = np.array([1.1, 1.1, 1.1])  # Anti-optimal reference point
        hv_indicator = HV(ref_point=ref_point)
        igd_indicator = IGD(pf)
        
        # Run each algorithm multiple times
        for alg_name, alg_constructor in algorithms.items():
            print(f"  Running {alg_name}...", end=" ")
            
            for run in range(N_RUNS):
                if (run + 1) % 10 == 0:
                    print(f"{run + 1}", end=" ")
                
                # Create fresh algorithm instance for each run
                algorithm = alg_constructor()
                
                # Run optimization
                res = minimize(
                    problem,
                    algorithm,
                    ('n_gen', N_GEN),
                    seed=run,  # Different seed for each run
                    verbose=False
                )
                
                # Calculate performance indicators
                hv_value = hv_indicator(res.F)
                igd_value = igd_indicator(res.F)
                
                # Store results
                results.append({
                    'Problem': problem_name.upper(),
                    'Algorithm': alg_name,
                    'Run': run + 1,
                    'Hypervolume': hv_value,
                    'IGD': igd_value,
                    'N_Solutions': len(res.F),
                    'Final_Pop': res.F  # Store final population for additional analysis
                })
            
            print("  Completed!")
    
    return pd.DataFrame(results)

def statistical_analysis(df):
    """Perform statistical analysis on benchmark results"""
    
    print("\nStatistical Analysis")
    print("=" * 40)
    
    # Summary statistics
    print("\nDescriptive Statistics:")
    summary = df.groupby(['Problem', 'Algorithm'])[['Hypervolume', 'IGD']].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).round(6)
    print(summary)
    
    # Statistical significance tests
    print("\nStatistical Significance Tests (Mann-Whitney U):")
    print("Null hypothesis: No difference between algorithms")
    print(f"Significance level: α = {ALPHA}")
    
    test_results = []
    
    for problem in df['Problem'].unique():
        problem_data = df[df['Problem'] == problem]
        
        # Get data for each algorithm
        nsga2_data = problem_data[problem_data['Algorithm'] == 'NSGA-II']
        sms_data = problem_data[problem_data['Algorithm'] == 'SMS-EMOA']
        
        # Test for Hypervolume
        hv_stat, hv_p = mannwhitneyu(
            nsga2_data['Hypervolume'], 
            sms_data['Hypervolume'], 
            alternative='two-sided'
        )
        
        # Test for IGD
        igd_stat, igd_p = mannwhitneyu(
            nsga2_data['IGD'], 
            sms_data['IGD'], 
            alternative='two-sided'
        )
        
        test_results.append({
            'Problem': problem,
            'Metric': 'Hypervolume',
            'U_statistic': hv_stat,
            'p_value': hv_p,
            'Significant': hv_p < ALPHA,
            'Effect_Size': calculate_effect_size(nsga2_data['Hypervolume'], sms_data['Hypervolume'])
        })
        
        test_results.append({
            'Problem': problem,
            'Metric': 'IGD',
            'U_statistic': igd_stat,
            'p_value': igd_p,
            'Significant': igd_p < ALPHA,
            'Effect_Size': calculate_effect_size(nsga2_data['IGD'], sms_data['IGD'])
        })
        
        print(f"\n{problem}:")
        print(f"  Hypervolume: U={hv_stat:.2f}, p={hv_p:.4f} {'*' if hv_p < ALPHA else ''}")
        print(f"  IGD: U={igd_stat:.2f}, p={igd_p:.4f} {'*' if igd_p < ALPHA else ''}")
    
    # Multiple comparison correction (Bonferroni)
    test_df = pd.DataFrame(test_results)
    test_df['p_value_corrected'] = test_df['p_value'] * len(test_df)
    test_df['Significant_corrected'] = test_df['p_value_corrected'] < ALPHA
    
    print("\nBonferroni-Corrected Results:")
    for _, row in test_df.iterrows():
        print(f"{row['Problem']} {row['Metric']}: p_corrected={row['p_value_corrected']:.4f} "
              f"{'*' if row['Significant_corrected'] else ''}")
    
    return test_df

def calculate_effect_size(group1, group2):
    """Calculate Vargha-Delaney A12 effect size"""
    n1, n2 = len(group1), len(group2)
    r1 = rankdata(np.concatenate([group1, group2]))[:n1]
    return (np.sum(r1) - n1 * (n1 + 1) / 2) / (n1 * n2)

def create_visualizations(df):
    """Create publication-quality visualizations"""
    
    # Set style for publication
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Performance Comparison: NSGA-II vs SMS-EMOA', fontsize=16, fontweight='bold')
    
    problems = df['Problem'].unique()
    metrics = ['Hypervolume', 'IGD']
    
    for i, metric in enumerate(metrics):
        for j, problem in enumerate(problems):
            ax = axes[i, j]
            
            # Filter data for current problem
            problem_data = df[df['Problem'] == problem]
            
            # Create box plot
            box_data = [
                problem_data[problem_data['Algorithm'] == 'NSGA-II'][metric],
                problem_data[problem_data['Algorithm'] == 'SMS-EMOA'][metric]
            ]
            
            bp = ax.boxplot(box_data, labels=['NSGA-II', 'SMS-EMOA'], patch_artist=True)
            
            # Customize colors
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightgreen')
            
            # Set titles and labels
            ax.set_title(f'{problem}')
            if j == 0:
                ax.set_ylabel(metric)
            if i == 1:
                ax.set_xlabel('Algorithm')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_performance_tables(df):
    """Generate LaTeX tables for the report"""
    
    # Summary statistics table
    summary = df.groupby(['Problem', 'Algorithm'])[['Hypervolume', 'IGD']].agg([
        'mean', 'std'
    ]).round(4)
    
    # Reshape for better presentation
    table_data = []
    for problem in df['Problem'].unique():
        for metric in ['Hypervolume', 'IGD']:
            nsga2_mean = summary.loc[(problem, 'NSGA-II'), (metric, 'mean')]
            nsga2_std = summary.loc[(problem, 'NSGA-II'), (metric, 'std')]
            sms_mean = summary.loc[(problem, 'SMS-EMOA'), (metric, 'mean')]
            sms_std = summary.loc[(problem, 'SMS-EMOA'), (metric, 'std')]
            
            table_data.append({
                'Problem': problem,
                'Metric': metric,
                'NSGA-II': f"{nsga2_mean:.4f} ± {nsga2_std:.4f}",
                'SMS-EMOA': f"{sms_mean:.4f} ± {sms_std:.4f}"
            })
    
    table_df = pd.DataFrame(table_data)
    
    # Generate LaTeX table
    latex_table = table_df.to_latex(index=False, escape=False, column_format='llcc')
    
    print("\nLaTeX Table for Results:")
    print(latex_table)
    
    # Save to file
    with open('results_table.tex', 'w') as f:
        f.write(latex_table)
    
    return table_df

def save_results(df, test_results):
    """Save all results to files"""
    
    # Save raw results
    df.to_csv('benchmark_results.csv', index=False)
    
    # Save statistical test results
    test_results.to_csv('statistical_tests.csv', index=False)
    
    # Save summary
    summary = df.groupby(['Problem', 'Algorithm'])[['Hypervolume', 'IGD']].describe()
    summary.to_csv('summary_statistics.csv')
    
    print("\nResults saved to:")
    print("- benchmark_results.csv")
    print("- statistical_tests.csv") 
    print("- summary_statistics.csv")
    print("- performance_comparison.pdf")
    print("- results_table.tex")

def main():
    """Main execution function"""
    
    print("Multi-Objective Optimization Benchmark Study")
    print("NSGA-II vs SMS-EMOA on DTLZ Problems")
    print("=" * 50)
    
    # Run benchmark
    df_results = run_benchmark()
    
    # Perform statistical analysis
    test_results = statistical_analysis(df_results)
    
    # Create visualizations
    create_visualizations(df_results)
    
    # Generate tables
    generate_performance_tables(df_results)
    
    # Save all results
    save_results(df_results, test_results)
    
    print("\nBenchmark study completed successfully!")
    print("All results, plots, and tables have been generated.")

if __name__ == "__main__":
    main()