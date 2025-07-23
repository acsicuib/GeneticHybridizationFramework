import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_bash_config
import os

COLORS = dict(zip("NSGA2 NSGA3 UNSGA3 SMSEMOA Hybrid".split(),['#ff8500', '#FF595E', '#1982C4', '#6A4C93' ,'#8AC926']))


def perform_wilcoxon_tests(df, metrics, algorithms, alpha=0.05):
    """
    Perform Wilcoxon rank sum tests for all pairs of algorithms on each metric
    
    Args:
        df: DataFrame with metrics data
        metrics: List of metric names to test
        algorithms: List of algorithm names
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Get the final generation data for each algorithm and seed
    print(df["Algorithm"].unique())
    print("--------------------------------")
    final_gen_data = df.groupby(['Algorithm', 'Seed'])['Generation'].max().reset_index()
    final_data = df.merge(final_gen_data, on=['Algorithm', 'Seed', 'Generation'])
    
    print(final_data.tail())
    print(final_data.head())
    print(final_data["Algorithm"].unique())

    print(f"Final generation data shape: {final_data.shape}")
    print(f"Algorithms found: {final_data['Algorithm'].unique()}")
    
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        results[metric] = {}
        
        # Get all unique algorithm pairs
        algorithm_pairs = list(combinations(algorithms, 2))
        
        for alg1, alg2 in algorithm_pairs:
            # Get data for both algorithms
            data1 = final_data[final_data['Algorithm'] == alg1][metric].values
            data2 = final_data[final_data['Algorithm'] == alg2][metric].values
            
            # Remove NaN values
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]
            
            if len(data1) == 0 or len(data2) == 0:
                print(f"Warning: No valid data for {alg1} vs {alg2} in {metric}")
                continue
            
            # Perform Wilcoxon rank sum test
            try:
                statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
                
                # Determine significance
                is_significant = p_value < alpha
                
                # Calculate effect size (rank-biserial correlation)
                n1, n2 = len(data1), len(data2)
                effect_size = 1 - (2 * statistic) / (n1 * n2)
                
                results[metric][f"{alg1}_vs_{alg2}"] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'effect_size': effect_size,
                    'n1': n1,
                    'n2': n2,
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'std1': np.std(data1),
                    'std2': np.std(data2)
                }
                
                print(f"  {alg1} vs {alg2}: p={p_value:.9f}, significant={is_significant}, effect_size={effect_size:.9f}")
                
            except Exception as e:
                print(f"Error in Wilcoxon test for {alg1} vs {alg2} in {metric}: {e}")
                results[metric][f"{alg1}_vs_{alg2}"] = {
                    'error': str(e)
                }
    
    return results

def create_summary_table(results, metrics, algorithms):
    """Create a summary table of all test results"""
    summary_data = []
    
    for metric in metrics:
        for comparison, result in results[metric].items():
            if 'error' not in result:
                alg1, alg2 = comparison.split('_vs_')
                summary_data.append({
                    'Metric': metric,
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'P_Value': result['p_value'],
                    'Significant': result['is_significant'],
                    'Effect_Size': result['effect_size'],
                    'Mean_1': result['mean1'],
                    'Mean_2': result['mean2'],
                    'Std_1': result['std1'],
                    'Std_2': result['std2'],
                    'N_1': result['n1'],
                    'N_2': result['n2']
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_heatmap(results, metrics, algorithms, output_file='plots/wilcoxon_heatmap.png'):
    """Create a heatmap showing p-values for all comparisons"""
    # Create a matrix for p-values
    n_algs = len(algorithms)
    p_matrix = np.full((n_algs, n_algs), np.nan)
    
    # Fill the matrix with p-values
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i != j:
                # Find the comparison
                comparison = f"{alg1}_vs_{alg2}"
                reverse_comparison = f"{alg2}_vs_{alg1}"
                
                # Check if this comparison exists in any metric
                p_values = []
                for metric in metrics:
                    if comparison in results[metric]:
                        if 'error' not in results[metric][comparison]:
                            p_values.append(results[metric][comparison]['p_value'])
                    elif reverse_comparison in results[metric]:
                        if 'error' not in results[metric][reverse_comparison]:
                            p_values.append(results[metric][reverse_comparison]['p_value'])
                
                if p_values:
                    # Use the minimum p-value across all metrics
                    p_matrix[i, j] = min(p_values)
    
    # Create the heatmap
    fig, axes = plt.subplots(2, 3, figsize=(5*3, 5*2))
    axes = axes.flatten()
    if len(metrics) == 1:
        axes = [axes[0]]
    
    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Create metric-specific matrix
        metric_matrix = np.full((n_algs, n_algs), np.nan)
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i != j:
                    comparison = f"{alg1}_vs_{alg2}"
                    reverse_comparison = f"{alg2}_vs_{alg1}"
                    
                    if comparison in results[metric]:
                        if 'error' not in results[metric][comparison]:
                            metric_matrix[i, j] = results[metric][comparison]['p_value']
                    elif reverse_comparison in results[metric]:
                        if 'error' not in results[metric][reverse_comparison]:
                            metric_matrix[i, j] = results[metric][reverse_comparison]['p_value']
        
        # Create heatmap
        sns.heatmap(metric_matrix, 
                   annot=True, 
                   fmt='.4f',
                   cmap='RdYlBu_r',
                   vmin=0, 
                   vmax=0.05,
                   cbar_kws={'label': 'p-value'},
                   xticklabels=algorithms,
                   yticklabels=algorithms,
                   ax=ax)
        
        ax.set_title(f'{metric} - Wilcoxon p-values')
        ax.set_xlabel('Algorithm 2')
        ax.set_ylabel('Algorithm 1')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def create_boxplots(df, metrics, algorithms, output_file='plots/metrics_boxplots.png'):
    """Create boxplots for each metric across algorithms in a 2-row, 3-column layout"""
    # Get final generation data
    final_gen_data = df.groupby(['Algorithm', 'Seed'])['Generation'].max().reset_index()
    final_data = df.merge(final_gen_data, on=['Algorithm', 'Seed', 'Generation'])
    
    # Filter for algorithms we're interested in
    final_data = final_data[final_data['Algorithm'].isin(algorithms)]
    final_data["Algorithm"] = final_data["Algorithm"].str.replace("Hybrids", "Hybrid")

    n_metrics = len(metrics)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 6*nrows))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        # Create boxplot
        sns.boxplot(data=final_data, x='Algorithm', y=metric, ax=ax, palette=COLORS)
        ax.set_title(f'{metric} Distribution by Algorithm')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide any unused subplots
    for idx in range(n_metrics, nrows * ncols):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_convergence_speed(df, metrics, algorithms, alpha=0.05):
    """
    Analyze convergence speed by comparing algorithms across generations
    
    Args:
        df: DataFrame with metrics data
        metrics: List of metric names to test
        algorithms: List of algorithm names
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with convergence analysis results
    """
    results = {}
    
    for metric in metrics:
        print(f"\nAnalyzing convergence for metric: {metric}")
        results[metric] = {}
        
        # Get all unique algorithm pairs
        algorithm_pairs = list(combinations(algorithms, 2))
        
        for alg1, alg2 in algorithm_pairs:
            # Get data for both algorithms across all generations
            data1 = df[df['Algorithm'] == alg1][['Seed', 'Generation', metric]].copy()
            data2 = df[df['Algorithm'] == alg2][['Seed', 'Generation', metric]].copy()
            
            # Calculate convergence speed (generation to reach 90% of final performance)
            convergence_speeds = []
            
            for seed in data1['Seed'].unique():
                if seed in data2['Seed'].unique():
                    # Get performance trajectory for this seed
                    perf1 = data1[data1['Seed'] == seed].sort_values('Generation')[metric].values
                    perf2 = data2[data2['Seed'] == seed].sort_values('Generation')[metric].values
                    
                    if len(perf1) > 0 and len(perf2) > 0:
                        # Calculate convergence speed (generation to reach 90% of final performance)
                        final1, final2 = perf1[-1], perf2[-1]
                        threshold1, threshold2 = final1 * 0.9, final2 * 0.9
                        
                        # Find generation where performance reaches threshold
                        conv1 = np.argmax(perf1 >= threshold1) if np.any(perf1 >= threshold1) else len(perf1)
                        conv2 = np.argmax(perf2 >= threshold2) if np.any(perf2 >= threshold2) else len(perf2)
                        
                        convergence_speeds.append((conv1, conv2))
            
            if convergence_speeds:
                conv1_vals = [c[0] for c in convergence_speeds]
                conv2_vals = [c[1] for c in convergence_speeds]
                
                # Perform Wilcoxon test on convergence speeds
                try:
                    statistic, p_value = wilcoxon(conv1_vals, conv2_vals, alternative='two-sided')
                    is_significant = p_value < alpha
                    effect_size = 1 - (2 * statistic) / (len(conv1_vals) * len(conv2_vals))
                    
                    results[metric][f"{alg1}_vs_{alg2}"] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'effect_size': effect_size,
                        'n1': len(conv1_vals),
                        'n2': len(conv2_vals),
                        'mean1': np.mean(conv1_vals),
                        'mean2': np.mean(conv2_vals),
                        'std1': np.std(conv1_vals),
                        'std2': np.std(conv2_vals)
                    }
                    
                    print(f"  {alg1} vs {alg2}: p={p_value:.9f}, significant={is_significant}, effect_size={effect_size:.9f}")
                    print(f"    Avg convergence: {np.mean(conv1_vals):.8f} vs {np.mean(conv2_vals):.8f} generations")
                    
                except Exception as e:
                    print(f"Error in convergence analysis for {alg1} vs {alg2} in {metric}: {e}")
                    results[metric][f"{alg1}_vs_{alg2}"] = {'error': str(e)}
    
    return results

def create_convergence_plot(df, metrics, algorithms, output_file='plots/convergence_analysis.png'):
    """Create convergence plots showing performance over generations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Calculate mean performance per generation for each algorithm
        for algorithm in algorithms:
            alg_data = df[df['Algorithm'] == algorithm]
            if len(alg_data) > 0:
                mean_performance = alg_data.groupby('Generation')[metric].mean()
                std_performance = alg_data.groupby('Generation')[metric].std()
                
                generations = mean_performance.index
                means = mean_performance.values
                stds = std_performance.values
                
                ax.plot(generations, means, label=algorithm, linewidth=2)
                ax.fill_between(generations, means - stds, means + stds, alpha=0.2)
        
        ax.set_title(f'{metric} Convergence')
        ax.set_xlabel('Generation')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    if len(metrics) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_individual_performance(df, metrics, algorithms):
    """
    Analyze which algorithm is better for each individual metric, seed, and replicate
    
    Args:
        df: DataFrame with metrics data
        metrics: List of metric names to test
        algorithms: List of algorithm names
    
    Returns:
        Dictionary with detailed performance comparisons
    """
    results = {}
    
    # Get final generation data for each algorithm and seed
    final_gen_data = df.groupby(['Algorithm', 'Seed'])['Generation'].max().reset_index()
    final_data = df.merge(final_gen_data, on=['Algorithm', 'Seed', 'Generation'])
    
    for metric in metrics:
        print(f"\nAnalyzing individual performance for metric: {metric}")
        results[metric] = {}
        
        # Get all unique algorithm pairs
        algorithm_pairs = list(combinations(algorithms, 2))
        
        for alg1, alg2 in algorithm_pairs:
            comparison_key = f"{alg1}_vs_{alg2}"
            results[metric][comparison_key] = {
                'wins_alg1': 0,
                'wins_alg2': 0,
                'ties': 0,
                'total_comparisons': 0,
                'detailed_results': []
            }
            
            # Get data for both algorithms
            data1 = final_data[final_data['Algorithm'] == alg1][['Seed', metric]].copy()
            data2 = final_data[final_data['Algorithm'] == alg2][['Seed', metric]].copy()
            
            # Find common seeds
            common_seeds = set(data1['Seed'].unique()) & set(data2['Seed'].unique())
            
            for seed in common_seeds:
                val1 = data1[data1['Seed'] == seed][metric].iloc[0]
                val2 = data2[data2['Seed'] == seed][metric].iloc[0]
                
                # Determine which algorithm is better based on metric type
                # For GD, IGD, S, STE: lower is better
                # For HV: higher is better
                if metric in ['GD', 'IGD', 'S', 'STE']:
                    if val1 < val2:
                        winner = alg1
                        results[metric][comparison_key]['wins_alg1'] += 1
                    elif val2 < val1:
                        winner = alg2
                        results[metric][comparison_key]['wins_alg2'] += 1
                    else:
                        winner = 'Tie'
                        results[metric][comparison_key]['ties'] += 1
                else:  # HV
                    if val1 > val2:
                        winner = alg1
                        results[metric][comparison_key]['wins_alg1'] += 1
                    elif val2 > val1:
                        winner = alg2
                        results[metric][comparison_key]['wins_alg2'] += 1
                    else:
                        winner = 'Tie'
                        results[metric][comparison_key]['ties'] += 1
                
                results[metric][comparison_key]['total_comparisons'] += 1
                results[metric][comparison_key]['detailed_results'].append({
                    'seed': seed,
                    f'{alg1}_value': val1,
                    f'{alg2}_value': val2,
                    'winner': winner,
                    'difference': val1 - val2
                })
            
            # Calculate win rates
            total = results[metric][comparison_key]['total_comparisons']
            if total > 0:
                win_rate_alg1 = results[metric][comparison_key]['wins_alg1'] / total * 100
                win_rate_alg2 = results[metric][comparison_key]['wins_alg2'] / total * 100
                tie_rate = results[metric][comparison_key]['ties'] / total * 100
                
                print(f"  {alg1} vs {alg2}:")
                print(f"    {alg1} wins: {results[metric][comparison_key]['wins_alg1']}/{total} ({win_rate_alg1:.1f}%)")
                print(f"    {alg2} wins: {results[metric][comparison_key]['wins_alg2']}/{total} ({win_rate_alg2:.1f}%)")
                print(f"    Ties: {results[metric][comparison_key]['ties']}/{total} ({tie_rate:.1f}%)")
    
    return results

def create_individual_performance_table(individual_results, metrics, algorithms):
    """Create a summary table of individual performance comparisons"""
    summary_data = []
    
    for metric in metrics:
        for comparison, result in individual_results[metric].items():
            alg1, alg2 = comparison.split('_vs_')
            total = result['total_comparisons']
            
            if total > 0:
                win_rate_alg1 = result['wins_alg1'] / total * 100
                win_rate_alg2 = result['wins_alg2'] / total * 100
                tie_rate = result['ties'] / total * 100
                
                summary_data.append({
                    'Metric': metric,
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Total_Comparisons': total,
                    'Wins_Alg1': result['wins_alg1'],
                    'Wins_Alg2': result['wins_alg2'],
                    'Ties': result['ties'],
                    'Win_Rate_Alg1_%': win_rate_alg1,
                    'Win_Rate_Alg2_%': win_rate_alg2,
                    'Tie_Rate_%': tie_rate,
                    'Dominant_Algorithm': alg1 if win_rate_alg1 > win_rate_alg2 else alg2 if win_rate_alg2 > win_rate_alg1 else 'Tie'
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_detailed_seed_analysis(individual_results, metrics, algorithms, output_file='individual_performance_detailed.csv'):
    """Create detailed CSV with individual seed comparisons"""
    detailed_data = []
    
    for metric in metrics:
        for comparison, result in individual_results[metric].items():
            alg1, alg2 = comparison.split('_vs_')
            
            for detail in result['detailed_results']:
                detailed_data.append({
                    'Metric': metric,
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Seed': detail['seed'],
                    f'{alg1}_Value': detail[f'{alg1}_value'],
                    f'{alg2}_Value': detail[f'{alg2}_value'],
                    'Winner': detail['winner'],
                    'Difference': detail['difference']
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(output_file, index=False)
    print(f"Detailed seed analysis saved to {output_file}")
    return detailed_df

def create_win_rate_heatmap(individual_results, metrics, algorithms, output_file='plots/win_rate_heatmap.png'):
    """Create a heatmap showing win rates for each algorithm pair"""
    fig, axes = plt.subplots(2, 3, figsize=(5*3, 5*2))
    axes = axes.flatten()
    if len(metrics) == 1:
        axes = [axes[0]]
    algorithmsLabels = ["NSGA2", "NSGA3", "UNSGA3", "SMSEMOA", "Hybrid"]
    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Create win rate matrix
        n_algs = len(algorithms)
        win_rate_matrix = np.full((n_algs, n_algs), np.nan)
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i != j:
                    comparison = f"{alg1}_vs_{alg2}"
                    reverse_comparison = f"{alg2}_vs_{alg1}"
                    
                    if comparison in individual_results[metric]:
                        result = individual_results[metric][comparison]
                        total = result['total_comparisons']
                        if total > 0:
                            win_rate = result['wins_alg1'] / total * 100
                            win_rate_matrix[i, j] = win_rate
                    elif reverse_comparison in individual_results[metric]:
                        result = individual_results[metric][reverse_comparison]
                        total = result['total_comparisons']
                        if total > 0:
                            win_rate = result['wins_alg2'] / total * 100
                            win_rate_matrix[i, j] = win_rate
        
        # Create heatmap
        # Only show colorbar for the first subplot in each row
        # show_cbar = (idx % 3 == 0)
        # show_cbar = (idx % 3 == 0)
        show_cbar = False
        sns.heatmap(win_rate_matrix, 
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlBu',
                   vmin=0, 
                   vmax=100,
                   cbar=show_cbar,
                   cbar_kws={'label': 'Win Rate (%)'} if show_cbar else None,
                   xticklabels=algorithmsLabels,
                   yticklabels=algorithmsLabels,
                   ax=ax)
        
        ax.set_title(f'{metric} - Win Rates (%)', fontsize=18)
        # ax.set_xlabel('Algorithm 2')
        # ax.set_ylabel('Algorithm 1')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Remove extra subplot if not used
    if len(metrics) < len(axes):
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def create_win_rate_pvalue_effectsize_heatmap(individual_results, results, metrics, algorithms, output_file='plots/win_rate_pvalue_effectsize_heatmap.png'):
    """Create a heatmap showing win rate, p-value, and effect size for each algorithm pair"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        n_algs = len(algorithms)
        win_rate_matrix = np.full((n_algs, n_algs), np.nan)
        pvalue_matrix = np.full((n_algs, n_algs), np.nan)
        effectsize_matrix = np.full((n_algs, n_algs), np.nan)
        annot_matrix = np.empty((n_algs, n_algs), dtype=object)
        annot_matrix[:] = ''

        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i == j:
                    continue
                # Win rate
                win_rate = None
                comparison = f"{alg1}_vs_{alg2}"
                reverse_comparison = f"{alg2}_vs_{alg1}"
                if comparison in individual_results[metric]:
                    result = individual_results[metric][comparison]
                    total = result['total_comparisons']
                    if total > 0:
                        win_rate = result['wins_alg1'] / total * 100
                elif reverse_comparison in individual_results[metric]:
                    result = individual_results[metric][reverse_comparison]
                    total = result['total_comparisons']
                    if total > 0:
                        win_rate = result['wins_alg2'] / total * 100
                # p-value and effect size
                p_value = None
                effect_size = None
                if comparison in results[metric]:
                    res = results[metric][comparison]
                    if 'error' not in res:
                        p_value = res['p_value']
                        effect_size = res['effect_size']
                elif reverse_comparison in results[metric]:
                    res = results[metric][reverse_comparison]
                    if 'error' not in res:
                        p_value = res['p_value']
                        effect_size = res['effect_size']
                # Fill matrices
                if win_rate is not None:
                    win_rate_matrix[i, j] = win_rate
                if p_value is not None:
                    pvalue_matrix[i, j] = p_value
                if effect_size is not None:
                    effectsize_matrix[i, j] = effect_size
                # Annotation
                if win_rate is not None and p_value is not None and effect_size is not None:
                    annot_matrix[i, j] = f"{win_rate:.1f}%\np={p_value:.3g}\nr={effect_size:.2f}"
                elif win_rate is not None:
                    annot_matrix[i, j] = f"{win_rate:.1f}%"
                elif p_value is not None:
                    annot_matrix[i, j] = f"p={p_value:.3g}"
                elif effect_size is not None:
                    annot_matrix[i, j] = f"r={effect_size:.2f}"
                else:
                    annot_matrix[i, j] = ""
        # Plot heatmap (color by win rate)
        sns.heatmap(win_rate_matrix, annot=annot_matrix, fmt='', cmap='RdYlBu', vmin=0, vmax=100,
                    cbar_kws={'label': 'Win Rate (%)'}, xticklabels=algorithms, yticklabels=algorithms, ax=ax, linewidths=0.5, linecolor='gray')
        ax.set_title(f'{metric} - Win Rate / p-value / Effect Size')
        ax.set_xlabel('Algorithm 2')
        ax.set_ylabel('Algorithm 1')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def create_win_rate_pvalue_effectsize_latex(individual_results, results, metrics, algorithms, output_dir='plots/'):
    """Create a LaTeX table for each metric showing win rate, p-value, and effect size for each algorithm pair"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for metric in metrics:
        n_algs = len(algorithms)
        # Build table header
        header = ' & ' + ' & '.join(algorithms) + ' \\\n'
        header = '\\toprule\nAlgorithm ' + header + '\\midrule\n'
        rows = []
        for i, alg1 in enumerate(algorithms):
            row = [alg1]
            for j, alg2 in enumerate(algorithms):
                if i == j:
                    row.append('--')
                    continue
                # Win rate
                win_rate = None
                comparison = f"{alg1}_vs_{alg2}"
                reverse_comparison = f"{alg2}_vs_{alg1}"
                if comparison in individual_results[metric]:
                    result = individual_results[metric][comparison]
                    total = result['total_comparisons']
                    if total > 0:
                        win_rate = result['wins_alg1'] / total * 100
                elif reverse_comparison in individual_results[metric]:
                    result = individual_results[metric][reverse_comparison]
                    total = result['total_comparisons']
                    if total > 0:
                        win_rate = result['wins_alg2'] / total * 100
                # p-value and effect size
                p_value = None
                effect_size = None
                if comparison in results[metric]:
                    res = results[metric][comparison]
                    if 'error' not in res:
                        p_value = res['p_value']
                        effect_size = res['effect_size']
                elif reverse_comparison in results[metric]:
                    res = results[metric][reverse_comparison]
                    if 'error' not in res:
                        p_value = res['p_value']
                        effect_size = res['effect_size']
                # Format cell
                if win_rate is not None and p_value is not None and effect_size is not None:
                    cell = f"{win_rate:.1f}\\% / p={p_value:.3g} / r={effect_size:.2f}"
                elif win_rate is not None:
                    cell = f"{win_rate:.1f}\\%"
                elif p_value is not None:
                    cell = f"p={p_value:.3g}"
                elif effect_size is not None:
                    cell = f"r={effect_size:.2f}"
                else:
                    cell = ""
                row.append(cell)
            rows.append(' & '.join(row) + ' \\\n')
        # Compose LaTeX table
        table = (
            '\\begin{table}[ht]\n'
            '\\centering\n'
            f'\\caption{{Win rate, p-value, and effect size for {metric}}}\n'
            f'\\label{{tab:winrate_{metric}}}\n'
            '\\begin{tabular}{l' + 'c'*n_algs + '}\n'
            '\\toprule\n'
            + header +
            ''.join(rows) +
            '\\bottomrule\n'
            '\\end{tabular}\n'
            '\\end{table}\n'
        )
        # Write to file
        tex_file = os.path.join(output_dir, f'win_rate_pvalue_effectsize_{metric}.tex')
        with open(tex_file, 'w') as f:
            f.write(table)
        print(f"LaTeX table for {metric} saved to {tex_file}")

def create_seed_best_algorithm_summary(df, metrics, algorithms, output_file='seed_best_algorithm_summary.csv'):
    """
    Create a summary showing which algorithm is best for each seed across all metrics
    
    Args:
        df: DataFrame with metrics data
        metrics: List of metric names to test
        algorithms: List of algorithm names
        output_file: Output CSV file name
    
    Returns:
        DataFrame with best algorithm per seed per metric
    """
    # Get final generation data for each algorithm and seed
    final_gen_data = df.groupby(['Algorithm', 'Seed'])['Generation'].max().reset_index()
    final_data = df.merge(final_gen_data, on=['Algorithm', 'Seed', 'Generation'])
    
    summary_data = []
    
    # Get all unique seeds
    all_seeds = final_data['Seed'].unique()
    
    for seed in all_seeds:
        seed_data = final_data[final_data['Seed'] == seed]
        
        for metric in metrics:
            metric_data = seed_data[['Algorithm', metric]].copy()
            
            if len(metric_data) > 0:
                # Find the best algorithm for this metric and seed
                if metric in ['GD', 'IGD', 'S', 'STE']:
                    # Lower is better
                    best_idx = metric_data[metric].idxmin()
                    best_algorithm = metric_data.loc[best_idx, 'Algorithm']
                    best_value = metric_data.loc[best_idx, metric]
                    worst_idx = metric_data[metric].idxmax()
                    worst_algorithm = metric_data.loc[worst_idx, 'Algorithm']
                    worst_value = metric_data.loc[worst_idx, metric]
                else:  # HV
                    # Higher is better
                    best_idx = metric_data[metric].idxmax()
                    best_algorithm = metric_data.loc[best_idx, 'Algorithm']
                    best_value = metric_data.loc[best_idx, metric]
                    worst_idx = metric_data[metric].idxmin()
                    worst_algorithm = metric_data.loc[worst_idx, 'Algorithm']
                    worst_value = metric_data.loc[worst_idx, metric]
                
                # Get all values for this metric and seed
                all_values = metric_data[metric].values
                all_algorithms = metric_data['Algorithm'].values
                
                summary_data.append({
                    'Seed': seed,
                    'Metric': metric,
                    'Best_Algorithm': best_algorithm,
                    'Best_Value': best_value,
                    'Worst_Algorithm': worst_algorithm,
                    'Worst_Value': worst_value,
                    'Value_Range': worst_value - best_value,
                    'Num_Algorithms': len(all_values),
                    'All_Algorithms': ', '.join(all_algorithms),
                    'All_Values': ', '.join([f"{alg}: {val:.8f}" for alg, val in zip(all_algorithms, all_values)])
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Seed best algorithm summary saved to {output_file}")
    
    # Print summary statistics
    print(f"\n=== SEED BEST ALGORITHM SUMMARY ===")
    for metric in metrics:
        metric_summary = summary_df[summary_df['Metric'] == metric]
        print(f"\n{metric}:")
        
        # Count wins per algorithm
        win_counts = metric_summary['Best_Algorithm'].value_counts()
        total_seeds = len(metric_summary)
        
        for alg, wins in win_counts.items():
            win_rate = wins / total_seeds * 100
            print(f"  {alg}: {wins}/{total_seeds} wins ({win_rate:.1f}%)")
    
    return summary_df

def create_algorithm_dominance_matrix(df, metrics, algorithms, output_file='algorithm_dominance_matrix.csv'):
    """
    Create a dominance matrix showing how often each algorithm wins against others
    
    Args:
        df: DataFrame with metrics data
        metrics: List of metric names to test
        algorithms: List of algorithm names
        output_file: Output CSV file name
    
    Returns:
        DataFrame with dominance matrix
    """
    # Get final generation data
    final_gen_data = df.groupby(['Algorithm', 'Seed'])['Generation'].max().reset_index()
    final_data = df.merge(final_gen_data, on=['Algorithm', 'Seed', 'Generation'])
    
    dominance_data = []
    
    for metric in metrics:
        for alg1 in algorithms:
            for alg2 in algorithms:
                if alg1 != alg2:
                    # Get data for both algorithms
                    data1 = final_data[final_data['Algorithm'] == alg1][['Seed', metric]].copy()
                    data2 = final_data[final_data['Algorithm'] == alg2][['Seed', metric]].copy()
                    
                    # Find common seeds
                    common_seeds = set(data1['Seed'].unique()) & set(data2['Seed'].unique())
                    
                    wins_alg1 = 0
                    wins_alg2 = 0
                    ties = 0
                    
                    for seed in common_seeds:
                        val1 = data1[data1['Seed'] == seed][metric].iloc[0]
                        val2 = data2[data2['Seed'] == seed][metric].iloc[0]
                        
                        # Determine winner
                        if metric in ['GD', 'IGD', 'S', 'STE']:
                            if val1 < val2:
                                wins_alg1 += 1
                            elif val2 < val1:
                                wins_alg2 += 1
                            else:
                                ties += 1
                        else:  # HV
                            if val1 > val2:
                                wins_alg1 += 1
                            elif val2 > val1:
                                wins_alg2 += 1
                            else:
                                ties += 1
                    
                    total = wins_alg1 + wins_alg2 + ties
                    if total > 0:
                        dominance_data.append({
                            'Metric': metric,
                            'Algorithm_1': alg1,
                            'Algorithm_2': alg2,
                            'Wins_Alg1': wins_alg1,
                            'Wins_Alg2': wins_alg2,
                            'Ties': ties,
                            'Total_Comparisons': total,
                            'Win_Rate_Alg1_%': wins_alg1 / total * 100,
                            'Win_Rate_Alg2_%': wins_alg2 / total * 100,
                            'Tie_Rate_%': ties / total * 100,
                            'Dominant': alg1 if wins_alg1 > wins_alg2 else alg2 if wins_alg2 > wins_alg1 else 'Tie'
                        })
    
    dominance_df = pd.DataFrame(dominance_data)
    dominance_df.to_csv(output_file, index=False)
    print(f"Algorithm dominance matrix saved to {output_file}")
    
    return dominance_df

def load_metrics_data(standard_file, hybrid_file):
    """Load both standard and hybrid metrics data"""

    dataframes = []
    
    if os.path.exists(standard_file):
        df_standard = pd.read_csv(standard_file)
        df_standard.drop(columns=["TimeDelta"],inplace=True)
        # dataframes.append(df_standard)
        # print(f"Loaded standard data: {len(df_standard)} rows")
    
    if os.path.exists(hybrid_file):
        df_hybrid = pd.read_csv(hybrid_file)
        # dataframes.append(df_hybrid)
        # print(f"Loaded hybrid data: {len(df_hybrid)} rows")
 
    # Combine all data


    ##### NOTE: Continuos testing using only partial results
    available_seeds = df_hybrid["Seed"].unique()
    df_standard = df_standard[df_standard["Seed"].isin(available_seeds)]
    dataframes.append(df_standard)
    dataframes.append(df_hybrid)
    ###
   
    
    if not dataframes:
        raise FileNotFoundError("No metrics data files found")
    
    df_combined = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data: {len(df_combined)} rows")
    
    return df_combined


def main():

    config = load_bash_config('script_constants.sh')
    # path_results = "results_100/"
    # output_path = "results_100/"
    # N_EXECUTIONS = 1
    path_results = "results/"
    output_path = "results/"
    N_EXECUTIONS = 30

    path_results = "results_imperium/"
    output_path = "results_imperium/"
    N_EXECUTIONS = 1

    path_results = "results_hybrid_400_500/"
    output_path = "results_hybrid_400_500/"
    N_EXECUTIONS = 3

    path_results = "results_longest/"
    output_path = "results_longest/"
    N_EXECUTIONS = 30

    # Load standard algorithms data
    standard_file = path_results+f"table_standard_{config['POP_SIZE']}_{config['N_GEN']}.csv" #Note: they are computed until CUT_GENERATION
    hybrid_file = path_results+f"table_hybrids_{config['HYBRID_POP_SIZE']}_{config['HYBRID_N_GEN']}.csv"
    

    """Main function to perform statistical analysis"""
    print("Loading metrics data...")
    df = load_metrics_data(standard_file, hybrid_file)
    
    # Define metrics and algorithms
    metrics = ['GD', 'IGD', 'HV', 'S', 'STE']
    algorithms = config['ALGORITHMS'] + ['Hybrids']  # Include hybrid algorithms
    
    print(f"Metrics to analyze: {metrics}")
    print(f"Algorithms to compare: {algorithms}")
    
    # Perform Wilcoxon tests on final generation
    print("\nPerforming Wilcoxon rank sum tests on final generation...")
    results = perform_wilcoxon_tests(df, metrics, algorithms, alpha=0.05)
    
    # Create summary table for final generation
    print("\nCreating summary table for final generation...")
    summary_df = create_summary_table(results, metrics, algorithms)
    
    # Save final generation results
    summary_df.to_csv(output_path+'wilcoxon_test_results_final_generation.csv', index=False)
    print(f"Final generation results saved to wilcoxon_test_results_final_generation.csv")
    
    # Print significant results for final generation
    print("\n=== FINAL GENERATION - SIGNIFICANT RESULTS (p < 0.05) ===")
    significant_results = summary_df[summary_df['Significant'] == True]
    if len(significant_results) > 0:
        for _, row in significant_results.iterrows():
            print(f"{row['Metric']}: {row['Algorithm_1']} vs {row['Algorithm_2']}")
            print(f"  p-value: {row['P_Value']:.8f}")
            print(f"  effect size: {row['Effect_Size']:.8f}")
            print(f"  means: {row['Mean_1']:.8f} vs {row['Mean_2']:.8f}")
            print()
    else:
        print("No significant differences found in final generation.")
    
    # Analyze convergence speed
    print("\nAnalyzing convergence speed...")
    convergence_results = analyze_convergence_speed(df, metrics, algorithms, alpha=0.05)
    
    # Create summary table for convergence analysis
    print("\nCreating summary table for convergence analysis...")
    convergence_summary_df = create_summary_table(convergence_results, metrics, algorithms)
    
    # Save convergence results
    convergence_summary_df.to_csv(output_path+'wilcoxon_test_results_convergence.csv', index=False)
    print(f"Convergence results saved to wilcoxon_test_results_convergence.csv")
    
    # Print significant convergence results
    print("\n=== CONVERGENCE SPEED - SIGNIFICANT RESULTS (p < 0.05) ===")
    significant_convergence = convergence_summary_df[convergence_summary_df['Significant'] == True]
    if len(significant_convergence) > 0:
        for _, row in significant_convergence.iterrows():
            print(f"{row['Metric']}: {row['Algorithm_1']} vs {row['Algorithm_2']}")
            print(f"  p-value: {row['P_Value']:.8f}")
            print(f"  effect size: {row['Effect_Size']:.8f}")
            print(f"  avg convergence (generations): {row['Mean_1']:.8f} vs {row['Mean_2']:.8f}")
            print()
    else:
        print("No significant differences found in convergence speed.")
    
    # Analyze individual performance
    print("\nAnalyzing individual performance...")
    individual_results = analyze_individual_performance(df, metrics, algorithms)
    
    # Create individual performance table
    print("\nCreating individual performance table...")
    individual_summary_df = create_individual_performance_table(individual_results, metrics, algorithms)
    
    # Save individual performance results
    individual_summary_df.to_csv(output_path+'individual_performance_summary.csv', index=False)
    print(f"Individual performance results saved to individual_performance_summary.csv")
    
    # Create detailed seed analysis
    print("\nCreating detailed seed analysis...")
    create_detailed_seed_analysis(individual_results, metrics, algorithms)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_heatmap(results, metrics, algorithms, output_path+'plots/wilcoxon_heatmap_final_generation.png')
    create_boxplots(df, metrics, algorithms, output_path+'plots/metrics_boxplots_final_generation.png')
    create_convergence_plot(df, metrics, algorithms, output_path+'plots/convergence_analysis.png')
    create_win_rate_heatmap(individual_results, metrics, algorithms, output_path+'plots/win_rate_heatmap.png')
    # create_win_rate_pvalue_effectsize_heatmap(individual_results, results, metrics, algorithms, output_path+'plots/win_rate_pvalue_effectsize_heatmap.png')
    # create_win_rate_pvalue_effectsize_latex(individual_results, results, metrics, algorithms, output_path+'plots/')
    
    # Print overall summary statistics
    print("\n=== OVERALL SUMMARY STATISTICS ===")
    print(f"Final generation comparisons: {len(summary_df)}")
    print(f"Final generation significant: {len(significant_results)}")
    print(f"Final generation significance rate: {len(significant_results)/len(summary_df)*100:.1f}%")
    print(f"Convergence comparisons: {len(convergence_summary_df)}")
    print(f"Convergence significant: {len(significant_convergence)}")
    print(f"Convergence significance rate: {len(significant_convergence)/len(convergence_summary_df)*100:.1f}%")
    
    # Create seed best algorithm summary
    print("\nCreating seed best algorithm summary...")
    seed_summary_df = create_seed_best_algorithm_summary(df, metrics, algorithms)
    
    # Create algorithm dominance matrix
    print("\nCreating algorithm dominance matrix...")
    dominance_df = create_algorithm_dominance_matrix(df, metrics, algorithms)
    
    return results, summary_df, convergence_results, convergence_summary_df, individual_results, individual_summary_df, seed_summary_df, dominance_df

if __name__ == "__main__":
    main() 