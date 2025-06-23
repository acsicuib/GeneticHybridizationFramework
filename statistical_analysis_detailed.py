import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.stats import rankdata
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_bash_config
import os
from statsmodels.stats.multitest import multipletests

def load_and_prepare_data():
    """Load and prepare data for statistical analysis"""
    config = load_bash_config('script_constants.sh')
    
    # Load standard algorithms data
    standard_file = f"results/table_standard_{config['POP_SIZE']}_{config['N_GEN']}.csv"
    hybrid_file = f"results/table_hybrids_{config['HYBRID_POP_SIZE']}_{config['HYBRID_N_GEN']}.csv"
    
    dataframes = []
    
    if os.path.exists(standard_file):
        df_standard = pd.read_csv(standard_file)
        dataframes.append(df_standard)
        print(f"✓ Loaded standard data: {len(df_standard)} rows")
    else:
        print(f"✗ Standard data file not found: {standard_file}")
    
    if os.path.exists(hybrid_file):
        df_hybrid = pd.read_csv(hybrid_file)
        dataframes.append(df_hybrid)
        print(f"✓ Loaded hybrid data: {len(df_hybrid)} rows")
    else:
        print(f"✗ Hybrid data file not found: {hybrid_file}")
    
    if not dataframes:
        raise FileNotFoundError("No metrics data files found")
    
    # Combine all data
    df_combined = pd.concat(dataframes, ignore_index=True)
    print(f"✓ Combined data: {len(df_combined)} rows")
    
    return df_combined, config

def get_final_generation_data(df):
    """Extract final generation data for each algorithm and seed"""
    # Get the final generation for each algorithm and seed
    final_gen_data = df.groupby(['Algorithm', 'Seed'])['Generation'].max().reset_index()
    
    # Merge to get only final generation data
    final_data = df.merge(final_gen_data, on=['Algorithm', 'Seed', 'Generation'])
    
    print(f"✓ Final generation data: {len(final_data)} rows")
    print(f"✓ Algorithms found: {sorted(final_data['Algorithm'].unique())}")
    print(f"✓ Seeds found: {sorted(final_data['Seed'].unique())}")
    
    return final_data

def perform_wilcoxon_pairwise_tests(final_data, metrics, algorithms, alpha=0.05):
    """
    Perform pairwise Wilcoxon rank sum tests with multiple testing correction
    """
    print(f"\n{'='*60}")
    print("PERFORMING WILCOXON RANK SUM TESTS")
    print(f"{'='*60}")
    
    all_results = []
    all_p_values = []
    
    for metric in metrics:
        print(f"\n📊 Analyzing metric: {metric}")
        print("-" * 40)
        
        # Get all unique algorithm pairs
        algorithm_pairs = list(combinations(algorithms, 2))
        
        for alg1, alg2 in algorithm_pairs:
            # Get data for both algorithms
            data1 = final_data[final_data['Algorithm'] == alg1][metric].values
            data2 = final_data[final_data['Algorithm'] == alg2][metric].values
            
            # Remove NaN values
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]
            
            if len(data1) < 3 or len(data2) < 3:
                print(f"⚠️  Insufficient data for {alg1} vs {alg2} in {metric} (n1={len(data1)}, n2={len(data2)})")
                continue
            
            # Perform Wilcoxon rank sum test
            try:
                statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
                
                # Calculate effect size (rank-biserial correlation)
                n1, n2 = len(data1), len(data2)
                effect_size = 1 - (2 * statistic) / (n1 * n2)
                
                # Determine which algorithm is better (lower is better for GD, IGD, S, STE; higher is better for HV)
                mean1, mean2 = np.mean(data1), np.mean(data2)
                if metric in ['HV']:  # Higher is better
                    better_alg = alg1 if mean1 > mean2 else alg2
                    better_mean = max(mean1, mean2)
                    worse_mean = min(mean1, mean2)
                else:  # Lower is better (GD, IGD, S, STE)
                    better_alg = alg1 if mean1 < mean2 else alg2
                    better_mean = min(mean1, mean2)
                    worse_mean = max(mean1, mean2)
                
                result = {
                    'Metric': metric,
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Statistic': statistic,
                    'P_Value': p_value,
                    'Effect_Size': effect_size,
                    'Mean_1': mean1,
                    'Mean_2': mean2,
                    'Std_1': np.std(data1),
                    'Std_2': np.std(data2),
                    'N_1': n1,
                    'N_2': n2,
                    'Better_Algorithm': better_alg,
                    'Better_Mean': better_mean,
                    'Worse_Mean': worse_mean,
                    'Difference': abs(mean1 - mean2)
                }
                
                all_results.append(result)
                all_p_values.append(p_value)
                
                # Print result
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {alg1} vs {alg2}: p={p_value:.6f}{significance}")
                print(f"    Effect size: {effect_size:.4f}, Better: {better_alg} ({better_mean:.4f} vs {worse_mean:.4f})")
                
            except Exception as e:
                print(f"❌ Error in Wilcoxon test for {alg1} vs {alg2} in {metric}: {e}")
    
    # Apply multiple testing correction
    if all_p_values:
        print(f"\n🔧 Applying multiple testing correction (Benjamini-Hochberg)...")
        rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
        
        # Update results with corrected p-values
        for i, result in enumerate(all_results):
            result['P_Value_Corrected'] = p_corrected[i]
            result['Significant_Raw'] = result['P_Value'] < alpha
            result['Significant_Corrected'] = rejected[i]
        
        print(f"  Raw significant tests: {sum([r['Significant_Raw'] for r in all_results])}")
        print(f"  Corrected significant tests: {sum([r['Significant_Corrected'] for r in all_results])}")
    
    return all_results

def perform_friedman_test(final_data, metrics, algorithms):
    """
    Perform Friedman test to check if there are significant differences across all algorithms
    """
    print(f"\n{'='*60}")
    print("PERFORMING FRIEDMAN TEST")
    print(f"{'='*60}")
    
    friedman_results = {}
    
    for metric in metrics:
        print(f"\n📊 Friedman test for metric: {metric}")
        
        # Prepare data for Friedman test (each row is a seed, each column is an algorithm)
        friedman_data = []
        valid_seeds = []
        
        for seed in sorted(final_data['Seed'].unique()):
            seed_data = []
            for alg in algorithms:
                alg_data = final_data[(final_data['Algorithm'] == alg) & (final_data['Seed'] == seed)][metric].values
                if len(alg_data) > 0 and not np.isnan(alg_data[0]):
                    seed_data.append(alg_data[0])
                else:
                    seed_data.append(np.nan)
            
            # Only include seeds with complete data
            if not any(np.isnan(seed_data)):
                friedman_data.append(seed_data)
                valid_seeds.append(seed)
        
        if len(friedman_data) < 3:
            print(f"⚠️  Insufficient data for Friedman test in {metric}")
            continue
        
        friedman_data = np.array(friedman_data)
        
        try:
            statistic, p_value = friedmanchisquare(*friedman_data.T)
            
            friedman_results[metric] = {
                'statistic': statistic,
                'p_value': p_value,
                'n_seeds': len(valid_seeds),
                'n_algorithms': len(algorithms)
            }
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  Friedman statistic: {statistic:.4f}, p-value: {p_value:.6f}{significance}")
            print(f"  Sample size: {len(valid_seeds)} seeds, {len(algorithms)} algorithms")
            
        except Exception as e:
            print(f"❌ Error in Friedman test for {metric}: {e}")
    
    return friedman_results

def create_comprehensive_report(results, friedman_results, metrics, algorithms, alpha=0.05):
    """Create a comprehensive statistical report"""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE STATISTICAL REPORT")
    print(f"{'='*60}")
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"  Total pairwise comparisons: {len(df_results)}")
    print(f"  Raw significant comparisons (p < {alpha}): {sum(df_results['Significant_Raw'])}")
    print(f"  Corrected significant comparisons: {sum(df_results['Significant_Corrected'])}")
    print(f"  Significance rate (raw): {sum(df_results['Significant_Raw'])/len(df_results)*100:.1f}%")
    print(f"  Significance rate (corrected): {sum(df_results['Significant_Corrected'])/len(df_results)*100:.1f}%")
    
    # Significant results summary
    print(f"\n🎯 SIGNIFICANT RESULTS (after correction):")
    significant_results = df_results[df_results['Significant_Corrected'] == True]
    
    if len(significant_results) > 0:
        for _, row in significant_results.iterrows():
            print(f"  {row['Metric']}: {row['Algorithm_1']} vs {row['Algorithm_2']}")
            print(f"    p-value (raw): {row['P_Value']:.6f}")
            print(f"    p-value (corrected): {row['P_Value_Corrected']:.6f}")
            print(f"    effect size: {row['Effect_Size']:.4f}")
            print(f"    better algorithm: {row['Better_Algorithm']} ({row['Better_Mean']:.4f} vs {row['Worse_Mean']:.4f})")
            print()
    else:
        print("  No significant differences found after multiple testing correction.")
    
    # Friedman test summary
    print(f"\n🔍 FRIEDMAN TEST RESULTS:")
    for metric, result in friedman_results.items():
        significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        print(f"  {metric}: χ² = {result['statistic']:.4f}, p = {result['p_value']:.6f}{significance}")
    
    return df_results

def save_results(df_results, friedman_results, output_prefix="statistical_analysis"):
    """Save all results to files"""
    # Save detailed results
    df_results.to_csv(f'{output_prefix}_detailed_results.csv', index=False)
    print(f"✓ Detailed results saved to {output_prefix}_detailed_results.csv")
    
    # Save significant results only
    significant_results = df_results[df_results['Significant_Corrected'] == True]
    if len(significant_results) > 0:
        significant_results.to_csv(f'{output_prefix}_significant_results.csv', index=False)
        print(f"✓ Significant results saved to {output_prefix}_significant_results.csv")
    
    # Save summary statistics
    summary_stats = {
        'Total_Comparisons': len(df_results),
        'Raw_Significant': sum(df_results['Significant_Raw']),
        'Corrected_Significant': sum(df_results['Significant_Corrected']),
        'Raw_Significance_Rate': sum(df_results['Significant_Raw'])/len(df_results)*100,
        'Corrected_Significance_Rate': sum(df_results['Significant_Corrected'])/len(df_results)*100
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
    print(f"✓ Summary statistics saved to {output_prefix}_summary.csv")
    
    # Save Friedman results
    if friedman_results:
        friedman_df = pd.DataFrame(friedman_results).T
        friedman_df.to_csv(f'{output_prefix}_friedman_results.csv')
        print(f"✓ Friedman test results saved to {output_prefix}_friedman_results.csv")

def main():
    """Main function to perform comprehensive statistical analysis"""
    print("🚀 Starting Comprehensive Statistical Analysis")
    print("=" * 60)
    
    # Load and prepare data
    df, config = load_and_prepare_data()
    
    # Define metrics and algorithms
    metrics = ['GD', 'IGD', 'HV', 'S', 'STE']
    algorithms = config['ALGORITHMS'] + ['Hybrids']
    
    print(f"\n📋 Analysis Configuration:")
    print(f"  Metrics: {metrics}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Significance level: α = 0.05")
    
    # Get final generation data
    final_data = get_final_generation_data(df)
    
    # Perform statistical tests
    results = perform_wilcoxon_pairwise_tests(final_data, metrics, algorithms, alpha=0.05)
    friedman_results = perform_friedman_test(final_data, metrics, algorithms)
    
    # Create comprehensive report
    df_results = create_comprehensive_report(results, friedman_results, metrics, algorithms)
    
    # Save results
    save_results(df_results, friedman_results)
    
    print(f"\n✅ Statistical analysis completed successfully!")
    print(f"📁 Results saved to CSV files with prefix 'statistical_analysis_'")
    
    return df_results, friedman_results

if __name__ == "__main__":
    main() 