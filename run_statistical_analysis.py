#!/usr/bin/env python3
"""
Statistical Analysis Runner Script

This script runs Wilcoxon rank sum tests on GD, IGD, HV, S, and STE metrics
for all competing algorithms with a significance level of 0.05.

Usage:
    python run_statistical_analysis.py [--detailed] [--simple]

Options:
    --detailed: Run the comprehensive analysis with multiple testing correction
    --simple: Run the basic analysis without multiple testing correction
"""

import sys
import argparse
from statistical_analysis_detailed import main as run_detailed_analysis
from wilcoxon_statistical_analysis import main as run_simple_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on genetic algorithm metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_statistical_analysis.py --detailed    # Run comprehensive analysis
    python run_statistical_analysis.py --simple      # Run basic analysis
    python run_statistical_analysis.py               # Run detailed analysis (default)
        """
    )
    
    parser.add_argument(
        '--detailed', 
        action='store_true',
        help='Run comprehensive analysis with multiple testing correction (default)'
    )
    
    parser.add_argument(
        '--simple', 
        action='store_true',
        help='Run basic analysis without multiple testing correction'
    )
    
    args = parser.parse_args()
    
    # Default to detailed analysis if no option specified
    if not args.simple:
        print("Running comprehensive statistical analysis...")
        print("This includes:")
        print("- Wilcoxon rank sum tests for all algorithm pairs")
        print("- Multiple testing correction (Benjamini-Hochberg)")
        print("- Friedman tests for overall differences")
        print("- Comprehensive reporting and visualization")
        print()
        
        try:
            results, friedman_results = run_detailed_analysis()
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Output files created:")
            print("- statistical_analysis_detailed_results.csv")
            print("- statistical_analysis_significant_results.csv")
            print("- statistical_analysis_summary.csv")
            print("- statistical_analysis_friedman_results.csv")
            print("- wilcoxon_heatmap.png")
            print("- metrics_boxplots.png")
            
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            sys.exit(1)
    
    else:
        print("Running simple statistical analysis...")
        print("This includes:")
        print("- Wilcoxon rank sum tests for all algorithm pairs")
        print("- Basic reporting and visualization")
        print()
        
        try:
            results, summary_df = run_simple_analysis()
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Output files created:")
            print("- wilcoxon_test_results.csv")
            print("- wilcoxon_heatmap.png")
            print("- metrics_boxplots.png")
            
        except Exception as e:
            print(f"Error in simple analysis: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 