import pandas as pd
import numpy as np

def analyze_metrics():
    # Load the data
    df = pd.read_csv('metrics_results.csv')
    
    # Create output string to store our report
    report = []
    
    report.append("# Metrics Analysis Report\n")
    
    # 1. Overall statistics
    report.append("## 1. Overall Statistics\n")
    report.append("### Faithfulness")
    report.append(f"  - Mean: {df['faithfulness'].mean():.4f}")
    report.append(f"  - Median: {df['faithfulness'].median():.4f}")
    report.append(f"  - Min: {df['faithfulness'].min():.4f}")
    report.append(f"  - Max: {df['faithfulness'].max():.4f}")
    report.append(f"  - Standard Deviation: {df['faithfulness'].std():.4f}\n")
    
    report.append("### Relevancy")
    report.append(f"  - Mean: {df['relevancy'].mean():.4f}")
    report.append(f"  - Median: {df['relevancy'].median():.4f}")
    report.append(f"  - Min: {df['relevancy'].min():.4f}")
    report.append(f"  - Max: {df['relevancy'].max():.4f}")
    report.append(f"  - Standard Deviation: {df['relevancy'].std():.4f}\n")
    
    # 2. Best parameter combinations
    report.append("## 2. Best Parameter Combinations\n")
    
    # Group by parameter combinations and calculate mean metrics
    param_groups = df.groupby(['chunk_size', 'chunk_overlap', 'k']).agg({
        'faithfulness': 'mean', 
        'relevancy': 'mean'
    }).reset_index()
    
    # Best for faithfulness
    best_faith = param_groups.loc[param_groups['faithfulness'].idxmax()]
    report.append("### Best for Faithfulness")
    report.append(f"  - Chunk Size: {best_faith['chunk_size']}")
    report.append(f"  - Chunk Overlap: {best_faith['chunk_overlap']}")
    report.append(f"  - k: {best_faith['k']}")
    report.append(f"  - Faithfulness Score: {best_faith['faithfulness']:.4f}")
    report.append(f"  - Corresponding Relevancy: {best_faith['relevancy']:.4f}\n")
    
    # Best for relevancy
    best_rel = param_groups.loc[param_groups['relevancy'].idxmax()]
    report.append("### Best for Relevancy")
    report.append(f"  - Chunk Size: {best_rel['chunk_size']}")
    report.append(f"  - Chunk Overlap: {best_rel['chunk_overlap']}")
    report.append(f"  - k: {best_rel['k']}")
    report.append(f"  - Relevancy Score: {best_rel['relevancy']:.4f}")
    report.append(f"  - Corresponding Faithfulness: {best_rel['faithfulness']:.4f}\n")
    
    # Best balanced (average of faithfulness and relevancy)
    param_groups['balanced'] = (param_groups['faithfulness'] + param_groups['relevancy']) / 2
    best_balanced = param_groups.loc[param_groups['balanced'].idxmax()]
    report.append("### Best Balanced (Average of Both Metrics)")
    report.append(f"  - Chunk Size: {best_balanced['chunk_size']}")
    report.append(f"  - Chunk Overlap: {best_balanced['chunk_overlap']}")
    report.append(f"  - k: {best_balanced['k']}")
    report.append(f"  - Faithfulness Score: {best_balanced['faithfulness']:.4f}")
    report.append(f"  - Relevancy Score: {best_balanced['relevancy']:.4f}")
    report.append(f"  - Balanced Score: {best_balanced['balanced']:.4f}\n")
    
    # 3. Parameter effects analysis
    report.append("## 3. Parameter Effects Analysis\n")
    
    # Effect of chunk_size
    chunk_size_effect = df.groupby('chunk_size').agg({
        'faithfulness': ['mean', 'std'],
        'relevancy': ['mean', 'std']
    })
    
    report.append("### Effect of Chunk Size")
    for size in sorted(df['chunk_size'].unique()):
        faith_mean = chunk_size_effect.loc[size, ('faithfulness', 'mean')]
        faith_std = chunk_size_effect.loc[size, ('faithfulness', 'std')]
        rel_mean = chunk_size_effect.loc[size, ('relevancy', 'mean')]
        rel_std = chunk_size_effect.loc[size, ('relevancy', 'std')]
        
        report.append(f"  - Chunk Size {size}:")
        report.append(f"    * Faithfulness: {faith_mean:.4f} ± {faith_std:.4f}")
        report.append(f"    * Relevancy: {rel_mean:.4f} ± {rel_std:.4f}")
    report.append("")
    
    # Effect of chunk_overlap
    chunk_overlap_effect = df.groupby('chunk_overlap').agg({
        'faithfulness': ['mean', 'std'],
        'relevancy': ['mean', 'std']
    })
    
    report.append("### Effect of Chunk Overlap")
    for overlap in sorted(df['chunk_overlap'].unique()):
        faith_mean = chunk_overlap_effect.loc[overlap, ('faithfulness', 'mean')]
        faith_std = chunk_overlap_effect.loc[overlap, ('faithfulness', 'std')]
        rel_mean = chunk_overlap_effect.loc[overlap, ('relevancy', 'mean')]
        rel_std = chunk_overlap_effect.loc[overlap, ('relevancy', 'std')]
        
        report.append(f"  - Chunk Overlap {overlap}:")
        report.append(f"    * Faithfulness: {faith_mean:.4f} ± {faith_std:.4f}")
        report.append(f"    * Relevancy: {rel_mean:.4f} ± {rel_std:.4f}")
    report.append("")
    
    # Effect of k
    k_effect = df.groupby('k').agg({
        'faithfulness': ['mean', 'std'],
        'relevancy': ['mean', 'std']
    })
    
    report.append("### Effect of k (Number of Retrieved Chunks)")
    for k_val in sorted(df['k'].unique()):
        faith_mean = k_effect.loc[k_val, ('faithfulness', 'mean')]
        faith_std = k_effect.loc[k_val, ('faithfulness', 'std')]
        rel_mean = k_effect.loc[k_val, ('relevancy', 'mean')]
        rel_std = k_effect.loc[k_val, ('relevancy', 'std')]
        
        report.append(f"  - k = {k_val}:")
        report.append(f"    * Faithfulness: {faith_mean:.4f} ± {faith_std:.4f}")
        report.append(f"    * Relevancy: {rel_mean:.4f} ± {rel_std:.4f}")
    report.append("")
    
    # 4. User-specific results
    report.append("## 4. User-specific Analysis\n")
    
    for user_id in sorted(df['user_id'].unique()):
        user_df = df[df['user_id'] == user_id]
        # Find best parameters for this user
        user_best = user_df.loc[user_df['faithfulness'].idxmax()]
        
        report.append(f"### User {user_id}")
        report.append(f"  - Average Faithfulness: {user_df['faithfulness'].mean():.4f}")
        report.append(f"  - Average Relevancy: {user_df['relevancy'].mean():.4f}")
        report.append("  - Best Parameter Combination:")
        report.append(f"    * Chunk Size: {user_best['chunk_size']}")
        report.append(f"    * Chunk Overlap: {user_best['chunk_overlap']}")
        report.append(f"    * k: {user_best['k']}")
        report.append(f"    * Faithfulness: {user_best['faithfulness']:.4f}")
        report.append(f"    * Relevancy: {user_best['relevancy']:.4f}")
        report.append("")
    
    # 5. Recommendations
    report.append("## 5. Recommendations\n")
    
    # Create recommendations based on our analysis
    report.append("### General Recommendations")
    report.append("Based on the analysis, we recommend:")
    
    # Best chunk size recommendation
    chunk_size_rel = df.groupby('chunk_size')['relevancy'].mean()
    chunk_size_faith = df.groupby('chunk_size')['faithfulness'].mean()
    best_chunk_size = df.groupby('chunk_size')[['faithfulness', 'relevancy']].mean().mean(axis=1).idxmax()
    
    report.append(f"  - **Chunk Size**: {best_chunk_size} is the overall best performing chunk size, balancing faithfulness and relevancy.")
    
    # Best chunk overlap recommendation
    best_chunk_overlap = df.groupby('chunk_overlap')[['faithfulness', 'relevancy']].mean().mean(axis=1).idxmax()
    report.append(f"  - **Chunk Overlap**: {best_chunk_overlap} provides the best balance of metrics.")
    
    # Best k recommendation
    best_k = df.groupby('k')[['faithfulness', 'relevancy']].mean().mean(axis=1).idxmax()
    report.append(f"  - **k (Retrieved Chunks)**: {best_k} retrieves the optimal number of chunks for balance between metrics.")
    
    # Faithfulness vs Relevancy tradeoff
    # Correlation between metrics
    corr = df['faithfulness'].corr(df['relevancy'])
    report.append(f"\n### Faithfulness vs. Relevancy Tradeoff")
    report.append(f"  - Correlation between metrics: {corr:.4f}")
    
    if corr < -0.2:
        report.append("  - There is a significant negative correlation between faithfulness and relevancy, indicating a tradeoff.")
        report.append("  - To prioritize faithfulness, consider using smaller chunks with less overlap.")
        report.append("  - To prioritize relevancy, consider using larger chunks with more overlap.")
    elif corr > 0.2:
        report.append("  - There is a positive correlation between faithfulness and relevancy, suggesting that optimizing for one may help the other.")
    else:
        report.append("  - There is little correlation between the metrics, suggesting they can be optimized independently.")
    
    # Write to file
    with open('metrics_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("Report generated: metrics_report.md")

if __name__ == "__main__":
    analyze_metrics() 