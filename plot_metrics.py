import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for nicer plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Load data
df = pd.read_csv('metrics_results.csv')

# Create output directory if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Plot chunk_size vs metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='chunk_size', y='faithfulness', data=df)
plt.title('Chunk Size vs Faithfulness')
plt.xlabel('Chunk Size')
plt.ylabel('Faithfulness Score')

plt.subplot(1, 2, 2)
sns.boxplot(x='chunk_size', y='relevancy', data=df)
plt.title('Chunk Size vs Relevancy')
plt.xlabel('Chunk Size')
plt.ylabel('Relevancy Score')
plt.tight_layout()
plt.savefig('plots/chunk_size_vs_metrics.png')

# 2. Plot chunk_overlap vs metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='chunk_overlap', y='faithfulness', data=df)
plt.title('Chunk Overlap vs Faithfulness')
plt.xlabel('Chunk Overlap')
plt.ylabel('Faithfulness Score')

plt.subplot(1, 2, 2)
sns.boxplot(x='chunk_overlap', y='relevancy', data=df)
plt.title('Chunk Overlap vs Relevancy')
plt.xlabel('Chunk Overlap')
plt.ylabel('Relevancy Score')
plt.tight_layout()
plt.savefig('plots/chunk_overlap_vs_metrics.png')

# 3. Plot k vs metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='k', y='faithfulness', data=df)
plt.title('k vs Faithfulness')
plt.xlabel('k (Number of Retrieved Chunks)')
plt.ylabel('Faithfulness Score')

plt.subplot(1, 2, 2)
sns.boxplot(x='k', y='relevancy', data=df)
plt.title('k vs Relevancy')
plt.xlabel('k (Number of Retrieved Chunks)')
plt.ylabel('Relevancy Score')
plt.tight_layout()
plt.savefig('plots/k_vs_metrics.png')

# 4. Heatmap of chunk_size and chunk_overlap vs metrics
# First for faithfulness
chunk_sizes = df['chunk_size'].unique()
chunk_overlaps = df['chunk_overlap'].unique()

# Create pivot tables for heatmaps
faithfulness_pivot = df.pivot_table(
    values='faithfulness', 
    index='chunk_size', 
    columns='chunk_overlap', 
    aggfunc='mean'
)

relevancy_pivot = df.pivot_table(
    values='relevancy', 
    index='chunk_size', 
    columns='chunk_overlap', 
    aggfunc='mean'
)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.heatmap(faithfulness_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Average Faithfulness by Chunk Size and Overlap')
plt.xlabel('Chunk Overlap')
plt.ylabel('Chunk Size')

plt.subplot(2, 1, 2)
sns.heatmap(relevancy_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Average Relevancy by Chunk Size and Overlap')
plt.xlabel('Chunk Overlap')
plt.ylabel('Chunk Size')
plt.tight_layout()
plt.savefig('plots/heatmap_metrics.png')

# 5. Interactive 3D plots for chunk_size, chunk_overlap, k vs metrics
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Average by chunk_size, chunk_overlap, k combination
grouped_data = df.groupby(['chunk_size', 'chunk_overlap', 'k']).agg({
    'faithfulness': 'mean',
    'relevancy': 'mean'
}).reset_index()

# Create 3D plots
fig = plt.figure(figsize=(15, 7))

# Faithfulness 3D plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
p1 = ax1.scatter(
    grouped_data['chunk_size'], 
    grouped_data['chunk_overlap'], 
    grouped_data['k'], 
    c=grouped_data['faithfulness'],
    s=50,
    cmap='viridis'
)
ax1.set_xlabel('Chunk Size')
ax1.set_ylabel('Chunk Overlap')
ax1.set_zlabel('k')
ax1.set_title('Faithfulness by Parameters')
fig.colorbar(p1, ax=ax1, label='Faithfulness Score')

# Relevancy 3D plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
p2 = ax2.scatter(
    grouped_data['chunk_size'], 
    grouped_data['chunk_overlap'], 
    grouped_data['k'], 
    c=grouped_data['relevancy'],
    s=50,
    cmap='plasma'
)
ax2.set_xlabel('Chunk Size')
ax2.set_ylabel('Chunk Overlap')
ax2.set_zlabel('k')
ax2.set_title('Relevancy by Parameters')
fig.colorbar(p2, ax=ax2, label='Relevancy Score')

plt.tight_layout()
plt.savefig('plots/3d_parameter_space.png')

# 6. User-specific patterns
plt.figure(figsize=(15, 10))
user_ids = sorted(df['user_id'].unique())
n_users = len(user_ids)
cols = 3
rows = (n_users + cols - 1) // cols

for i, user_id in enumerate(user_ids):
    user_df = df[df['user_id'] == user_id]
    plt.subplot(rows, cols, i+1)
    
    x = user_df['chunk_size']
    y = user_df['chunk_overlap']
    colors = user_df['faithfulness']
    sizes = user_df['relevancy'] * 100
    
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='coolwarm')
    plt.colorbar(label='Faithfulness')
    plt.title(f'User {user_id}')
    plt.xlabel('Chunk Size')
    plt.ylabel('Chunk Overlap')
    
plt.tight_layout()
plt.savefig('plots/user_patterns.png')

# 7. Line plots for each parameter's effect
# For chunk size
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
for size in sorted(df['chunk_size'].unique()):
    subset = df[df['chunk_size'] == size]
    avg_by_k = subset.groupby('k')['faithfulness'].mean()
    plt.plot(avg_by_k.index, avg_by_k.values, marker='o', label=f'Size: {size}')
plt.title('Chunk Size Effect on Faithfulness by k')
plt.xlabel('k Value')
plt.ylabel('Avg Faithfulness')
plt.legend()

plt.subplot(1, 2, 2)
for size in sorted(df['chunk_size'].unique()):
    subset = df[df['chunk_size'] == size]
    avg_by_k = subset.groupby('k')['relevancy'].mean()
    plt.plot(avg_by_k.index, avg_by_k.values, marker='o', label=f'Size: {size}')
plt.title('Chunk Size Effect on Relevancy by k')
plt.xlabel('k Value')
plt.ylabel('Avg Relevancy')
plt.legend()
plt.tight_layout()
plt.savefig('plots/chunk_size_effect.png')

# For chunk overlap
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
for overlap in sorted(df['chunk_overlap'].unique()):
    subset = df[df['chunk_overlap'] == overlap]
    avg_by_k = subset.groupby('k')['faithfulness'].mean()
    plt.plot(avg_by_k.index, avg_by_k.values, marker='o', label=f'Overlap: {overlap}')
plt.title('Chunk Overlap Effect on Faithfulness by k')
plt.xlabel('k Value')
plt.ylabel('Avg Faithfulness')
plt.legend()

plt.subplot(1, 2, 2)
for overlap in sorted(df['chunk_overlap'].unique()):
    subset = df[df['chunk_overlap'] == overlap]
    avg_by_k = subset.groupby('k')['relevancy'].mean()
    plt.plot(avg_by_k.index, avg_by_k.values, marker='o', label=f'Overlap: {overlap}')
plt.title('Chunk Overlap Effect on Relevancy by k')
plt.xlabel('k Value')
plt.ylabel('Avg Relevancy')
plt.legend()
plt.tight_layout()
plt.savefig('plots/chunk_overlap_effect.png')

print("All plots have been saved to the 'plots' directory.") 