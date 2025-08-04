"""
Test script for the new handle_outliers_median function
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import edaflow
sys.path.insert(0, os.getcwd())

import edaflow

# Create sample data with clear outliers
np.random.seed(42)
data = {
    'normal_col': np.random.normal(50, 10, 100),  # Normal distribution
    'outlier_col': np.concatenate([
        np.random.normal(20, 5, 95),  # Most data around 20
        [100, 150, -50, 200, -100]   # Clear outliers
    ]),
    'no_outliers': np.random.normal(100, 15, 100),  # Clean data
    'categorical': ['A', 'B', 'C'] * 33 + ['A']  # Non-numerical column
}

df = pd.DataFrame(data)

print("🚀 Testing handle_outliers_median function")
print("=" * 50)

print("📊 Original data summary:")
print(df.describe())
print()

print("📈 Step 1: Visualize outliers with boxplots")
try:
    edaflow.visualize_numerical_boxplots(df, title="Original Data with Outliers")
    print("✅ Boxplot visualization successful")
except Exception as e:
    print(f"❌ Error in boxplot visualization: {e}")

print("\n🔧 Step 2: Handle outliers using IQR method")
try:
    df_clean_iqr = edaflow.handle_outliers_median(df, method='iqr', verbose=True)
    print("✅ IQR outlier handling successful")
except Exception as e:
    print(f"❌ Error in IQR outlier handling: {e}")

print("\n🔧 Step 3: Handle outliers using Z-score method")
try:
    df_clean_zscore = edaflow.handle_outliers_median(df, method='zscore', verbose=True)
    print("✅ Z-score outlier handling successful")
except Exception as e:
    print(f"❌ Error in Z-score outlier handling: {e}")

print("\n🔧 Step 4: Handle outliers on specific column only")
try:
    df_clean_specific = edaflow.handle_outliers_median(df, columns=['outlier_col'], verbose=True)
    print("✅ Specific column outlier handling successful")
except Exception as e:
    print(f"❌ Error in specific column outlier handling: {e}")

print("\n📊 Comparison of results:")
print("Original outlier_col stats:")
print(f"  Min: {df['outlier_col'].min():.2f}")
print(f"  Max: {df['outlier_col'].max():.2f}")
print(f"  Mean: {df['outlier_col'].mean():.2f}")
print(f"  Median: {df['outlier_col'].median():.2f}")

if 'df_clean_iqr' in locals():
    print("After IQR outlier handling:")
    print(f"  Min: {df_clean_iqr['outlier_col'].min():.2f}")
    print(f"  Max: {df_clean_iqr['outlier_col'].max():.2f}")
    print(f"  Mean: {df_clean_iqr['outlier_col'].mean():.2f}")
    print(f"  Median: {df_clean_iqr['outlier_col'].median():.2f}")

print("\n🎉 Testing completed!")
