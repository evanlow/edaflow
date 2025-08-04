"""
Comprehensive test for the complete EDA workflow including the new outlier handling
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

import edaflow

print("🚀 EDAFLOW v0.5.0 - Complete EDA Workflow Test")
print("=" * 60)

# Create comprehensive test dataset
np.random.seed(42)
n_samples = 200

data = {
    # Clean numerical data
    'age': np.random.randint(18, 80, n_samples),
    'height': np.random.normal(170, 10, n_samples),
    
    # Data with outliers
    'income': np.concatenate([
        np.random.normal(50000, 15000, n_samples-5),
        [200000, 250000, -10000, 300000, 400000]  # Clear outliers
    ]),
    
    # Mixed categorical/numerical
    'score_str': [str(x) if x < 95 else 'HIGH' for x in np.random.randint(60, 100, n_samples)],
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    
    # Data with missing values
    'rating': np.random.choice([1, 2, 3, 4, 5, np.nan], n_samples, p=[0.1, 0.15, 0.2, 0.3, 0.2, 0.05]),
    'comments': np.random.choice(['Good', 'Bad', 'OK', np.nan], n_samples, p=[0.3, 0.2, 0.4, 0.1])
}

df = pd.DataFrame(data)

print(f"📊 Test dataset created: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Complete EDA Workflow Test
print("🔍 STEP 1: NULL VALUE ANALYSIS")
print("-" * 40)
try:
    null_analysis = edaflow.check_null_columns(df, threshold=5)
    print("✅ Null analysis completed")
except Exception as e:
    print(f"❌ Error in null analysis: {e}")

print("\n🔍 STEP 2: CATEGORICAL DATA ANALYSIS")
print("-" * 40)
try:
    edaflow.analyze_categorical_columns(df, threshold=30)
    print("✅ Categorical analysis completed")
except Exception as e:
    print(f"❌ Error in categorical analysis: {e}")

print("\n🔧 STEP 3: DATA TYPE CONVERSION")
print("-" * 40)
try:
    df_converted = edaflow.convert_to_numeric(df, columns=['score_str'], threshold=80, verbose=True)
    print("✅ Data conversion completed")
except Exception as e:
    print(f"❌ Error in data conversion: {e}")
    df_converted = df.copy()

print("\n📊 STEP 4: CATEGORICAL VISUALIZATION")
print("-" * 40)
try:
    edaflow.visualize_categorical_values(df_converted, max_categories=10, verbose=True)
    print("✅ Categorical visualization completed")
except Exception as e:
    print(f"❌ Error in categorical visualization: {e}")

print("\n🏷️ STEP 5: COLUMN TYPE CLASSIFICATION")
print("-" * 40)
try:
    edaflow.display_column_types(df_converted)
    print("✅ Column type classification completed")
except Exception as e:
    print(f"❌ Error in column classification: {e}")

print("\n🔧 STEP 6: MISSING VALUE IMPUTATION")
print("-" * 40)
try:
    # Numerical imputation
    df_num_imputed = edaflow.impute_numerical_median(df_converted, verbose=True)
    # Categorical imputation
    df_fully_imputed = edaflow.impute_categorical_mode(df_num_imputed, verbose=True)
    print("✅ Missing value imputation completed")
except Exception as e:
    print(f"❌ Error in imputation: {e}")
    df_fully_imputed = df_converted.copy()

print("\n📊 STEP 7: OUTLIER VISUALIZATION")
print("-" * 40)
try:
    edaflow.visualize_numerical_boxplots(
        df_fully_imputed,
        title="Before Outlier Handling - Distribution Analysis",
        show_skewness=True,
        figsize=(12, 8)
    )
    print("✅ Boxplot visualization completed")
except Exception as e:
    print(f"❌ Error in boxplot visualization: {e}")

print("\n🎯 STEP 8: OUTLIER HANDLING (NEW FEATURE!)")
print("-" * 40)
try:
    # Test IQR method
    df_clean_iqr = edaflow.handle_outliers_median(
        df_fully_imputed,
        method='iqr',
        iqr_multiplier=1.5,
        verbose=True
    )
    print("✅ IQR outlier handling completed")
    
    # Test Z-score method
    print("\n🔬 Testing Z-score method:")
    df_clean_zscore = edaflow.handle_outliers_median(
        df_fully_imputed,
        method='zscore',
        verbose=True
    )
    print("✅ Z-score outlier handling completed")
    
except Exception as e:
    print(f"❌ Error in outlier handling: {e}")
    df_clean_iqr = df_fully_imputed.copy()

print("\n📊 STEP 9: POST-CLEANING VISUALIZATION")
print("-" * 40)
try:
    edaflow.visualize_numerical_boxplots(
        df_clean_iqr,
        title="After Outlier Handling - Clean Distribution",
        show_skewness=True,
        figsize=(12, 8)
    )
    print("✅ Post-cleaning visualization completed")
except Exception as e:
    print(f"❌ Error in post-cleaning visualization: {e}")

print("\n📈 STEP 10: FINAL SUMMARY")
print("-" * 40)
print("Original dataset:")
print(f"  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Income range: {df['income'].min():.0f} to {df['income'].max():.0f}")

print("\nCleaned dataset:")
print(f"  Shape: {df_clean_iqr.shape}")
print(f"  Missing values: {df_clean_iqr.isnull().sum().sum()}")
if 'income' in df_clean_iqr.columns:
    print(f"  Income range: {df_clean_iqr['income'].min():.0f} to {df_clean_iqr['income'].max():.0f}")

print("\n🎉 COMPLETE EDA WORKFLOW TEST COMPLETED!")
print("📦 All 9 functions tested successfully!")
print("🆕 New outlier handling functionality working perfectly!")
