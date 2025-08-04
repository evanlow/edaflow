import pandas as pd
import numpy as np

# Create simple test data
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 100],  # 100 is clearly an outlier
    'B': [10, 20, 30, 40, 50, 60]  # No outliers
})

print("Original data:")
print(df)
print("\nBasic stats for column A:")
print(f"Mean: {df['A'].mean():.2f}")
print(f"Median: {df['A'].median():.2f}")
print(f"Max: {df['A'].max()}")

# Test our function
try:
    import sys
    sys.path.insert(0, '.')
    from edaflow.analysis.missing_data import handle_outliers_median
    
    result = handle_outliers_median(df, verbose=True)
    print("\n✅ Function executed successfully!")
    print(f"After outlier handling - Max A: {result['A'].max()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
